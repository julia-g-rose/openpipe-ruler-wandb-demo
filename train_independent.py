"""
Training script for the email search agent.

This script initializes the model, sets up the training configuration,
and runs the training loop with validation at regular intervals.

Usage:
    python train.py                           # Uses default config.yaml
    python train.py --config custom.yaml      # Uses custom config file
"""
import argparse
import json
import random
import yaml
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

import wandb
import art
from art.serverless.backend import ServerlessBackend
from art.utils import iterate_dataset

from helpers import EmailScenario, rollout, initialize_weave
from enron_helpers import Scenario


# Load environment variables
load_dotenv()


def calculate_independent_reward(trajectory) -> float:
    """
    Calculate independent reward from trajectory metrics.
    
    This combines multiple signals:
    - Correctness: Whether the answer is correct (binary 0/1)
    - Retrieved sources: Whether correct sources were retrieved (binary 0/1)
    - Tool optimality: How well tools were used (continuous 0-1)
    
    Args:
        trajectory: A trajectory with metrics
        
    Returns:
        Combined independent reward score
    """
    correct = trajectory.metrics.get("correct", 0.0)
    retrieved_correct_sources = trajectory.metrics.get("retrieved_correct_sources", 0.0)
    tool_optimal_rate = trajectory.metrics.get("tool_optimal_rate", 0.0)
    
    # Combine metrics with weights
    # Correctness is most important (weight: 0.5)
    # Retrieved sources is also important (weight: 0.3)
    # Tool optimality is a bonus (weight: 0.2)
    independent_reward = (
        0.5 * correct +
        0.3 * retrieved_correct_sources +
        0.2 * tool_optimal_rate
    )
    
    return independent_reward


async def main(config_path: str = "config.yaml"):
    """Main training loop.
    
    Args:
        config_path: Path to the YAML configuration file
    """
    
    # Load configuration from YAML file
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        training_config = yaml.safe_load(f)
    
    # Set random seed for reproducibility
    random.seed(training_config.get("random_seed", 42))
    
    # Initialize W&B run with config
    lr_str = f"{training_config['learning_rate']:.0e}".replace('-0', '-')  # Format like 1e-5
    run_name = f"train-independent-d{training_config['training_dataset_size']}-v{training_config['validation_dataset_size']}-g{training_config['groups_per_step']}-r{training_config['rollouts_per_group']}-lr{lr_str}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    run = wandb.init(
        project=training_config["project"],
        name=run_name,
        config=training_config,
        job_type=training_config.get("wandb_job_type", "train"),
    )
    
    # Declare the model
    model = art.TrainableModel(
        name=run.config.model_name + "-independent",
        project=run.config.project,
        base_model=run.config.base_model,
    )

    # Initialize the server
    # Training and inference will run on Weights & Biases servers
    backend = ServerlessBackend()

    # Register the model with the Serverless Backend (sets up logging, inference, and training)
    await model.register(backend)

    # Initialize Weave for logging
    initialize_weave(model.project)

    # Load training dataset from W&B artifact
    training_artifact = run.use_artifact(run.config.training_dataset_artifact)
    training_dir = training_artifact.download()
    
    with open(Path(training_dir) / "training_scenarios.json", "r") as f:
        training_data = json.load(f)
    training_scenarios = [Scenario(**scenario) for scenario in training_data]
    
    # Load validation dataset from W&B artifact
    validation_artifact = run.use_artifact(run.config.validation_dataset_artifact)
    validation_dir = validation_artifact.download()
    
    with open(Path(validation_dir) / "validation_scenarios.json", "r") as f:
        validation_data = json.load(f)
    validation_scenarios = [Scenario(**scenario) for scenario in validation_data]

    # Fetch the validation table from the validation-table artifact
    try:
        validation_table_artifact = run.use_artifact("validation-table:latest")
        validation_table = validation_table_artifact.get("validation_table")
        print("✓ Successfully loaded validation table from artifact")
    except Exception as e:
        # Fallback: create the table if artifact not found
        print(f"Warning: Could not fetch validation table artifact ({e}). Creating from scratch.")
        validation_table = wandb.Table(
            columns=[
                "ID",
                "Split",
                "Question",
                "Answer",
                "Inbox",
                "Num Messages",
                "Realism Score",
            ]
        )
        
        for scenario in validation_scenarios:
            validation_table.add_data(
                scenario.id,
                scenario.split,
                scenario.question[:100] + "..." if len(scenario.question) > 100 else scenario.question,
                scenario.answer[:100] + "..." if len(scenario.answer) > 100 else scenario.answer,
                scenario.inbox_address,
                len(scenario.message_ids),
                scenario.how_realistic,
            )

    # Use iterate_dataset with real training scenarios
    training_iterator = iterate_dataset(
        training_scenarios,
        groups_per_step=run.config.groups_per_step,
        num_epochs=run.config.num_epochs,
        initial_step=await model.get_step(),
    )

    # Main training loop
    for batch in training_iterator:
        # Create trajectory groups for this batch
        train_groups = []
        for scenario in batch.items:
            train_groups.append(
                art.TrajectoryGroup(
                    (
                        rollout(
                            model, 
                            EmailScenario(step=batch.step, scenario=scenario),
                            correctness_judge_model=run.config.correctness_judge_model
                        )
                        for _ in range(run.config.rollouts_per_group)
                    )
                )
            )

        # Gather all trajectory groups
        finished_train_groups = await art.gather_trajectory_groups(
            train_groups,
            pbar_desc="gather",
            max_exceptions=run.config.rollouts_per_group * len(batch.items),
        )

        # Apply independent rewards to each trajectory (no RULER scoring)
        for group in finished_train_groups:
            for traj in group.trajectories:
                independent_reward = calculate_independent_reward(traj)
                # Set the reward to the independent reward
                traj.reward = independent_reward
                # Store independent reward in metrics for tracking
                traj.metrics["independent_reward"] = independent_reward

        # Train the model on the trajectories with independent rewards
        await model.train(
            finished_train_groups,
            config=art.TrainConfig(learning_rate=run.config.learning_rate),
        )
        
        # Explicitly log training metrics together for scatter plot compatibility
        # Calculate aggregate metrics from finished_train_groups
        all_train_trajectories = [t for g in finished_train_groups for t in g.trajectories]
        
        if all_train_trajectories:
            # Calculate reward statistics
            train_rewards = [t.reward for t in all_train_trajectories]
            avg_train_reward = sum(train_rewards) / len(train_rewards)
            reward_std_dev = (sum((r - avg_train_reward) ** 2 for r in train_rewards) / len(train_rewards)) ** 0.5
            
            # Track independent reward (same as reward in this version)
            avg_train_independent = sum(t.metrics.get("independent_reward", 0.0) for t in all_train_trajectories) / len(all_train_trajectories)
            
            avg_train_correct = sum(t.metrics.get("correct", 0.0) for t in all_train_trajectories) / len(all_train_trajectories)
            avg_train_retrieved_correct_sources = sum(t.metrics.get("retrieved_correct_sources", 0.0) for t in all_train_trajectories) / len(all_train_trajectories)
            avg_train_tool_optimal_rate = sum(t.metrics.get("tool_optimal_rate", 0.0) for t in all_train_trajectories) / len(all_train_trajectories)
            
            # Calculate total completion tokens
            total_train_completion_tokens = sum(
                t.metadata.get("completion_tokens", 0) if hasattr(t, 'metadata') else 0
                for t in all_train_trajectories
            )
            
            # Extract training step metrics (loss, grad_norm, entropy) if available
            train_metrics = {
                "train/reward": avg_train_reward,  # Independent reward only
                "train/independent_reward": avg_train_independent,  # Same as reward
                "train/correct": avg_train_correct,
                "train/retrieved_correct_sources": avg_train_retrieved_correct_sources,
                "train/tool_optimal_rate": avg_train_tool_optimal_rate,
                "train/reward_std_dev": reward_std_dev,
                "train/completion_tokens": total_train_completion_tokens,
            }
            
            # Try to extract loss, grad_norm, entropy from trajectory metadata
            # These are typically added by the ART training step
            if all_train_trajectories and hasattr(all_train_trajectories[0], 'metadata'):
                # Average loss across trajectories if available
                losses = [t.metadata.get("loss") for t in all_train_trajectories if hasattr(t, 'metadata') and t.metadata.get("loss") is not None]
                if losses:
                    train_metrics["train/loss"] = sum(losses) / len(losses)
                
                # Use the most recent grad_norm (typically the same for all trajectories in a batch)
                for t in reversed(all_train_trajectories):
                    if hasattr(t, 'metadata') and t.metadata.get("grad_norm") is not None:
                        train_metrics["train/grad_norm"] = t.metadata["grad_norm"]
                        break
                
                # Average entropy across trajectories if available
                entropies = [t.metadata.get("entropy") for t in all_train_trajectories if hasattr(t, 'metadata') and t.metadata.get("entropy") is not None]
                if entropies:
                    train_metrics["train/entropy"] = sum(entropies) / len(entropies)
            
            # Log all training metrics together in the same step
            wandb.log(train_metrics, step=batch.step)
        
        # Run validation at specified intervals
        if batch.step % run.config.validation_step_interval == 0:
            validation_groups = []
            for scenario in validation_scenarios:
                validation_groups.append(
                    art.TrajectoryGroup([
                        rollout(
                            model, 
                            EmailScenario(step=batch.step, scenario=scenario),
                            correctness_judge_model=run.config.correctness_judge_model
                        )
                    ])
                )

            finished_validation_groups = await art.gather_trajectory_groups(
                validation_groups,
                pbar_desc="validation",
                max_exceptions=run.config.rollouts_per_group * len(validation_scenarios),
            )
            
            # Apply independent rewards to validation groups (no RULER scoring)
            for group in finished_validation_groups:
                for traj in group.trajectories:
                    independent_reward = calculate_independent_reward(traj)
                    # Set the reward to the independent reward
                    traj.reward = independent_reward
                    # Store independent reward in metrics for tracking
                    traj.metrics["independent_reward"] = independent_reward

            await model.log(
                finished_validation_groups,
                split="val"
            )
            
            # Explicitly log validation metrics together for scatter plot compatibility
            all_val_trajectories = [t for g in finished_validation_groups for t in g.trajectories]
            
            if all_val_trajectories:
                avg_val_reward = sum(t.reward for t in all_val_trajectories) / len(all_val_trajectories)
                avg_val_independent = sum(t.metrics.get("independent_reward", 0.0) for t in all_val_trajectories) / len(all_val_trajectories)
                avg_val_correct = sum(t.metrics.get("correct", 0.0) for t in all_val_trajectories) / len(all_val_trajectories)
                avg_val_retrieved_correct_sources = sum(t.metrics.get("retrieved_correct_sources", 0.0) for t in all_val_trajectories) / len(all_val_trajectories)
                avg_val_tool_optimal_rate = sum(t.metrics.get("tool_optimal_rate", 0.0) for t in all_val_trajectories) / len(all_val_trajectories)
                
                # Calculate total completion tokens
                total_val_completion_tokens = sum(
                    t.metadata.get("completion_tokens", 0) if hasattr(t, 'metadata') else 0 
                    for t in all_val_trajectories
                )
                
                # Log all validation metrics together in the same step
                wandb.log({
                    "val/reward": avg_val_reward,  # Independent reward only
                    "val/independent_reward": avg_val_independent,  # Same as reward
                    "val/correct": avg_val_correct,
                    "val/retrieved_correct_sources": avg_val_retrieved_correct_sources,
                    "val/tool_optimal_rate": avg_val_tool_optimal_rate,
                    "val/completion_tokens": total_val_completion_tokens,
                }, step=batch.step)
                
                # Create scatter plots for correlation analysis with per-trajectory data
                # Build data table with individual trajectory metrics
                scatter_data = []
                for traj in all_val_trajectories:
                    independent_reward = traj.reward  # Independent reward only
                    correct = traj.metrics.get("correct", 0.0)
                    retrieved_sources = traj.metrics.get("retrieved_correct_sources", 0.0)
                    tool_optimal = traj.metrics.get("tool_optimal_rate", 0.0)
                    # Get completion tokens
                    completion_tokens = traj.metadata.get("completion_tokens", 0) if hasattr(traj, 'metadata') else 0
                    
                    scatter_data.append([
                        independent_reward, correct, retrieved_sources, tool_optimal, completion_tokens
                    ])
                
                scatter_columns = ["independent_reward", "correct", "retrieved_correct_sources", "tool_optimal_rate", "completion_tokens"]
            
            # Create a new validation table for this step with consistent column names
            # This allows W&B to visualize predictions over time
            step_validation_table = wandb.Table(
                columns=[
                    "ID",
                    "Split",
                    "Question",
                    "Answer",
                    "Inbox",
                    "Num Messages",
                    "Realism Score",
                    "model_output",
                    "model_source_ids",
                    "judge_correct",
                    "judge_reasoning",
                    "independent_reward",
                    "retrieved_correct_sources",
                    "tool_optimal_rate",
                    "tool_reasoning",
                ]
            )
            
            for scenario, group in zip(validation_scenarios, finished_validation_groups):
                # Get the first (and only) trajectory from the group
                if len(group.trajectories) > 0:
                    traj = group.trajectories[0]
                    
                    # Extract metrics for this step
                    model_output = traj.final_answer.answer if traj.final_answer else ""
                    source_ids = str(traj.final_answer.source_ids) if traj.final_answer else "[]"
                    
                    # Extract tool reasoning
                    if hasattr(traj, 'tool_evaluations') and traj.tool_evaluations:
                        tool_reasoning_parts = []
                        for i, eval in enumerate(traj.tool_evaluations, 1):
                            tool_reasoning_parts.append(
                                f"[{i}] {eval['tool_name']} ({eval['label']}): {eval['reasoning']}"
                            )
                        tool_reasoning = "\n\n".join(tool_reasoning_parts)
                    else:
                        tool_reasoning = ""
                    
                    judge_correct = traj.metrics.get("correct", 0.0)
                    judge_reasoning = traj.metadata.get("judge_reasoning", "")
                    independent_reward = traj.reward  # Independent reward only
                    retrieved_correct_sources = traj.metrics.get("retrieved_correct_sources", 0.0)
                    tool_optimal_rate = traj.metrics.get("tool_optimal_rate", 0.0)
                else:
                    # Handle case where trajectory failed
                    model_output = ""
                    source_ids = "[]"
                    tool_reasoning = ""
                    judge_correct = 0.0
                    judge_reasoning = ""
                    independent_reward = 0.0
                    retrieved_correct_sources = 0.0
                    tool_optimal_rate = 0.0
                
                # Add row to the table
                step_validation_table.add_data(
                    scenario.id,
                    scenario.split,
                    scenario.question[:100] + "..." if len(scenario.question) > 100 else scenario.question,
                    scenario.answer[:100] + "..." if len(scenario.answer) > 100 else scenario.answer,
                    scenario.inbox_address,
                    len(scenario.message_ids),
                    scenario.how_realistic,
                    model_output,
                    source_ids,
                    judge_correct,
                    judge_reasoning,
                    independent_reward,
                    retrieved_correct_sources,
                    tool_optimal_rate,
                    tool_reasoning,
                )
            
            # Log the table as an artifact - W&B will track changes over time
            validation_artifact = wandb.Artifact(
                name="validation_predictions",
                type="predictions",
            )
            validation_artifact.add(step_validation_table, "validation_results_2")
            run.log_artifact(validation_artifact)
        
        # Save checkpoint metadata as W&B artifact if enabled
        if run.config.get("save_checkpoint_artifact", True):
            current_step = await model.get_step()
            checkpoint_artifact = wandb.Artifact(
                name=f"{run.config.base_model.replace('/', '-')}-checkpoint",
                type="model",
                description=f"Model checkpoint at step {current_step}",
                metadata={
                    "step": current_step,
                    "epoch": batch.epoch,
                    "epoch_step": batch.epoch_step,
                    "learning_rate": run.config.learning_rate,
                    "base_model": run.config.base_model,
                    "ruler_judge_model": run.config.ruler_judge_model,
                    "correctness_judge_model": run.config.correctness_judge_model,
                    "training_dataset": run.config.training_dataset_artifact,
                    "validation_dataset": run.config.validation_dataset_artifact,
                }
            )
            
            # Note: The actual checkpoint files are managed by the ART ServerlessBackend
            # This artifact records checkpoint metadata and training configuration
            run.log_artifact(checkpoint_artifact)

    # Finish the W&B run
    run.finish()


if __name__ == "__main__":
    import asyncio
    
    parser = argparse.ArgumentParser(
        description="Train the email search agent with specified configuration."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)"
    )
    args = parser.parse_args()
    
    asyncio.run(main(config_path=args.config))

