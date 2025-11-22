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

import wandb
import art
from art.serverless.backend import ServerlessBackend
from art.rewards import ruler_score_group
from art.utils import iterate_dataset

from helpers import EmailScenario, rollout, initialize_weave
from enron_helpers import Scenario


# Load environment variables
load_dotenv()


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
    run = wandb.init(
        project=training_config["project"],
        name=training_config.get("wandb_run_name", "email-agent-training"),
        config=training_config,
        job_type=training_config.get("wandb_job_type", "train"),
    )
    
    # Declare the model
    model = art.TrainableModel(
        name="email-agent-qwen",
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

        # Use RULER to assign relative scores to each trajectory
        judged_groups = []
        for group in finished_train_groups:
            judged_group = await ruler_score_group(group, run.config.ruler_judge_model, debug=True)
            judged_groups.append(judged_group)

        # Train the model on the judged trajectories
        await model.train(
            judged_groups,
            config=art.TrainConfig(learning_rate=run.config.learning_rate),
        )
        
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
            
            # Apply RULER scoring to validation groups to get rewards
            judged_validation_groups = []
            for group in finished_validation_groups:
                judged_group = await ruler_score_group(group, run.config.ruler_judge_model, debug=True)
                judged_validation_groups.append(judged_group)

            await model.log(
                judged_validation_groups,
                split="val"
            )
            
            # Add new columns for this validation step
            step_suffix = f"_step_{batch.step}"
            
            # Collect data for new columns
            model_outputs = []
            source_ids_list = []
            judge_corrects = []
            judge_reasonings = []
            ruler_rewards = []
            retrieved_correct_sources_list = []
            tool_appropriate_rates = []
            
            for scenario, group in zip(validation_scenarios, judged_validation_groups):
                # Get the first (and only) trajectory from the group
                if len(group.trajectories) > 0:
                    traj = group.trajectories[0]
                    
                    # Extract metrics for this step
                    model_outputs.append(traj.final_answer.answer if traj.final_answer else "")
                    source_ids_list.append(str(traj.final_answer.source_ids) if traj.final_answer else "[]")
                    judge_corrects.append(traj.metrics.get("correct", 0.0))
                    judge_reasonings.append(traj.metadata.get("judge_reasoning", ""))
                    ruler_rewards.append(traj.reward)
                    retrieved_correct_sources_list.append(traj.metrics.get("retrieved_correct_sources", 0.0))
                    tool_appropriate_rates.append(traj.metrics.get("tool_appropriate_rate", 0.0))
                else:
                    # Handle case where trajectory failed
                    model_outputs.append("")
                    source_ids_list.append("[]")
                    judge_corrects.append(0.0)
                    judge_reasonings.append("")
                    ruler_rewards.append(0.0)
                    retrieved_correct_sources_list.append(0.0)
                    tool_appropriate_rates.append(0.0)
            
            # Add columns with step-specific names
            validation_table.add_column(f"model_output{step_suffix}", model_outputs)
            validation_table.add_column(f"model_source_ids{step_suffix}", source_ids_list)
            validation_table.add_column(f"judge_correct{step_suffix}", judge_corrects)
            validation_table.add_column(f"judge_reasoning{step_suffix}", judge_reasonings)
            validation_table.add_column(f"ruler_reward{step_suffix}", ruler_rewards)
            validation_table.add_column(f"retrieved_correct_sources{step_suffix}", retrieved_correct_sources_list)
            validation_table.add_column(f"tool_appropriate_rate{step_suffix}", tool_appropriate_rates)
            
            # Log the updated table with a consistent name
            run.log({"validation_results": validation_table})
        
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

