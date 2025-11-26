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
    lr_str = f"{training_config['learning_rate']:.0e}".replace('-0', '-')  # Format like 1e-5
    run_name = f"train-d{training_config['training_dataset_size']}-v{training_config['validation_dataset_size']}-g{training_config['groups_per_step']}-r{training_config['rollouts_per_group']}-lr{lr_str}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    run = wandb.init(
        project=training_config["project"],
        name=run_name,
        config=training_config,
        job_type=training_config.get("wandb_job_type", "train"),
    )
    
    # Declare the model
    model = art.TrainableModel(
        name=run.config.model_name,
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
        
        # Explicitly log training metrics together for scatter plot compatibility
        # Calculate aggregate metrics from judged_groups
        all_train_trajectories = [t for g in judged_groups for t in g.trajectories]
        # Get original finished trajectories for completion token counts
        all_finished_train_trajectories = [t for g in finished_train_groups for t in g.trajectories]
        
        if all_train_trajectories:
            # Calculate reward statistics
            train_rewards = [t.reward for t in all_train_trajectories]
            avg_train_reward = sum(train_rewards) / len(train_rewards)
            reward_std_dev = (sum((r - avg_train_reward) ** 2 for r in train_rewards) / len(train_rewards)) ** 0.5
            
            avg_train_correct = sum(t.metrics.get("correct", 0.0) for t in all_train_trajectories) / len(all_train_trajectories)
            avg_train_retrieved_correct_sources = sum(t.metrics.get("retrieved_correct_sources", 0.0) for t in all_train_trajectories) / len(all_train_trajectories)
            avg_train_tool_optimal_rate = sum(t.metrics.get("tool_optimal_rate", 0.0) for t in all_train_trajectories) / len(all_train_trajectories)
            
            # Calculate total completion tokens from the original finished trajectories
            total_train_completion_tokens = sum(
                t.metadata.get("completion_tokens", 0) if hasattr(t, 'metadata') else 0
                for t in all_finished_train_trajectories
            )
            
            # Extract training step metrics (loss, grad_norm, entropy) if available
            train_metrics = {
                "train/reward": avg_train_reward,
                "train/ruler_score": avg_train_reward,
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
            
            # Apply RULER scoring to validation groups to get rewards
            judged_validation_groups = []
            for group in finished_validation_groups:
                judged_group = await ruler_score_group(group, run.config.ruler_judge_model, debug=True)
                judged_validation_groups.append(judged_group)

            await model.log(
                judged_validation_groups,
                split="val"
            )
            
            # Explicitly log validation metrics together for scatter plot compatibility
            all_val_trajectories = [t for g in judged_validation_groups for t in g.trajectories]
            # Get original finished trajectories for completion token counts
            all_finished_val_trajectories = [t for g in finished_validation_groups for t in g.trajectories]
            
            if all_val_trajectories:
                avg_val_reward = sum(t.reward for t in all_val_trajectories) / len(all_val_trajectories)
                avg_val_correct = sum(t.metrics.get("correct", 0.0) for t in all_val_trajectories) / len(all_val_trajectories)
                avg_val_retrieved_correct_sources = sum(t.metrics.get("retrieved_correct_sources", 0.0) for t in all_val_trajectories) / len(all_val_trajectories)
                avg_val_tool_optimal_rate = sum(t.metrics.get("tool_optimal_rate", 0.0) for t in all_val_trajectories) / len(all_val_trajectories)
                
                # Calculate total completion tokens from the original finished validation trajectories
                total_val_completion_tokens = sum(
                    t.metadata.get("completion_tokens", 0) if hasattr(t, 'metadata') else 0 
                    for t in all_finished_val_trajectories
                )
                
                # Log all validation metrics together in the same step
                wandb.log({
                    "val/reward": avg_val_reward,
                    "val/ruler_score": avg_val_reward,  # Add explicit RULER score logging
                    "val/correct": avg_val_correct,
                    "val/retrieved_correct_sources": avg_val_retrieved_correct_sources,
                    "val/tool_optimal_rate": avg_val_tool_optimal_rate,
                    "val/completion_tokens": total_val_completion_tokens,
                }, step=batch.step)
                
                # Create scatter plots for correlation analysis with per-trajectory data
                # Build data table with individual trajectory metrics
                scatter_data = []
                for judged_traj, finished_traj in zip(all_val_trajectories, all_finished_val_trajectories):
                    ruler_score = judged_traj.reward
                    correct = judged_traj.metrics.get("correct", 0.0)
                    retrieved_sources = judged_traj.metrics.get("retrieved_correct_sources", 0.0)
                    tool_optimal = judged_traj.metrics.get("tool_optimal_rate", 0.0)
                    # Get completion tokens from the original finished trajectory
                    completion_tokens = finished_traj.metadata.get("completion_tokens", 0) if hasattr(finished_traj, 'metadata') else 0
                    
                    scatter_data.append([
                        ruler_score, correct, retrieved_sources, tool_optimal, completion_tokens
                    ])
                
                scatter_columns = ["ruler_score", "correct", "retrieved_correct_sources", "tool_optimal_rate", "completion_tokens"]
                
                # Helper function to create bar chart comparing correct vs incorrect
                def create_correctness_bar_chart(metric_col, metric_label, title):
                    # Group data by correct (1.0) vs incorrect (0.0)
                    col_index_map = {col: i for i, col in enumerate(scatter_columns)}
                    correct_idx = col_index_map["correct"]
                    metric_idx = col_index_map[metric_col]
                    
                    correct_values = []
                    incorrect_values = []
                    
                    for row in scatter_data:
                        if row[correct_idx] == 1.0:
                            correct_values.append(row[metric_idx])
                        else:
                            incorrect_values.append(row[metric_idx])
                    
                    # Calculate averages
                    avg_correct = sum(correct_values) / len(correct_values) if correct_values else 0
                    avg_incorrect = sum(incorrect_values) / len(incorrect_values) if incorrect_values else 0
                    
                    # Create bar chart data
                    bar_data = [
                        {"category": "Correct", "value": avg_correct, "count": len(correct_values)},
                        {"category": "Incorrect", "value": avg_incorrect, "count": len(incorrect_values)}
                    ]
                    
                    # Build Vega-Lite bar chart specification
                    vega_spec = {
                        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                        "title": title,
                        "width": "container",
                        "height": "container",
                        "data": {"values": bar_data},
                        "mark": {
                            "type": "bar",
                            "color": "#3b82f6",
                            "opacity": 0.8
                        },
                        "encoding": {
                            "x": {
                                "field": "category",
                                "type": "nominal",
                                "axis": {"title": "Prediction Result", "labelAngle": 0}
                            },
                            "y": {
                                "field": "value",
                                "type": "quantitative",
                                "axis": {"title": metric_label}
                            },
                            "tooltip": [
                                {"field": "category", "type": "nominal", "title": "Category"},
                                {"field": "value", "type": "quantitative", "format": ".3f", "title": metric_label},
                                {"field": "count", "type": "quantitative", "title": "Count"}
                            ]
                        }
                    }
                    
                    # Return HTML with embedded Vega-Lite spec
                    html = f"""
                    <html>
                    <head>
                        <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
                        <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
                        <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
                    </head>
                    <body>
                        <div id="vis"></div>
                        <script type="text/javascript">
                            var spec = {json.dumps(vega_spec)};
                            vegaEmbed('#vis', spec);
                        </script>
                    </body>
                    </html>
                    """
                    return wandb.Html(html)
                
                def create_four_quadrant_heatmap(title):
                    """Create a 2x2 heatmap showing correctness vs retrieved sources"""
                    # Get column indices
                    col_index_map = {col: i for i, col in enumerate(scatter_columns)}
                    correct_idx = col_index_map["correct"]
                    retrieved_idx = col_index_map["retrieved_correct_sources"]
                    
                    # Count cases in each quadrant
                    correct_retrieved = 0
                    correct_not_retrieved = 0
                    incorrect_retrieved = 0
                    incorrect_not_retrieved = 0
                    
                    for row in scatter_data:
                        is_correct = row[correct_idx] == 1.0
                        has_retrieved = row[retrieved_idx] > 0
                        
                        if is_correct and has_retrieved:
                            correct_retrieved += 1
                        elif is_correct and not has_retrieved:
                            correct_not_retrieved += 1
                        elif not is_correct and has_retrieved:
                            incorrect_retrieved += 1
                        else:
                            incorrect_not_retrieved += 1
                    
                    # Create heatmap data - 2x2 grid
                    heatmap_data = [
                        {"Correctness": "Correct", "Retrieved": "Retrieved", "count": correct_retrieved},
                        {"Correctness": "Correct", "Retrieved": "Not Retrieved", "count": correct_not_retrieved},
                        {"Correctness": "Incorrect", "Retrieved": "Retrieved", "count": incorrect_retrieved},
                        {"Correctness": "Incorrect", "Retrieved": "Not Retrieved", "count": incorrect_not_retrieved}
                    ]
                    
                    # Build Vega-Lite heatmap specification
                    vega_spec = {
                        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                        "title": title,
                        "width": "container",
                        "height": "container",
                        "data": {"values": heatmap_data},
                        "mark": "rect",
                        "encoding": {
                            "x": {
                                "field": "Retrieved",
                                "type": "nominal",
                                "axis": {"title": "Retrieved Correct Sources", "labelAngle": 0}
                            },
                            "y": {
                                "field": "Correctness",
                                "type": "nominal",
                                "axis": {"title": "Answer Correctness"}
                            },
                            "color": {
                                "field": "count",
                                "type": "quantitative",
                                "scale": {"scheme": "blues"},
                                "legend": {"title": "Count"}
                            },
                            "tooltip": [
                                {"field": "Correctness", "type": "nominal", "title": "Correctness"},
                                {"field": "Retrieved", "type": "nominal", "title": "Retrieved Sources"},
                                {"field": "count", "type": "quantitative", "title": "Count"}
                            ]
                        }
                    }
                    
                    # Return HTML with embedded Vega-Lite spec
                    html = f"""
                    <html>
                    <head>
                        <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
                        <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
                        <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
                    </head>
                    <body>
                        <div id="vis"></div>
                        <script type="text/javascript">
                            var spec = {json.dumps(vega_spec)};
                            vegaEmbed('#vis', spec);
                        </script>
                    </body>
                    </html>
                    """
                    return wandb.Html(html)
                
                # Correctness Correlations panel - Bar charts comparing correct vs incorrect
                wandb.log({
                    "correctness_correlations/correct_vs_tool_optimal": create_correctness_bar_chart(
                        "tool_optimal_rate",
                        "Average Tool Optimal Rate",
                        "Tool Optimal Rate: Correct vs Incorrect"
                    ),
                    "correctness_correlations/correct_vs_retrieved_sources": create_four_quadrant_heatmap(
                        "Correctness vs Retrieved Sources Distribution"
                    ),
                    "correctness_correlations/correct_vs_completion_tokens": create_correctness_bar_chart(
                        "completion_tokens",
                        "Average Completion Tokens",
                        "Completion Tokens: Correct vs Incorrect"
                    ),
                    "correctness_correlations/correct_vs_ruler_score": create_correctness_bar_chart(
                        "ruler_score",
                        "Average RULER Score",
                        "RULER Score: Correct vs Incorrect"
                    ),
                }, step=batch.step)
            
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
                    "ruler_reward",
                    "retrieved_correct_sources",
                    "tool_optimal_rate",
                    "tool_reasoning",
                ]
            )
            
            for scenario, group, finished_group in zip(validation_scenarios, judged_validation_groups, finished_validation_groups):
                # Get the first (and only) trajectory from the group
                if len(group.trajectories) > 0:
                    traj = group.trajectories[0]  # For RULER rewards and metrics
                    finished_traj = finished_group.trajectories[0]  # For original trajectory data
                    
                    # Extract metrics for this step - use finished_traj for final_answer
                    model_output = finished_traj.final_answer.answer if finished_traj.final_answer else ""
                    source_ids = str(finished_traj.final_answer.source_ids) if finished_traj.final_answer else "[]"
                    
                    # Extract tool reasoning from finished trajectory
                    if hasattr(finished_traj, 'tool_evaluations') and finished_traj.tool_evaluations:
                        tool_reasoning_parts = []
                        for i, eval in enumerate(finished_traj.tool_evaluations, 1):
                            tool_reasoning_parts.append(
                                f"[{i}] {eval['tool_name']} ({eval['label']}): {eval['reasoning']}"
                            )
                        tool_reasoning = "\n\n".join(tool_reasoning_parts)
                    else:
                        tool_reasoning = ""
                    
                    judge_correct = traj.metrics.get("correct", 0.0)
                    judge_reasoning = traj.metadata.get("judge_reasoning", "")
                    ruler_reward = traj.reward
                    retrieved_correct_sources = traj.metrics.get("retrieved_correct_sources", 0.0)
                    tool_optimal_rate = traj.metrics.get("tool_optimal_rate", 0.0)
                else:
                    # Handle case where trajectory failed
                    model_output = ""
                    source_ids = "[]"
                    tool_reasoning = ""
                    judge_correct = 0.0
                    judge_reasoning = ""
                    ruler_reward = 0.0
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
                    ruler_reward,
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

