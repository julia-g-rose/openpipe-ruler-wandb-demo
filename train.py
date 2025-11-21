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
    
    print(f"Loaded configuration from {config_path}")
    
    # Set random seed for reproducibility
    random.seed(training_config.get("random_seed", 42))
    
    # Initialize W&B run with config
    run = wandb.init(
        project=training_config["project"],
        name=training_config.get("wandb_run_name", "email-agent-training"),
        config=training_config,
        job_type=training_config.get("wandb_job_type", "train"),
    )
    
    print(f"W&B run initialized: {run.url}")
    print(f"Training configuration: {dict(run.config)}")
    
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

    print("Model registered and Weave initialized!")

    # Load training dataset from W&B artifact
    print(f"Loading training dataset artifact: {run.config.training_dataset_artifact}")
    training_artifact = run.use_artifact(run.config.training_dataset_artifact)
    training_dir = training_artifact.download()
    
    with open(Path(training_dir) / "training_scenarios.json", "r") as f:
        training_data = json.load(f)
    training_scenarios = [Scenario(**scenario) for scenario in training_data]
    print(f"Loaded {len(training_scenarios)} training scenarios")
    
    # Load validation dataset from W&B artifact
    print(f"Loading validation dataset artifact: {run.config.validation_dataset_artifact}")
    validation_artifact = run.use_artifact(run.config.validation_dataset_artifact)
    validation_dir = validation_artifact.download()
    
    with open(Path(validation_dir) / "validation_scenarios.json", "r") as f:
        validation_data = json.load(f)
    validation_scenarios = [Scenario(**scenario) for scenario in validation_data]
    print(f"Loaded {len(validation_scenarios)} validation scenarios")

    # Use iterate_dataset with real training scenarios
    training_iterator = iterate_dataset(
        training_scenarios,
        groups_per_step=run.config.groups_per_step,
        num_epochs=run.config.num_epochs,
        initial_step=await model.get_step(),
    )

    # Main training loop
    for batch in training_iterator:
        print(
            f"Training step {batch.step}, epoch {batch.epoch}, epoch step {batch.epoch_step}"
        )
        print(f"Batch contains {len(batch.items)} scenarios")

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

        # Run validation at specified intervals
        if batch.step % run.config.validation_step_interval == 0:
            print(f"Running validation at step {batch.step}")
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

            await model.log(
                finished_validation_groups,
                split="val"
            )
        
        # Train the model on the judged trajectories
        await model.train(
            judged_groups,
            config=art.TrainConfig(learning_rate=run.config.learning_rate),
        )

        print(f"Completed training step {batch.step}")
        
        # Save checkpoint metadata as W&B artifact if enabled
        if run.config.get("save_checkpoint_artifact", True):
            current_step = await model.get_step()
            checkpoint_artifact = wandb.Artifact(
                name=f"{run.config.model_name}-checkpoint",
                type="model",
                description=f"Model checkpoint at step {current_step}",
                metadata={
                    "step": current_step,
                    "epoch": batch.epoch,
                    "epoch_step": batch.epoch_step,
                    "learning_rate": run.config.learning_rate,
                    "base_model": run.config.base_model,
                    "model_name": run.config.model_name,
                    "ruler_judge_model": run.config.ruler_judge_model,
                    "correctness_judge_model": run.config.correctness_judge_model,
                    "training_dataset": run.config.training_dataset_artifact,
                    "validation_dataset": run.config.validation_dataset_artifact,
                }
            )
            
            # Note: The actual checkpoint files are managed by the ART ServerlessBackend
            # This artifact records checkpoint metadata and training configuration
            run.log_artifact(checkpoint_artifact)
            print(f"Logged checkpoint artifact for step {current_step}")

    print("\n🎉 Training completed successfully!")
    print(f"Model checkpoint available at step {await model.get_step()}")
    
    # Finish the W&B run
    run.finish()
    print("W&B run finished and logged.")


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

