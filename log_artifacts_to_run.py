import json
import os

from dotenv import load_dotenv
import wandb

from enron_helpers import load_scenarios

# Load environment variables from .env file
load_dotenv()


def main():
    """Save Enron email scenarios as W&B artifacts with a persistent run"""
    project_name = os.environ.get("WANDB_PROJECT", "enron-email-search")
    
    # Initialize W&B run (this creates a persistent run)
    run = wandb.init(
        project=project_name,
        job_type="dataset-creation",
        tags=["enron", "email-scenarios", "dataset"],
        name="upload-enron-scenarios"
    )
    
    # Load training scenarios
    training_scenarios = load_scenarios(
        split="train", limit=50, max_messages=1, shuffle=True, seed=42
    )

    # Load validation scenarios
    validation_scenarios = load_scenarios(
        split="test", limit=20, max_messages=1, shuffle=True, seed=42
    )

    # Log dataset statistics to the run
    run.summary["training_scenarios_count"] = len(training_scenarios)
    run.summary["validation_scenarios_count"] = len(validation_scenarios)
    run.summary["total_scenarios"] = len(training_scenarios) + len(validation_scenarios)

    # Convert scenarios to dictionaries for saving
    training_data = [scenario.model_dump() for scenario in training_scenarios]
    validation_data = [scenario.model_dump() for scenario in validation_scenarios]
    
    # Save scenarios to JSON files
    with open("training_scenarios.json", "w") as f:
        json.dump(training_data, f, indent=2)
    
    with open("validation_scenarios.json", "w") as f:
        json.dump(validation_data, f, indent=2)
    
    # Log to W&B as artifacts (using run.log_artifact() with a persistent run)
    training_artifact = wandb.Artifact(
        name="enron-training-scenarios",
        type="dataset",
        description=f"Training scenarios from Enron email dataset ({len(training_scenarios)} examples)",
        metadata={
            "split": "train",
            "num_scenarios": len(training_scenarios),
            "max_messages": 1,
            "seed": 42
        }
    )
    training_artifact.add_file("training_scenarios.json")
    run.log_artifact(training_artifact)
    
    validation_artifact = wandb.Artifact(
        name="enron-validation-scenarios",
        type="dataset",
        description=f"Validation scenarios from Enron email dataset ({len(validation_scenarios)} examples)",
        metadata={
            "split": "test",
            "num_scenarios": len(validation_scenarios),
            "max_messages": 1,
            "seed": 42
        }
    )
    validation_artifact.add_file("validation_scenarios.json")
    run.log_artifact(validation_artifact)
    
    # Finish the run
    run.finish()
    
    return training_scenarios, validation_scenarios


if __name__ == "__main__":
    main()

