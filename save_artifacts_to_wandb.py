import json

from dotenv import load_dotenv
import wandb

from enron_helpers import load_scenarios
from helpers import load_config

# Load environment variables from .env file
load_dotenv()


def main():
    """Save Enron email scenarios as W&B artifacts"""
    # Load configuration from config.yaml
    config = load_config()
    project_name = config["project"]
    
    # Load training scenarios
    training_scenarios = load_scenarios(
        split="train", 
        limit=config["training_dataset_size"], 
        max_messages=1, 
        shuffle=True, 
        seed=config["dataset_seed"]
    )

    # Load validation scenarios
    validation_scenarios = load_scenarios(
        split="test", 
        limit=config["validation_dataset_size"], 
        max_messages=1, 
        shuffle=True, 
        seed=config["dataset_seed"]
    )

    # Convert scenarios to dictionaries for saving
    training_data = [scenario.model_dump() for scenario in training_scenarios]
    validation_data = [scenario.model_dump() for scenario in validation_scenarios]
    
    # Save scenarios to JSON files
    with open("training_scenarios.json", "w") as f:
        json.dump(training_data, f, indent=2)
    
    with open("validation_scenarios.json", "w") as f:
        json.dump(validation_data, f, indent=2)
    
    # Log to W&B as artifacts (using artifact.save() which creates ephemeral runs)
    # Note: artifact.save() will create temporary runs internally, but they won't persist
    training_artifact = wandb.Artifact(
        name="enron-training-scenarios",
        type="dataset",
        description=f"Training scenarios from Enron email dataset ({len(training_scenarios)} examples)",
        metadata={
            "split": "train",
            "num_scenarios": len(training_scenarios),
            "max_messages": 1,
            "seed": config["dataset_seed"]
        }
    )
    training_artifact.add_file("training_scenarios.json")
    training_artifact.save(project=project_name)
    
    validation_artifact = wandb.Artifact(
        name="enron-validation-scenarios",
        type="dataset",
        description=f"Validation scenarios from Enron email dataset ({len(validation_scenarios)} examples)",
        metadata={
            "split": "test",
            "num_scenarios": len(validation_scenarios),
            "max_messages": 1,
            "seed": config["dataset_seed"]
        }
    )
    validation_artifact.add_file("validation_scenarios.json")
    validation_artifact.save(project=project_name)
    
    return training_scenarios, validation_scenarios


if __name__ == "__main__":
    main()

