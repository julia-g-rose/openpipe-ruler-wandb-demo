"""
Unified script to save Enron email scenarios to all W&B formats.

This script combines the functionality of:
- save_artifacts_to_wandb.py: Saves scenarios as W&B artifacts
- save_datasets_to_weave.py: Publishes scenarios as Weave datasets
- log_table_to_run.py: Logs scenarios as W&B tables for visualization

Run this script once to create all dataset objects in a single W&B run.
"""
import json

from dotenv import load_dotenv
import wandb
import weave

from enron_helpers import load_scenarios
from helpers import load_config

# Load environment variables from .env file
load_dotenv()


def main():
    """Save Enron email scenarios to W&B Artifacts, Weave Datasets, and W&B Tables"""
    # Load configuration from config.yaml
    config = load_config()
    project_name = config["project"]
    
    # Initialize W&B run (this creates a persistent run)
    run = wandb.init(
        project=project_name,
        job_type="dataset-setup",
        tags=["enron", "email-scenarios", "dataset", "artifact", "weave", "table"],
        name="setup-all-datasets",
        config=config
    )
    
    # Initialize Weave (this will authenticate with W&B)
    weave.init(project_name)
    
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
    
    # Create and log W&B Artifacts
    
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
    run.log_artifact(training_artifact)
    
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
    run.log_artifact(validation_artifact)
    
    # Publish to Weave as datasets
    
    training_weave_dataset = weave.Dataset(
        name="enron-training-scenarios",
        rows=training_data
    )
    weave.publish(training_weave_dataset)
    
    validation_weave_dataset = weave.Dataset(
        name="enron-validation-scenarios",
        rows=validation_data
    )
    weave.publish(validation_weave_dataset)
    
    # Log sample scenarios as W&B Tables for visualization
    
    # Training scenarios table (first 10)
    training_table_data = []
    for scenario in training_scenarios[:10]:
        training_table_data.append([
            scenario.id,
            scenario.split,
            scenario.question[:100] + "..." if len(scenario.question) > 100 else scenario.question,
            scenario.answer[:100] + "..." if len(scenario.answer) > 100 else scenario.answer,
            scenario.inbox_address,
            len(scenario.message_ids),
            scenario.how_realistic
        ])
    
    training_table = wandb.Table(
        columns=["ID", "Split", "Question", "Answer", "Inbox", "Num Messages", "Realism Score"],
        data=training_table_data
    )
    run.log({"training_scenarios_sample": training_table})
    
    # Validation scenarios table (all)
    validation_table_data = []
    for scenario in validation_scenarios:
        validation_table_data.append([
            scenario.id,
            scenario.split,
            scenario.question[:100] + "..." if len(scenario.question) > 100 else scenario.question,
            scenario.answer[:100] + "..." if len(scenario.answer) > 100 else scenario.answer,
            scenario.inbox_address,
            len(scenario.message_ids),
            scenario.how_realistic
        ])
    
    validation_table = wandb.Table(
        columns=["ID", "Split", "Question", "Answer", "Inbox", "Num Messages", "Realism Score"],
        data=validation_table_data
    )
    run.log({"validation_scenarios_full": validation_table})
    
    # Finish the run
    run.finish()
    
    return training_scenarios, validation_scenarios


if __name__ == "__main__":
    main()

