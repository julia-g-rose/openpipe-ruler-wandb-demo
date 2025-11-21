import csv

from dotenv import load_dotenv
import wandb

from enron_helpers import load_scenarios
from helpers import load_config

# Load environment variables from .env file
load_dotenv()


def main():
    """Save Enron email scenarios as W&B artifacts (CSV format)"""
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
    
    # Save scenarios to CSV files
    # Define CSV columns
    csv_columns = ["id", "question", "answer", "message_ids", "how_realistic", "inbox_address", "query_date", "split"]
    
    # Write training CSV
    with open("training_scenarios.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in training_data:
            # Convert message_ids list to string
            row["message_ids"] = "|".join(row["message_ids"])
            writer.writerow(row)
    
    # Write validation CSV
    with open("validation_scenarios.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in validation_data:
            # Convert message_ids list to string
            row["message_ids"] = "|".join(row["message_ids"])
            writer.writerow(row)
    
    # Log to W&B as artifacts (using artifact.save() which creates ephemeral runs)
    # Note: artifact.save() will create temporary runs internally, but they won't persist
    training_artifact = wandb.Artifact(
        name="enron-training-scenarios-csv",
        type="dataset",
        description=f"Training scenarios from Enron email dataset ({len(training_scenarios)} examples) in CSV format",
        metadata={
            "split": "train",
            "num_scenarios": len(training_scenarios),
            "max_messages": 1,
            "seed": config["dataset_seed"],
            "format": "csv"
        }
    )
    training_artifact.add_file("training_scenarios.csv")
    training_artifact.save(project=project_name)
    
    validation_artifact = wandb.Artifact(
        name="enron-validation-scenarios-csv",
        type="dataset",
        description=f"Validation scenarios from Enron email dataset ({len(validation_scenarios)} examples) in CSV format",
        metadata={
            "split": "test",
            "num_scenarios": len(validation_scenarios),
            "max_messages": 1,
            "seed": config["dataset_seed"],
            "format": "csv"
        }
    )
    validation_artifact.add_file("validation_scenarios.csv")
    validation_artifact.save(project=project_name)
    
    print(f"✅ CSV artifacts saved to W&B project: {project_name}")
    return training_scenarios, validation_scenarios


if __name__ == "__main__":
    main()

