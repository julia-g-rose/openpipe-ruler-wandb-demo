import json

from dotenv import load_dotenv
import weave

from enron_helpers import load_scenarios
from helpers import load_config

# Load environment variables from .env file
load_dotenv()


def main():
    """Save Enron email scenarios as Weave datasets"""
    # Load configuration from config.yaml
    config = load_config()
    project_name = config["project"]
    
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

    # Convert scenarios to dictionaries for saving
    training_data = [scenario.model_dump() for scenario in training_scenarios]
    validation_data = [scenario.model_dump() for scenario in validation_scenarios]
    
    # Save scenarios to JSON files
    with open("training_scenarios.json", "w") as f:
        json.dump(training_data, f, indent=2)
    
    with open("validation_scenarios.json", "w") as f:
        json.dump(validation_data, f, indent=2)
    
    # Publish to Weave as datasets
    training_dataset = weave.Dataset(
        name="enron-training-scenarios",
        rows=training_data
    )
    weave.publish(training_dataset)
    
    validation_dataset = weave.Dataset(
        name="enron-validation-scenarios",
        rows=validation_data
    )
    weave.publish(validation_dataset)
    
    return training_scenarios, validation_scenarios


if __name__ == "__main__":
    main()

