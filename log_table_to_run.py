import json
import os

from dotenv import load_dotenv
import wandb

from enron_helpers import load_scenarios

# Load environment variables from .env file
load_dotenv()


def main():
    """Log Enron email scenarios as a W&B table for visualization"""
    project_name = os.environ.get("WANDB_PROJECT", "enron-email-search")
    
    # Initialize W&B run (this creates a persistent run)
    run = wandb.init(
        project=project_name,
        job_type="data-visualization",
        tags=["enron", "email-scenarios", "table"],
        name="visualize-enron-scenarios"
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
    
    # Log sample scenario as a table for visualization
    sample_table_data = []
    for scenario in training_scenarios[:5]:  # First 5 training scenarios
        sample_table_data.append([
            scenario.id,
            scenario.question,
            scenario.answer,
            scenario.inbox_address,
            len(scenario.message_ids)
        ])
    
    sample_table = wandb.Table(
        columns=["ID", "Question", "Answer", "Inbox", "Num Messages"],
        data=sample_table_data
    )
    run.log({"sample_scenarios": sample_table})
    
    # Finish the run
    run.finish()
    
    return training_scenarios, validation_scenarios


if __name__ == "__main__":
    main()

