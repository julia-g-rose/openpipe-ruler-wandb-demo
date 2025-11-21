"""
Evaluation script for the email search agent.

This script loads a trained model and evaluates it on test scenarios,
printing detailed output of the agent's reasoning and final answers.

Usage:
    python evaluate.py                           # Uses default config.yaml
    python evaluate.py --config custom.yaml      # Uses custom config file
"""
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

import wandb
import art
from art.serverless.backend import ServerlessBackend

from helpers import EmailScenario, rollout, initialize_weave, print_trajectory, load_config
from enron_helpers import Scenario


# Load environment variables
load_dotenv()


async def evaluate_model(
    model: art.Model, 
    scenarios: list, 
    scenario_name: str = "test",
    correctness_judge_model: str = "openai/gpt-4o"
):
    """
    Evaluate the model on a list of scenarios.
    
    Args:
        model: The ART model to evaluate
        scenarios: List of scenarios to evaluate on
        scenario_name: Name for this set of scenarios (for logging)
        correctness_judge_model: The model to use for judging answer correctness
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on {len(scenarios)} {scenario_name} scenarios")
    print(f"{'='*60}\n")
    
    correct_count = 0
    total_count = len(scenarios)
    
    for idx, scenario in enumerate(scenarios):
        print(f"\n--- Scenario {idx + 1}/{total_count} ---")
        print(f"Scenario ID: {scenario.id}")
        print(f"Question: {scenario.question}")
        print(f"Expected answer: {scenario.answer}")
        print(f"Reference message IDs: {scenario.message_ids}")
        print(f"Inbox: {scenario.inbox_address}")
        print(f"Query date: {scenario.query_date}")
        print("-" * 50)
        
        # Run the rollout function with the trained model
        email_scenario = EmailScenario(step=0, scenario=scenario)
        trajectory = await rollout(model, email_scenario, correctness_judge_model)
        
        # Print the trajectory details
        print_trajectory(trajectory, scenario)
        
        # Check if the answer was correct
        is_correct = trajectory.metrics.get("correct", 0.0) > 0.5
        if is_correct:
            correct_count += 1
            print("✅ CORRECT")
        else:
            print("❌ INCORRECT")
        
        print("\n" + "="*60 + "\n")
    
    # Print summary statistics
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY - {scenario_name.upper()}")
    print(f"{'='*60}")
    print(f"Total scenarios: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"{'='*60}\n")
    
    return {
        "total": total_count,
        "correct": correct_count,
        "accuracy": accuracy
    }


async def main(config_path: str = "config.yaml"):
    """Main evaluation function.
    
    Args:
        config_path: Path to the YAML configuration file
    """
    
    # Load configuration from YAML file
    config = load_config(config_path)
    print(f"Loaded configuration from {config_path}")
    
    # Initialize W&B for artifact loading
    run = wandb.init(
        project=config["project"],
        name="email-agent-evaluation",
        config=config,
        job_type="evaluation",
    )
    
    print(f"W&B run initialized: {run.url}")
    
    # Load training dataset from W&B artifact
    print(f"Loading training dataset artifact: {config['training_dataset_artifact']}")
    training_artifact = run.use_artifact(config["training_dataset_artifact"])
    training_dir = training_artifact.download()
    
    with open(Path(training_dir) / "training_scenarios.json", "r") as f:
        training_data = json.load(f)
    training_scenarios = [Scenario(**scenario) for scenario in training_data]
    print(f"Loaded {len(training_scenarios)} training scenarios")
    
    # Load validation dataset from W&B artifact
    print(f"Loading validation dataset artifact: {config['validation_dataset_artifact']}")
    validation_artifact = run.use_artifact(config["validation_dataset_artifact"])
    validation_dir = validation_artifact.download()
    
    with open(Path(validation_dir) / "validation_scenarios.json", "r") as f:
        validation_data = json.load(f)
    validation_scenarios = [Scenario(**scenario) for scenario in validation_data]
    print(f"Loaded {len(validation_scenarios)} validation scenarios")
    
    # Declare the model (same configuration as training)
    model = art.TrainableModel(
        name=config["model_name"],
        project=config["project"],
        base_model=config["base_model"],
    )

    # Initialize the server
    backend = ServerlessBackend()

    # Register the model with the Serverless Backend
    await model.register(backend)

    # Initialize Weave for logging
    initialize_weave(model.project)

    print("Model loaded and ready for evaluation!")
    print(f"Current model step: {await model.get_step()}")

    # Option 1: Evaluate on a single test scenario (quick test)
    print("\n" + "="*60)
    print("QUICK TEST - Single Scenario")
    print("="*60)
    test_scenario = training_scenarios[0]
    await evaluate_model(
        model, 
        [test_scenario], 
        "quick test",
        correctness_judge_model=config["correctness_judge_model"]
    )

    # Option 2: Evaluate on first 5 validation scenarios
    print("\n" + "="*60)
    print("VALIDATION SET EVALUATION")
    print("="*60)
    val_results = await evaluate_model(
        model, 
        validation_scenarios[:5],  # Adjust number as needed
        "validation",
        correctness_judge_model=config["correctness_judge_model"]
    )

    # Option 3: Evaluate on first 5 training scenarios (to check overfitting)
    print("\n" + "="*60)
    print("TRAINING SET EVALUATION (Check Overfitting)")
    print("="*60)
    train_results = await evaluate_model(
        model,
        training_scenarios[:5],  # Adjust number as needed
        "training",
        correctness_judge_model=config["correctness_judge_model"]
    )

    print("\n🎉 Evaluation completed!")
    print("\nFinal Results:")
    print(f"  Validation Accuracy: {val_results['accuracy']:.2%}")
    print(f"  Training Accuracy: {train_results['accuracy']:.2%}")
    
    # Finish the W&B run
    run.finish()
    print("W&B run finished and logged.")


if __name__ == "__main__":
    import asyncio
    
    parser = argparse.ArgumentParser(
        description="Evaluate the email search agent with specified configuration."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)"
    )
    args = parser.parse_args()
    
    asyncio.run(main(config_path=args.config))

