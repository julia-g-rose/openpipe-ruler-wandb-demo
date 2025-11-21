"""
Model comparison script using Weave evaluations and leaderboards.

This script compares three models using the three Weave scorers:
1. Model 1 (configurable, default: gpt-4o-mini)
2. Model 2 (configurable, default: gpt-5)
3. OpenPipe/Qwen base model (before fine-tuning)

It uses Weave's evaluation framework to run all scorers on the validation dataset
and creates a leaderboard for comparison. Results can be visualized with parallel
coordinates in the W&B UI.

Usage:
    python compare_models.py                           # Uses default config.yaml
    python compare_models.py --config custom.yaml      # Uses custom config file
"""
import argparse
import asyncio
import os
from typing import Any
from dotenv import load_dotenv

import weave
from weave.flow import leaderboard
from weave.trace.ref_util import get_ref
import wandb
import art
from art.serverless.backend import ServerlessBackend

from helpers import (
    EmailScenario,
    rollout,
    load_config,
    initialize_weave,
    CorrectnessJudgeScorer,
    SourceRetrievalScorer,
    ToolUsageScorer,
)
from enron_helpers import Scenario


# Load environment variables
load_dotenv()


class OpenAIArtModel:
    """Mock ART model that wraps OpenAI API for compatibility with rollout function.
    
    This class provides the minimal interface required by the rollout function:
    - inference_base_url
    - inference_api_key  
    - get_inference_name()
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.inference_base_url = "https://api.openai.com/v1"
        self.inference_api_key = os.getenv('OPENAI_API_KEY')
    
    def get_inference_name(self) -> str:
        """Return the OpenAI model name to use for inference."""
        return self.model_name


class WeaveModelWrapper(weave.Model):
    """Wrapper to make ART models compatible with Weave evaluations."""
    
    model: Any
    model_name: str
    correctness_judge_model: str
    tool_judge_model: str
    
    @weave.op()
    async def predict(self, scenario: dict) -> dict:
        """Run the model on a scenario and return raw trajectory data for scoring.
        
        Args:
            scenario: Dict containing scenario information
            
        Returns:
            Dict with trajectory data and scenario for scoring
        """
        # Convert dict to Scenario object
        scenario_obj = Scenario(**scenario)
        email_scenario = EmailScenario(step=0, scenario=scenario_obj)
        
        # Run rollout WITHOUT pre-computing scores (we'll do that in the scorers)
        # Note: We still need the rollout to do tool evaluation for ToolUsageScorer
        trajectory = await rollout(
            self.model,
            email_scenario,
            correctness_judge_model=self.correctness_judge_model,
            tool_judge_model=self.tool_judge_model
        )
        
        # Return raw data for scorers to process
        result = {
            "trajectory": trajectory,
            "scenario": scenario_obj,
            "answer": trajectory.final_answer.answer if trajectory.final_answer else "",
            "source_ids": trajectory.final_answer.source_ids if trajectory.final_answer else [],
        }
        
        return result


class OpenAIModelWrapper(weave.Model):
    """Wrapper for OpenAI models to work with Weave evaluations."""
    
    model_name: str
    correctness_judge_model: str
    tool_judge_model: str
    
    @weave.op()
    async def predict(self, scenario: dict) -> dict:
        """Run OpenAI model on a scenario and return raw trajectory data for scoring.
        
        This uses a mock ART model that wraps OpenAI's API,
        then runs the full rollout with tool calling.
        """
        # Convert dict to Scenario object
        scenario_obj = Scenario(**scenario)
        email_scenario = EmailScenario(step=0, scenario=scenario_obj)
        
        # Create a mock ART model that wraps OpenAI
        openai_model = OpenAIArtModel(self.model_name)
        
        # Run rollout WITHOUT pre-computing scores (we'll do that in the scorers)
        trajectory = await rollout(
            openai_model,
            email_scenario,
            correctness_judge_model=self.correctness_judge_model,
            tool_judge_model=self.tool_judge_model
        )
        
        # Return raw data for scorers to process
        result = {
            "trajectory": trajectory,
            "scenario": scenario_obj,
            "answer": trajectory.final_answer.answer if trajectory.final_answer else "",
            "source_ids": trajectory.final_answer.source_ids if trajectory.final_answer else [],
        }
        
        return result


# Weave-compatible scorer wrappers that use the actual scorer classes from helpers.py
# These run the real scoring logic instead of just extracting pre-computed metrics

def create_correctness_scorer(judge_model: str):
    """Factory function to create a correctness scorer with the specified judge model.
    
    Args:
        judge_model: The model to use for judging correctness (from config)
    
    Returns:
        A weave.op decorated async function for scoring correctness
    """
    @weave.op()
    async def score_correctness(model_output: dict) -> dict:
        """Score answer correctness using CorrectnessJudgeScorer from helpers.py.
        
        This uses the actual scoring logic defined in helpers.CorrectnessJudgeScorer
        (line 132) rather than extracting pre-computed metrics.
        """
        trajectory = model_output.get("trajectory")
        scenario = model_output.get("scenario")
        answer = model_output.get("answer", "")
        
        if not scenario:
            return {"correct": 0.0, "reasoning": "Missing scenario data"}
        
        # Initialize and use the actual scorer from helpers.py
        correctness_scorer = CorrectnessJudgeScorer(judge_model=judge_model)
        result = await correctness_scorer.score(
            output=answer,
            question=scenario.question,
            reference_answer=scenario.answer
        )
        return result
    
    return score_correctness


@weave.op()
async def score_source_retrieval(model_output: dict) -> dict:
    """Score source retrieval using SourceRetrievalScorer from helpers.py.
    
    This uses the actual scoring logic defined in helpers.SourceRetrievalScorer
    (line 214) rather than extracting pre-computed metrics.
    """
    scenario = model_output.get("scenario")
    source_ids = model_output.get("source_ids", [])
    
    if not scenario:
        return {
            "source_precision": 0.0,
            "source_recall": 0.0,
            "source_f1": 0.0,
            "retrieved_correct_sources": 0.0
        }
    
    # Initialize and use the actual scorer from helpers.py
    source_scorer = SourceRetrievalScorer()
    result = await source_scorer.score(
        output={"source_ids": source_ids},
        expected_source_ids=scenario.message_ids
    )
    return result


@weave.op()
def score_tool_usage(model_output: dict) -> dict:
    """Score tool usage by aggregating tool evaluations from the trajectory.
    
    Tool evaluations are computed during rollout using helpers.ToolUsageScorer (line 284).
    This function aggregates those evaluations using the same logic as 
    helpers._add_tool_usage_metrics (line 513).
    
    This ensures the leaderboard shows the actual tool evaluation code from helpers.py.
    """
    trajectory = model_output.get("trajectory")
    
    if not trajectory or not trajectory.tool_evaluations:
        return {
            "tool_appropriate_rate": 0.0,
            "tool_optimal_rate": 0.0
        }
    
    # Aggregate tool evaluations using the same logic as _add_tool_usage_metrics
    total = len(trajectory.tool_evaluations)
    appropriate_count = sum(1 for eval in trajectory.tool_evaluations if eval["appropriate"] == 1.0)
    optimal_count = sum(1 for eval in trajectory.tool_evaluations if eval["label"] == "optimal")
    
    return {
        "tool_appropriate_rate": appropriate_count / total if total > 0 else 0.0,
        "tool_optimal_rate": optimal_count / total if total > 0 else 0.0
    }


async def main(config_path: str = "config.yaml"):
    """Main comparison function.
    
    Args:
        config_path: Path to the YAML configuration file
    """
    
    # Load configuration
    config = load_config(config_path)
    
    # Initialize Weave and load existing dataset
    initialize_weave(config["project"])
    
    # Load the validation dataset that was already published to Weave
    dataset = weave.ref("enron-validation-scenarios").get()
    
    # Initialize W&B for logging results
    run = wandb.init(
        project=config["project"],
        name="model-comparison-evaluation",
        config=config,
        job_type="comparison",
    )
    
    # Model 1: Comparison model 1 (from config)
    comparison_model_1 = OpenAIModelWrapper(
        model_name=config["comparison_model_1"],
        correctness_judge_model=config["correctness_judge_model"],
        tool_judge_model=config["tool_judge_model"]
    )
    
    # Model 2: Comparison model 2 (from config)
    comparison_model_2 = OpenAIModelWrapper(
        model_name=config["comparison_model_2"],
        correctness_judge_model=config["correctness_judge_model"],
        tool_judge_model=config["tool_judge_model"]
    )
    
    # Model 3: OpenPipe/Qwen base model (untrained)
    # Use TrainableModel instead of Model to work with ServerlessBackend
    base_model = art.TrainableModel(
        name="qwen-base-comparison",  # Use a unique name for comparison
        project=config["project"],
        base_model=config["base_model"],
    )
    
    backend = ServerlessBackend()
    await base_model.register(backend)
    
    qwen_base = WeaveModelWrapper(
        model=base_model,
        model_name=config["base_model"],
        correctness_judge_model=config["correctness_judge_model"],
        tool_judge_model=config["tool_judge_model"]
    )
    
    # Preprocessing function to convert dataset rows to model input format
    def preprocess_model_input(example: dict) -> dict:
        """Convert dataset example to the format expected by model predict methods."""
        return {"scenario": example}
    
    # Common scorers for all evaluations
    # These use the actual scorer classes from helpers.py
    scorers = [
        create_correctness_scorer(config["correctness_judge_model"]),
        score_source_retrieval,
        score_tool_usage
    ]
    
    # Get model names from config
    comp_model_1_name = config["comparison_model_1"]
    comp_model_2_name = config["comparison_model_2"]
    
    # Create separate evaluation objects for each model
    # This is required for the Leaderboard API - we need to pass the Evaluation objects to get_ref()
    evaluations = [
        weave.Evaluation(
            name=f"{comp_model_1_name}-evaluation",
            dataset=dataset,
            scorers=scorers,
            preprocess_model_input=preprocess_model_input
        ),
        weave.Evaluation(
            name=f"{comp_model_2_name}-evaluation",
            dataset=dataset,
            scorers=scorers,
            preprocess_model_input=preprocess_model_input
        ),
        weave.Evaluation(
            name="qwen-base-evaluation",
            dataset=dataset,
            scorers=scorers,
            preprocess_model_input=preprocess_model_input
        ),
    ]
    
    # Models list
    models = [comparison_model_1, comparison_model_2, qwen_base]
    model_names = [comp_model_1_name, comp_model_2_name, "qwen-base"]
    display_names = [comp_model_1_name, comp_model_2_name, "OpenPipe/Qwen3-14B-Instruct base"]
    
    # Run evaluations and collect predictions
    results = {}
    all_predictions = {}  # Store individual predictions for table creation
    
    for evaluation, model, model_name, display_name in zip(evaluations, models, model_names, display_names):
        print(f"\n🔄 Evaluating {display_name}...")
        
        # Run evaluation
        eval_result = await evaluation.evaluate(
            model,
            __weave={"display_name": display_name}
        )
        results[model_name] = eval_result
        
        # Also collect individual predictions for the detailed table
        print(f"📊 Collecting detailed predictions for {model_name}...")
        model_predictions = []
        for row in dataset.rows:
            try:
                # Run predict on each example
                pred_input = {"scenario": row}
                prediction = await model.predict(row)
                model_predictions.append(prediction)
            except Exception as e:
                print(f"Warning: Prediction failed for model {model_name}: {e}")
                model_predictions.append({
                    "trajectory": None,
                    "scenario": None,
                    "answer": "",
                    "source_ids": []
                })
        
        all_predictions[model_name] = model_predictions
    
    # Create leaderboard summary table
    leaderboard_data = []
    
    for model_name, result in results.items():
        # The result dict directly contains scorer results at the top level
        # No nested 'summary' key - scorer names are the keys
        def get_metric(result_dict, scorer_name, metric_name):
            """Extract metric from Weave evaluation result"""
            scorer_result = result_dict.get(scorer_name, {})
            metric_value = scorer_result.get(metric_name, {})
            if isinstance(metric_value, dict):
                return metric_value.get('mean', 0)
            return metric_value if metric_value else 0
        
        leaderboard_entry = {
            "model": model_name,
            "correctness": get_metric(result, 'score_correctness', 'correct'),
            "source_precision": get_metric(result, 'score_source_retrieval', 'source_precision'),
            "source_recall": get_metric(result, 'score_source_retrieval', 'source_recall'),
            "source_f1": get_metric(result, 'score_source_retrieval', 'source_f1'),
            "tool_appropriate_rate": get_metric(result, 'score_tool_usage', 'tool_appropriate_rate'),
            "tool_optimal_rate": get_metric(result, 'score_tool_usage', 'tool_optimal_rate'),
            "retrieved_correct_sources": get_metric(result, 'score_source_retrieval', 'retrieved_correct_sources'),
        }
        
        leaderboard_data.append(leaderboard_entry)
    
    # Create detailed validation table with model outputs and scores
    # Similar to train.py lines 233-243, but for multiple models
    print("\nCreating detailed validation comparison table...")
    
    # Start with base validation dataset columns
    validation_rows = dataset.rows
    base_columns = ["ID", "Split", "Question", "Answer", "Inbox", "Num Messages", "Realism Score"]
    table_data = []
    
    for row in validation_rows:
        table_data.append([
            row.get("id", ""),
            row.get("split", ""),
            row.get("question", "")[:100] + "..." if len(row.get("question", "")) > 100 else row.get("question", ""),
            row.get("answer", "")[:100] + "..." if len(row.get("answer", "")) > 100 else row.get("answer", ""),
            row.get("inbox_address", ""),
            len(row.get("message_ids", [])),
            row.get("how_realistic", 0.0)
        ])
    
    # Create the W&B table
    validation_table = wandb.Table(columns=base_columns, data=table_data)
    
    # For each model, add columns with model-specific outputs and scores
    for model_name, predictions in all_predictions.items():
        # Create a safe column suffix from model name
        col_suffix = f"_{model_name.replace('/', '-').replace('.', '-')}"
        
        # Collect data for this model's columns
        model_outputs = []
        source_ids_list = []
        judge_corrects = []
        judge_reasonings = []
        ruler_rewards = []
        retrieved_correct_sources_list = []
        tool_appropriate_rates = []
        tool_optimal_rates = []
        
        # Process each prediction
        for i, pred in enumerate(predictions):
            try:
                if pred and pred.get('trajectory'):
                    traj = pred['trajectory']
                    scenario = pred.get('scenario')
                    
                    # Extract model output
                    answer = pred.get('answer', '')
                    source_ids = pred.get('source_ids', [])
                    
                    model_outputs.append(answer[:200] + "..." if len(answer) > 200 else answer)
                    source_ids_list.append(str(source_ids))
                    
                    # Extract metrics from trajectory
                    if hasattr(traj, 'metrics'):
                        judge_corrects.append(traj.metrics.get("correct", 0.0))
                        retrieved_correct_sources_list.append(traj.metrics.get("retrieved_correct_sources", 0.0))
                        tool_appropriate_rates.append(traj.metrics.get("tool_appropriate_rate", 0.0))
                        tool_optimal_rates.append(traj.metrics.get("tool_optimal_rate", 0.0))
                    else:
                        judge_corrects.append(0.0)
                        retrieved_correct_sources_list.append(0.0)
                        tool_appropriate_rates.append(0.0)
                        tool_optimal_rates.append(0.0)
                    
                    # Extract judge reasoning from metadata
                    if hasattr(traj, 'metadata'):
                        reasoning = traj.metadata.get("judge_reasoning", "")
                        judge_reasonings.append(reasoning[:100] + "..." if len(reasoning) > 100 else reasoning)
                    else:
                        judge_reasonings.append("")
                    
                    # Extract ruler reward
                    if hasattr(traj, 'reward'):
                        ruler_rewards.append(traj.reward)
                    else:
                        ruler_rewards.append(0.0)
                else:
                    # No valid prediction
                    model_outputs.append("")
                    source_ids_list.append("[]")
                    judge_corrects.append(0.0)
                    judge_reasonings.append("")
                    ruler_rewards.append(0.0)
                    retrieved_correct_sources_list.append(0.0)
                    tool_appropriate_rates.append(0.0)
                    tool_optimal_rates.append(0.0)
            except Exception as e:
                print(f"Warning: Could not process prediction {i} for model {model_name}: {e}")
                model_outputs.append("")
                source_ids_list.append("[]")
                judge_corrects.append(0.0)
                judge_reasonings.append("")
                ruler_rewards.append(0.0)
                retrieved_correct_sources_list.append(0.0)
                tool_appropriate_rates.append(0.0)
                tool_optimal_rates.append(0.0)
        
        # Add columns for this model (similar to train.py lines 234-240)
        validation_table.add_column(f"model_output{col_suffix}", model_outputs)
        validation_table.add_column(f"model_source_ids{col_suffix}", source_ids_list)
        validation_table.add_column(f"judge_correct{col_suffix}", judge_corrects)
        validation_table.add_column(f"judge_reasoning{col_suffix}", judge_reasonings)
        validation_table.add_column(f"ruler_reward{col_suffix}", ruler_rewards)
        validation_table.add_column(f"retrieved_correct_sources{col_suffix}", retrieved_correct_sources_list)
        validation_table.add_column(f"tool_appropriate_rate{col_suffix}", tool_appropriate_rates)
        validation_table.add_column(f"tool_optimal_rate{col_suffix}", tool_optimal_rates)
    
    # Log the comprehensive validation table
    run.log({"validation_comparison_table": validation_table})
    print(f"✓ Logged validation comparison table with {len(validation_rows)} rows and outputs from {len(results)} models")
    
    # Log summary metrics to W&B for parallel coordinates
    for model_name, result in results.items():
        if isinstance(result, dict) and 'summary' in result:
            summary = result['summary']
            wandb.log({
                f"{model_name}/correctness": summary.get('correct', {}).get('mean', 0),
                f"{model_name}/source_precision": summary.get('source_precision', {}).get('mean', 0),
                f"{model_name}/source_recall": summary.get('source_recall', {}).get('mean', 0),
                f"{model_name}/source_f1": summary.get('source_f1', {}).get('mean', 0),
                f"{model_name}/tool_appropriate_rate": summary.get('tool_appropriate_rate', {}).get('mean', 0),
                f"{model_name}/tool_optimal_rate": summary.get('tool_optimal_rate', {}).get('mean', 0),
            })
    
    # Create Weave Leaderboard object
    try:
        # Create leaderboard using the Evaluation objects (not their results)
        leaderboard_spec = leaderboard.Leaderboard(
            name="Email Agent Model Comparison",
            description="""
This leaderboard compares the performance of different models on the email search agent task.

### Models
- **Comparison Model 1**: Configurable OpenAI model (default: gpt-4o-mini)
- **Comparison Model 2**: Configurable OpenAI model (default: gpt-5)
- **qwen-base**: OpenPipe/Qwen3-14B-Instruct base model (before fine-tuning)

### Metrics
1. **Correctness**: Whether the model's answer matches the expected answer
2. **Tool Optimal Rate**: Percentage of tool calls that were optimal
3. **Tool Appropriate Rate**: Percentage of tool calls that were appropriate (not incorrect)
4. **Retrieved Correct Sources**: Percentage of scenarios where correct source emails were retrieved
""",
            columns=[
                # GPT-4o-mini metrics
                leaderboard.LeaderboardColumn(
                    evaluation_object_ref=get_ref(evaluations[0]).uri(),
                    scorer_name="score_correctness",
                    summary_metric_path="correct.mean",
                ),
                leaderboard.LeaderboardColumn(
                    evaluation_object_ref=get_ref(evaluations[0]).uri(),
                    scorer_name="score_tool_usage",
                    summary_metric_path="tool_optimal_rate.mean",
                ),
                leaderboard.LeaderboardColumn(
                    evaluation_object_ref=get_ref(evaluations[0]).uri(),
                    scorer_name="score_tool_usage",
                    summary_metric_path="tool_appropriate_rate.mean",
                ),
                leaderboard.LeaderboardColumn(
                    evaluation_object_ref=get_ref(evaluations[0]).uri(),
                    scorer_name="score_source_retrieval",
                    summary_metric_path="retrieved_correct_sources.mean",
                ),
                # GPT-4o metrics
                leaderboard.LeaderboardColumn(
                    evaluation_object_ref=get_ref(evaluations[1]).uri(),
                    scorer_name="score_correctness",
                    summary_metric_path="correct.mean",
                ),
                leaderboard.LeaderboardColumn(
                    evaluation_object_ref=get_ref(evaluations[1]).uri(),
                    scorer_name="score_tool_usage",
                    summary_metric_path="tool_optimal_rate.mean",
                ),
                leaderboard.LeaderboardColumn(
                    evaluation_object_ref=get_ref(evaluations[1]).uri(),
                    scorer_name="score_tool_usage",
                    summary_metric_path="tool_appropriate_rate.mean",
                ),
                leaderboard.LeaderboardColumn(
                    evaluation_object_ref=get_ref(evaluations[1]).uri(),
                    scorer_name="score_source_retrieval",
                    summary_metric_path="retrieved_correct_sources.mean",
                ),
                # Qwen base metrics
                leaderboard.LeaderboardColumn(
                    evaluation_object_ref=get_ref(evaluations[2]).uri(),
                    scorer_name="score_correctness",
                    summary_metric_path="correct.mean",
                ),
                leaderboard.LeaderboardColumn(
                    evaluation_object_ref=get_ref(evaluations[2]).uri(),
                    scorer_name="score_tool_usage",
                    summary_metric_path="tool_optimal_rate.mean",
                ),
                leaderboard.LeaderboardColumn(
                    evaluation_object_ref=get_ref(evaluations[2]).uri(),
                    scorer_name="score_tool_usage",
                    summary_metric_path="tool_appropriate_rate.mean",
                ),
                leaderboard.LeaderboardColumn(
                    evaluation_object_ref=get_ref(evaluations[2]).uri(),
                    scorer_name="score_source_retrieval",
                    summary_metric_path="retrieved_correct_sources.mean",
                ),
            ],
        )
        
        leaderboard_ref = weave.publish(leaderboard_spec)
    except Exception as e:
        leaderboard_ref = None
    
    # Finish W&B run
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare models using Weave evaluations and leaderboards."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)"
    )
    args = parser.parse_args()
    
    asyncio.run(main(config_path=args.config))

