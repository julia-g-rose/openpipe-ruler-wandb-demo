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


class ArtQwenModelWrapper(weave.Model):
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
            "tool_optimal_rate": 0.0
        }
    
    # Aggregate tool evaluations using the same logic as _add_tool_usage_metrics
    total = len(trajectory.tool_evaluations)
    optimal_count = sum(1 for eval in trajectory.tool_evaluations if eval["label"] == "optimal")
    
    return {
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
    
    # Comparison model (from config)
    comparison_model = OpenAIModelWrapper(
        model_name=config["comparison_model"],
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
    
    qwen_base = ArtQwenModelWrapper(
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
    comp_model_name = config["comparison_model"]
    
    # Create separate evaluation objects for each model
    # This is required for the Leaderboard API - we need to pass the Evaluation objects to get_ref()
    evaluations = [
        weave.Evaluation(
            name=f"{comp_model_name}-evaluation",
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
    models = [comparison_model, qwen_base]
    model_names = [comp_model_name, "qwen-base"]
    display_names = [comp_model_name, "OpenPipe/Qwen3-14B-Instruct base"]
    
    # Run evaluations
    results = {}
    
    for evaluation, model, model_name, display_name in zip(evaluations, models, model_names, display_names):
        print(f"\n🔄 Evaluating {display_name}...")
        
        # Run evaluation
        eval_result = await evaluation.evaluate(
            model,
            __weave={"display_name": display_name}
        )
        results[model_name] = eval_result
    
    print("\n✅ All evaluations complete!")
    
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
                    summary_metric_path="tool_optimal_rate.mean",
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
                    summary_metric_path="tool_optimal_rate.mean",
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
                    summary_metric_path="tool_optimal_rate.mean",
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

