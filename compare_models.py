"""
Model comparison script using Weave evaluations and leaderboards.

This script compares three models using the three Weave scorers:
1. gpt-4o-mini (OpenAI)
2. gpt-4o (OpenAI)
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
    correctness_judge_model: str = "openai/gpt-4o"
    tool_judge_model: str = "openai/gpt-4o"
    
    @weave.op()
    async def predict(self, scenario: dict) -> dict:
        """Run the model on a scenario and return results.
        
        Args:
            scenario: Dict containing scenario information
            
        Returns:
            Dict with model outputs and metrics
        """
        # Convert dict to Scenario object
        scenario_obj = Scenario(**scenario)
        email_scenario = EmailScenario(step=0, scenario=scenario_obj)
        
        # Run rollout with all scorers
        trajectory = await rollout(
            self.model,
            email_scenario,
            correctness_judge_model=self.correctness_judge_model,
            tool_judge_model=self.tool_judge_model
        )
        
        # Extract results
        result = {
            "answer": trajectory.final_answer.answer if trajectory.final_answer else "",
            "source_ids": trajectory.final_answer.source_ids if trajectory.final_answer else [],
            "metrics": dict(trajectory.metrics),
            "tool_evaluations": trajectory.tool_evaluations,
            "num_turns": len([m for m in trajectory.messages_and_choices if isinstance(m, dict) and m.get('role') == 'assistant'])
        }
        
        return result


class OpenAIModelWrapper(weave.Model):
    """Wrapper for OpenAI models to work with Weave evaluations."""
    
    model_name: str
    correctness_judge_model: str = "openai/gpt-4o"
    tool_judge_model: str = "openai/gpt-4o"
    
    @weave.op()
    async def predict(self, scenario: dict) -> dict:
        """Run OpenAI model on a scenario.
        
        This uses a mock ART model that wraps OpenAI's API,
        then runs the full rollout with tool calling and evaluation.
        """
        # Convert dict to Scenario object
        scenario_obj = Scenario(**scenario)
        email_scenario = EmailScenario(step=0, scenario=scenario_obj)
        
        # Create a mock ART model that wraps OpenAI
        openai_model = OpenAIArtModel(self.model_name)
        
        # Run rollout with all scorers
        trajectory = await rollout(
            openai_model,
            email_scenario,
            correctness_judge_model=self.correctness_judge_model,
            tool_judge_model=self.tool_judge_model
        )
        
        # Extract results
        result = {
            "answer": trajectory.final_answer.answer if trajectory.final_answer else "",
            "source_ids": trajectory.final_answer.source_ids if trajectory.final_answer else [],
            "metrics": dict(trajectory.metrics),
            "tool_evaluations": trajectory.tool_evaluations,
            "num_turns": len([m for m in trajectory.messages_and_choices if isinstance(m, dict) and m.get('role') == 'assistant'])
        }
        
        return result


# Define scorers that work with the model outputs
@weave.op()
def extract_correctness_score(model_output: dict) -> dict:
    """Extract correctness score from model output."""
    return {
        "correct": model_output.get("metrics", {}).get("correct", 0.0),
        "reasoning": model_output.get("metrics", {}).get("reasoning", "")
    }


@weave.op()
def extract_source_retrieval_scores(model_output: dict) -> dict:
    """Extract source retrieval scores from model output."""
    metrics = model_output.get("metrics", {})
    return {
        "source_precision": metrics.get("source_precision", 0.0),
        "source_recall": metrics.get("source_recall", 0.0),
        "source_f1": metrics.get("source_f1", 0.0),
        "retrieved_correct_sources": metrics.get("retrieved_correct_sources", 0.0)
    }


@weave.op()
def extract_tool_usage_scores(model_output: dict) -> dict:
    """Extract tool usage scores from model output."""
    metrics = model_output.get("metrics", {})
    return {
        "total_decisions_evaluated": metrics.get("total_decisions_evaluated", 0.0),
        "actual_tool_calls": metrics.get("actual_tool_calls", 0.0),
        "no_tool_call_instances": metrics.get("no_tool_call_instances", 0.0),
        "tool_appropriate_rate": metrics.get("tool_appropriate_rate", 0.0),
        "tool_optimal_rate": metrics.get("tool_optimal_rate", 0.0),
        "tool_optimal_count": metrics.get("tool_optimal_count", 0.0),
        "tool_suboptimal_count": metrics.get("tool_suboptimal_count", 0.0),
        "tool_incorrect_count": metrics.get("tool_incorrect_count", 0.0)
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
    
    # Model 1: GPT-4o-mini
    gpt4o_mini = OpenAIModelWrapper(
        model_name="gpt-4o-mini",
        correctness_judge_model=config["correctness_judge_model"],
        tool_judge_model=config.get("tool_judge_model", "openai/gpt-4o")
    )
    
    # Model 2: GPT-4o
    gpt4o = OpenAIModelWrapper(
        model_name="gpt-4o",
        correctness_judge_model=config["correctness_judge_model"],
        tool_judge_model=config.get("tool_judge_model", "openai/gpt-4o")
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
        tool_judge_model=config.get("tool_judge_model", "openai/gpt-4o")
    )
    
    # Preprocessing function to convert dataset rows to model input format
    def preprocess_model_input(example: dict) -> dict:
        """Convert dataset example to the format expected by model predict methods."""
        return {"scenario": example}
    
    # Create evaluation with all scorers
    evaluation = weave.Evaluation(
        name="email-agent-model-comparison",
        dataset=dataset,
        scorers=[
            extract_correctness_score,
            extract_source_retrieval_scores,
            extract_tool_usage_scores
        ],
        preprocess_model_input=preprocess_model_input
    )
    
    # Run evaluations on all models and store both results and evaluation objects
    results = {}
    evaluation_objects = {}
    
    eval_result_mini = await evaluation.evaluate(gpt4o_mini)
    results["gpt-4o-mini"] = eval_result_mini
    evaluation_objects["gpt-4o-mini"] = eval_result_mini
    
    eval_result_4o = await evaluation.evaluate(gpt4o)
    results["gpt-4o"] = eval_result_4o
    evaluation_objects["gpt-4o"] = eval_result_4o
    
    eval_result_qwen = await evaluation.evaluate(qwen_base)
    results["qwen-base"] = eval_result_qwen
    evaluation_objects["qwen-base"] = eval_result_qwen
    
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
            "correctness": get_metric(result, 'extract_correctness_score', 'correct'),
            "source_precision": get_metric(result, 'extract_source_retrieval_scores', 'source_precision'),
            "source_recall": get_metric(result, 'extract_source_retrieval_scores', 'source_recall'),
            "source_f1": get_metric(result, 'extract_source_retrieval_scores', 'source_f1'),
            "tool_appropriate_rate": get_metric(result, 'extract_tool_usage_scores', 'tool_appropriate_rate'),
            "tool_optimal_rate": get_metric(result, 'extract_tool_usage_scores', 'tool_optimal_rate'),
            "retrieved_correct_sources": get_metric(result, 'extract_source_retrieval_scores', 'retrieved_correct_sources'),
        }
        
        leaderboard_data.append(leaderboard_entry)
    
    # Publish leaderboard to Weave
    leaderboard_dataset = weave.Dataset(
        name="model-comparison-leaderboard",
        rows=leaderboard_data
    )
    weave.publish(leaderboard_dataset)
    
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
    print("\n📊 Creating Weave Leaderboard...")
    
    try:
        leaderboard_spec = leaderboard.Leaderboard(
            name="Email Agent Model Comparison",
            description="""
This leaderboard compares the performance of different models on the email search agent task.

### Models
- **gpt-4o-mini**: OpenAI's cost-effective model
- **gpt-4o**: OpenAI's most capable model
- **qwen-base**: OpenPipe/Qwen3-14B-Instruct base model (before fine-tuning)

### Metrics
1. **Correctness**: Whether the model's answer matches the expected answer
2. **Source F1**: F1 score for retrieving the correct source emails
3. **Tool Optimal Rate**: Percentage of tool calls that were optimal
4. **Retrieved Correct Sources**: Percentage of scenarios where correct source emails were retrieved
""",
            columns=[
                leaderboard.LeaderboardColumn(
                    evaluation_object_ref=get_ref(evaluation_objects["gpt-4o-mini"]).uri(),
                    scorer_name="extract_correctness_score",
                    summary_metric_path="correct.mean",
                ),
                leaderboard.LeaderboardColumn(
                    evaluation_object_ref=get_ref(evaluation_objects["gpt-4o"]).uri(),
                    scorer_name="extract_correctness_score",
                    summary_metric_path="correct.mean",
                ),
                leaderboard.LeaderboardColumn(
                    evaluation_object_ref=get_ref(evaluation_objects["qwen-base"]).uri(),
                    scorer_name="extract_correctness_score",
                    summary_metric_path="correct.mean",
                ),
            ],
        )
        
        leaderboard_ref = weave.publish(leaderboard_spec)
        print(f"✅ Leaderboard published!")
    except Exception as e:
        print(f"⚠️ Error creating Weave Leaderboard: {e}")
        print("   The leaderboard dataset has been published to Weave instead")
        leaderboard_ref = None
    
    # Finish W&B run
    run.finish()
    
    print("\n🎉 Evaluation complete!")
    if leaderboard_ref:
        print(f"📊 View leaderboard in Weave: {leaderboard_ref}")
    print(f"📈 View W&B run: https://wandb.ai/{run.entity}/{run.project}/runs/{run.id}")


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

