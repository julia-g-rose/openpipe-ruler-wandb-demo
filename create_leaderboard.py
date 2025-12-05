"""
Model comparison script using Weave evaluations and leaderboards.

This script compares up to 9 models using three Weave scorers:
1. Comparison model (configurable, default: gpt-5)
2. OpenPipe/Qwen base model (before fine-tuning)
3. RULER-trained model (if exists)
4. Independent reward trained model (if exists)
5. Combined trained model with RULER + independent rewards (if exists)
6. RULER-trained model with LR=1e-5 (if exists)
7. Independent reward trained model with LR=1e-5 (if exists)
8. Combined trained model with LR=1e-5 (if exists)

Each trained model uses a separate wrapper class to appear as a unique entry
in the leaderboard with a readable display name.

It uses Weave's evaluation framework to run all scorers on the validation dataset
and creates a leaderboard for comparison. New models are automatically added to
the leaderboard by running evaluations with the same evaluation object.

Usage:
    # FIRST TIME: Create and publish the leaderboard with initial models
    python create_leaderboard.py --models openai base --publish-leaderboard
    
    # ADDING NEW ROWS: Evaluate additional models WITHOUT publishing
    # This adds new model results as rows to the existing leaderboard
    python create_leaderboard.py --models ruler independent
    
    # OVERWRITING: Re-publish the leaderboard (overwrites existing one)
    # Only use this if you want to update the leaderboard definition/description
    python create_leaderboard.py --models openai base ruler independent --publish-leaderboard
    
    # Use custom config file
    python create_leaderboard.py --config custom.yaml --models ruler

Important:
    - Use --publish-leaderboard ONLY the first time to create the leaderboard
    - After that, running evaluations WITHOUT the flag adds new rows to the existing leaderboard
    - Using --publish-leaderboard again OVERWRITES the existing leaderboard
    
Model options: openai, base, ruler, independent, combined, ruler_lr1e5, independent_lr1e5, combined_lr1e5, all
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


def format_model_name_with_lr(base_name: str, learning_rate: float) -> str:
    """Construct model name with learning rate suffix.
    
    Args:
        base_name: Base model name from config
        learning_rate: Learning rate value
        
    Returns:
        Model name with learning rate suffix, e.g. "model-name-lr1e-5"
    """
    lr_str = f"{learning_rate:.0e}".replace('-0', '-')  # Format like 1e-5
    return f"{base_name}-lr{lr_str}"


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


class ArtQwenBaseModelWrapper(weave.Model):
    """Wrapper for the base (untrained) Qwen model."""
    
    model: Any
    model_name: str
    correctness_judge_model: str
    tool_judge_model: str
    
    @weave.op()
    async def predict(self, id, question, answer, message_ids, how_realistic, inbox_address, query_date, split) -> dict:
        """Run the model on a scenario and return raw trajectory data for scoring.
        
        Note: Weave passes dataset rows as individual kwargs (id=..., question=..., etc.).
        We removed preprocess_model_input from the Evaluation to allow proper serialization.
        """
        print(f"[DEBUG] ArtQwenBaseModelWrapper.predict called (scenario_id={id})")
        
        # Convert individual fields to Scenario object
        scenario_obj = Scenario(
            id=id,
            question=question,
            answer=answer,
            message_ids=message_ids,
            how_realistic=how_realistic,
            inbox_address=inbox_address,
            query_date=query_date,
            split=split
        )
        email_scenario = EmailScenario(step=0, scenario=scenario_obj)
        
        trajectory = await rollout(
            self.model,
            email_scenario,
            correctness_judge_model=self.correctness_judge_model,
            tool_judge_model=self.tool_judge_model
        )
        
        result = {
            "trajectory": trajectory,
            "scenario": scenario_obj,
            "answer": trajectory.final_answer.answer if trajectory.final_answer else "",
            "source_ids": trajectory.final_answer.source_ids if trajectory.final_answer else [],
        }
        
        print(f"[DEBUG] ArtQwenBaseModelWrapper.predict completed - "
              f"answer={'present' if result['answer'] else 'empty'}, "
              f"sources={len(result['source_ids'])}, "
              f"tool_evals={len(trajectory.tool_evaluations) if trajectory.tool_evaluations else 0}")
        
        return result


class ArtQwenRulerTrainedModelWrapper(weave.Model):
    """Wrapper for the RULER-trained Qwen model."""
    
    model: Any
    model_name: str
    correctness_judge_model: str
    tool_judge_model: str
    
    @weave.op()
    async def predict(self, id, question, answer, message_ids, how_realistic, inbox_address, query_date, split) -> dict:
        """Run the model on a scenario and return raw trajectory data for scoring.
        
        Note: Weave passes dataset rows as individual kwargs (id=..., question=..., etc.).
        We removed preprocess_model_input from the Evaluation to allow proper serialization.
        """
        # Convert individual fields to Scenario object
        scenario_obj = Scenario(
            id=id,
            question=question,
            answer=answer,
            message_ids=message_ids,
            how_realistic=how_realistic,
            inbox_address=inbox_address,
            query_date=query_date,
            split=split
        )
        email_scenario = EmailScenario(step=0, scenario=scenario_obj)
        
        trajectory = await rollout(
            self.model,
            email_scenario,
            correctness_judge_model=self.correctness_judge_model,
            tool_judge_model=self.tool_judge_model
        )
        
        result = {
            "trajectory": trajectory,
            "scenario": scenario_obj,
            "answer": trajectory.final_answer.answer if trajectory.final_answer else "",
            "source_ids": trajectory.final_answer.source_ids if trajectory.final_answer else [],
        }
        
        return result


class ArtQwenIndependentTrainedModelWrapper(weave.Model):
    """Wrapper for the independent reward trained Qwen model."""
    
    model: Any
    model_name: str
    correctness_judge_model: str
    tool_judge_model: str
    
    @weave.op()
    async def predict(self, id, question, answer, message_ids, how_realistic, inbox_address, query_date, split) -> dict:
        """Run the model on a scenario and return raw trajectory data for scoring.
        
        Note: Weave passes dataset rows as individual kwargs (id=..., question=..., etc.).
        We removed preprocess_model_input from the Evaluation to allow proper serialization.
        """
        # Convert individual fields to Scenario object
        scenario_obj = Scenario(
            id=id,
            question=question,
            answer=answer,
            message_ids=message_ids,
            how_realistic=how_realistic,
            inbox_address=inbox_address,
            query_date=query_date,
            split=split
        )
        email_scenario = EmailScenario(step=0, scenario=scenario_obj)
        
        trajectory = await rollout(
            self.model,
            email_scenario,
            correctness_judge_model=self.correctness_judge_model,
            tool_judge_model=self.tool_judge_model
        )
        
        result = {
            "trajectory": trajectory,
            "scenario": scenario_obj,
            "answer": trajectory.final_answer.answer if trajectory.final_answer else "",
            "source_ids": trajectory.final_answer.source_ids if trajectory.final_answer else [],
        }
        
        return result


class ArtQwenCombinedTrainedModelWrapper(weave.Model):
    """Wrapper for the combined (RULER + independent rewards) trained Qwen model."""
    
    model: Any
    model_name: str
    correctness_judge_model: str
    tool_judge_model: str
    
    @weave.op()
    async def predict(self, id, question, answer, message_ids, how_realistic, inbox_address, query_date, split) -> dict:
        """Run the model on a scenario and return raw trajectory data for scoring.
        
        Note: Weave passes dataset rows as individual kwargs (id=..., question=..., etc.).
        We removed preprocess_model_input from the Evaluation to allow proper serialization.
        """
        # Convert individual fields to Scenario object
        scenario_obj = Scenario(
            id=id,
            question=question,
            answer=answer,
            message_ids=message_ids,
            how_realistic=how_realistic,
            inbox_address=inbox_address,
            query_date=query_date,
            split=split
        )
        email_scenario = EmailScenario(step=0, scenario=scenario_obj)
        
        trajectory = await rollout(
            self.model,
            email_scenario,
            correctness_judge_model=self.correctness_judge_model,
            tool_judge_model=self.tool_judge_model
        )
        
        result = {
            "trajectory": trajectory,
            "scenario": scenario_obj,
            "answer": trajectory.final_answer.answer if trajectory.final_answer else "",
            "source_ids": trajectory.final_answer.source_ids if trajectory.final_answer else [],
        }
        
        return result


# LR=1e-5 variant wrappers (separate classes for distinct leaderboard entries)

class ArtQwenRulerLR1e5TrainedModelWrapper(weave.Model):
    """Wrapper for the RULER-trained Qwen model with LR=1e-5."""
    
    model: Any
    model_name: str
    correctness_judge_model: str
    tool_judge_model: str
    
    @weave.op()
    async def predict(self, id, question, answer, message_ids, how_realistic, inbox_address, query_date, split) -> dict:
        """Run the model on a scenario and return raw trajectory data for scoring."""
        scenario_obj = Scenario(
            id=id,
            question=question,
            answer=answer,
            message_ids=message_ids,
            how_realistic=how_realistic,
            inbox_address=inbox_address,
            query_date=query_date,
            split=split
        )
        email_scenario = EmailScenario(step=0, scenario=scenario_obj)
        
        trajectory = await rollout(
            self.model,
            email_scenario,
            correctness_judge_model=self.correctness_judge_model,
            tool_judge_model=self.tool_judge_model
        )
        
        result = {
            "trajectory": trajectory,
            "scenario": scenario_obj,
            "answer": trajectory.final_answer.answer if trajectory.final_answer else "",
            "source_ids": trajectory.final_answer.source_ids if trajectory.final_answer else [],
        }
        
        return result


class ArtQwenIndependentLR1e5TrainedModelWrapper(weave.Model):
    """Wrapper for the independent reward trained Qwen model with LR=1e-5."""
    
    model: Any
    model_name: str
    correctness_judge_model: str
    tool_judge_model: str
    
    @weave.op()
    async def predict(self, id, question, answer, message_ids, how_realistic, inbox_address, query_date, split) -> dict:
        """Run the model on a scenario and return raw trajectory data for scoring."""
        scenario_obj = Scenario(
            id=id,
            question=question,
            answer=answer,
            message_ids=message_ids,
            how_realistic=how_realistic,
            inbox_address=inbox_address,
            query_date=query_date,
            split=split
        )
        email_scenario = EmailScenario(step=0, scenario=scenario_obj)
        
        trajectory = await rollout(
            self.model,
            email_scenario,
            correctness_judge_model=self.correctness_judge_model,
            tool_judge_model=self.tool_judge_model
        )
        
        result = {
            "trajectory": trajectory,
            "scenario": scenario_obj,
            "answer": trajectory.final_answer.answer if trajectory.final_answer else "",
            "source_ids": trajectory.final_answer.source_ids if trajectory.final_answer else [],
        }
        
        return result


class ArtQwenCombinedLR1e5TrainedModelWrapper(weave.Model):
    """Wrapper for the combined (RULER + independent rewards) trained Qwen model with LR=1e-5."""
    
    model: Any
    model_name: str
    correctness_judge_model: str
    tool_judge_model: str
    
    @weave.op()
    async def predict(self, id, question, answer, message_ids, how_realistic, inbox_address, query_date, split) -> dict:
        """Run the model on a scenario and return raw trajectory data for scoring."""
        scenario_obj = Scenario(
            id=id,
            question=question,
            answer=answer,
            message_ids=message_ids,
            how_realistic=how_realistic,
            inbox_address=inbox_address,
            query_date=query_date,
            split=split
        )
        email_scenario = EmailScenario(step=0, scenario=scenario_obj)
        
        trajectory = await rollout(
            self.model,
            email_scenario,
            correctness_judge_model=self.correctness_judge_model,
            tool_judge_model=self.tool_judge_model
        )
        
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
    async def predict(self, id, question, answer, message_ids, how_realistic, inbox_address, query_date, split) -> dict:
        """Run OpenAI model on a scenario and return raw trajectory data for scoring.
        
        This uses a mock ART model that wraps OpenAI's API,
        then runs the full rollout with tool calling.
        
        Note: Weave passes dataset rows as individual kwargs (id=..., question=..., etc.).
        We removed preprocess_model_input from the Evaluation to allow proper serialization.
        """
        print(f"[DEBUG] OpenAIModelWrapper.predict called (scenario_id={id})")
        
        # Convert individual fields to Scenario object
        scenario_obj = Scenario(
            id=id,
            question=question,
            answer=answer,
            message_ids=message_ids,
            how_realistic=how_realistic,
            inbox_address=inbox_address,
            query_date=query_date,
            split=split
        )
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
        
        print(f"[DEBUG] OpenAIModelWrapper.predict completed - "
              f"answer={'present' if result['answer'] else 'empty'}, "
              f"sources={len(result['source_ids'])}, "
              f"tool_evals={len(trajectory.tool_evaluations) if trajectory.tool_evaluations else 0}")
        
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
        print(f"[DEBUG] score_correctness called - output_keys={list(model_output.keys()) if model_output else 'empty'}")
        
        trajectory = model_output.get("trajectory")
        scenario = model_output.get("scenario")
        answer = model_output.get("answer", "")
        
        if not scenario:
            print("[DEBUG] score_correctness FAILED: missing scenario!")
            return {"correct": 0.0, "reasoning": "Missing scenario data"}
        
        # Initialize and use the actual scorer from helpers.py
        correctness_scorer = CorrectnessJudgeScorer(judge_model=judge_model)
        result = await correctness_scorer.score(
            output=answer,
            question=scenario.question,
            reference_answer=scenario.answer
        )
        
        print(f"[DEBUG] score_correctness completed - correct={result.get('correct', 'N/A')}")
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
    
    print(f"[DEBUG] score_source_retrieval called - retrieved={len(source_ids)}, expected={len(scenario.message_ids) if scenario else 0}")
    
    if not scenario:
        print("[DEBUG] score_source_retrieval FAILED: missing scenario!")
        return {
            "retrieved_correct_sources": 0.0
        }
    
    # Initialize and use the actual scorer from helpers.py
    source_scorer = SourceRetrievalScorer()
    result = await source_scorer.score(
        output={"source_ids": source_ids},
        expected_source_ids=scenario.message_ids
    )
    
    print(f"[DEBUG] score_source_retrieval completed - score={result.get('retrieved_correct_sources', 'N/A')}")
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
        print(f"[DEBUG] score_tool_usage called - FAILED: no tool_evaluations (trajectory={'present' if trajectory else 'missing'})")
        return {
            "tool_optimal_rate": 0.0
        }
    
    # Aggregate tool evaluations using the same logic as _add_tool_usage_metrics
    total = len(trajectory.tool_evaluations)
    optimal_count = sum(1 for eval in trajectory.tool_evaluations if eval["label"] == "optimal")
    tool_rate = optimal_count / total if total > 0 else 0.0
    
    print(f"[DEBUG] score_tool_usage called - optimal={optimal_count}/{total}, rate={tool_rate:.2f}")
    
    return {
        "tool_optimal_rate": tool_rate
    }


async def main(config_path: str = "config.yaml", models_to_eval: list = None, publish_leaderboard: bool = False):
    """Main comparison function.
    
    Args:
        config_path: Path to the YAML configuration file
        models_to_eval: List of model names to evaluate (default: all)
        publish_leaderboard: Whether to publish the leaderboard (default: False)
    """
    if models_to_eval is None:
        models_to_eval = ["all"]
    
    # If "all" is in the list, evaluate all models
    eval_all = "all" in models_to_eval
    
    # Load configuration
    config = load_config(config_path)
    
    # Initialize Weave and load existing dataset
    initialize_weave(config["project"])
    
    # Load the original validation dataset and republish with -leaderboard suffix
    # This creates a fresh dataset to avoid any caching/memory from old evaluation runs
    print("\n🔄 Loading original validation dataset...")
    original_dataset = weave.ref("enron-validation-scenarios").get()
    print(f"✓ Loaded {len(original_dataset.rows)} scenarios from original dataset")
    
    # Try to load existing leaderboard dataset, or create new one
    try:
        dataset = weave.ref("enron-validation-scenarios-leaderboard").get()
        print("✓ Using existing leaderboard dataset")
    except:
        print("📦 Creating fresh leaderboard dataset...")
        # Create new dataset with -leaderboard suffix
        dataset = weave.Dataset(
            name="enron-validation-scenarios-leaderboard",
            rows=original_dataset.rows
        )
        weave.publish(dataset)
        print("✓ Published fresh leaderboard dataset")
    
    # Initialize W&B for logging results
    run = wandb.init(
        project=config["project"],
        name="model-comparison-evaluation",
        config=config,
        job_type="comparison",
    )
    
    # Determine which models to evaluate
    should_eval_openai = eval_all or "openai" in models_to_eval
    should_eval_base = eval_all or "base" in models_to_eval
    should_eval_ruler = eval_all or "ruler" in models_to_eval
    should_eval_independent = eval_all or "independent" in models_to_eval
    should_eval_combined = eval_all or "combined" in models_to_eval
    should_eval_ruler_lr1e5 = "ruler_lr1e5" in models_to_eval
    should_eval_independent_lr1e5 = "independent_lr1e5" in models_to_eval
    should_eval_combined_lr1e5 = "combined_lr1e5" in models_to_eval
    
    # Initialize backend for ART models
    backend = ServerlessBackend()
    
    # Comparison model (from config)
    comparison_model = None
    if should_eval_openai:
        print("\n🔄 Loading OpenAI comparison model...")
        comparison_model = OpenAIModelWrapper(
            model_name=config["comparison_model"],
            correctness_judge_model=config["correctness_judge_model"],
            tool_judge_model=config["tool_judge_model"]
        )
    
    # Model 3: OpenPipe/Qwen base model (untrained)
    qwen_base = None
    if should_eval_base:
        print("\n🔄 Loading base model...")
        # Use Model (not TrainableModel) since this is the untrained base model
        base_model = art.Model(
            name=config["base_model"],  # Use the base model name directly
            project=config["project"],
        )
        await base_model.register(backend)
        qwen_base = ArtQwenBaseModelWrapper(
            model=base_model,
            model_name=config["base_model"],
            correctness_judge_model=config["correctness_judge_model"],
            tool_judge_model=config["tool_judge_model"]
        )
    
    # Common scorers for all evaluations
    # These use the actual scorer classes from helpers.py
    scorers = [
        create_correctness_scorer(config["correctness_judge_model"]),
        score_source_retrieval,
        score_tool_usage
    ]
    print(f"\n✓ Created {len(scorers)} scorers")
    
    # Get model names from config
    comp_model_name = config["comparison_model"]
    
    print(f"\n{'='*80}")
    print("EVALUATION SETUP")
    print(f"{'='*80}")
    print(f"Comparison model: {comp_model_name}")
    print(f"Correctness judge: {config['correctness_judge_model']}")
    print(f"Tool judge: {config['tool_judge_model']}")
    print(f"Dataset: enron-validation-scenarios")
    
    # Try to load existing evaluation object, or create new one if it doesn't exist
    # This ensures all models appear as rows under the same columns in the leaderboard
    # and that the leaderboard automatically updates with new evaluation results
    #
    # IMPORTANT: We removed preprocess_model_input from the Evaluation to allow
    # proper serialization. The model wrappers now handle preprocessing directly.
    #
    # Using -leaderboard suffix to avoid conflicts with previous evaluation runs
    try:
        # Try to get the latest version of the evaluation object
        shared_evaluation = weave.ref("email-agent-evaluation-leaderboard").get()
        print("\n✓ Loaded existing evaluation object - new results will automatically appear in leaderboard")
    except Exception as e:
        # If it doesn't exist, create a new one
        print(f"\n⚠️  Could not load existing evaluation object, creating new one: {e}")
        shared_evaluation = weave.Evaluation(
            name="email-agent-evaluation-leaderboard",  # Single shared evaluation name with -leaderboard suffix
            dataset=dataset,
            scorers=scorers
            # Note: No preprocess_model_input - models handle preprocessing in their predict() methods
        )
    
    # Model 4: Try to load the RULER-trained model if it exists
    trained_ruler_model_wrapper = None
    trained_ruler_step = 0
    if should_eval_ruler:
        try:
            print("\n🔄 Checking for RULER-trained model...")
            ruler_model_name = format_model_name_with_lr(
                config["model_name_ruler"], 
                config["learning_rate"]
            )
            trained_ruler_model = art.TrainableModel(
                name=ruler_model_name,
                project=config["project"],
                base_model=config["base_model"],
            )
            await trained_ruler_model.register(backend)
            trained_ruler_step = await trained_ruler_model.get_step()
            
            if trained_ruler_step > 0:
                print(f"✓ Found RULER-trained model at step {trained_ruler_step}")
                trained_ruler_model_wrapper = ArtQwenRulerTrainedModelWrapper(
                    model=trained_ruler_model,
                    model_name=f"{config['base_model']} (RULER @ step {trained_ruler_step})",
                    correctness_judge_model=config["correctness_judge_model"],
                    tool_judge_model=config["tool_judge_model"]
                )
            else:
                print("⚠️  RULER model exists but is at step 0 (not trained yet)")
        except Exception as e:
            print(f"ℹ️  No RULER-trained model found: {e}")
    else:
        print("\nℹ️  Skipping RULER-trained model evaluation")
    
    # Model 5: Try to load the independent trained model if it exists
    trained_independent_model_wrapper = None
    trained_independent_step = 0
    if should_eval_independent:
        try:
            print("\n🔄 Checking for independent-trained model...")
            independent_model_name = format_model_name_with_lr(
                config["model_name_independent"], 
                config["learning_rate"]
            )
            trained_independent_model = art.TrainableModel(
                name=independent_model_name,
                project=config["project"],
                base_model=config["base_model"],
            )
            await trained_independent_model.register(backend)
            trained_independent_step = await trained_independent_model.get_step()
            
            if trained_independent_step > 0:
                print(f"✓ Found independent-trained model at step {trained_independent_step}")
                trained_independent_model_wrapper = ArtQwenIndependentTrainedModelWrapper(
                    model=trained_independent_model,
                    model_name=f"{config['base_model']} (Independent @ step {trained_independent_step})",
                    correctness_judge_model=config["correctness_judge_model"],
                    tool_judge_model=config["tool_judge_model"]
                )
            else:
                print("⚠️  Independent model exists but is at step 0 (not trained yet)")
        except Exception as e:
            print(f"ℹ️  No independent-trained model found: {e}")
    else:
        print("\nℹ️  Skipping independent-trained model evaluation")
    
    # Model 6: Try to load the combined trained model if it exists
    trained_combined_model_wrapper = None
    trained_combined_step = 0
    if should_eval_combined:
        try:
            print("\n🔄 Checking for combined-trained model...")
            combined_model_name = format_model_name_with_lr(
                config["model_name_combined"], 
                config["learning_rate"]
            )
            trained_combined_model = art.TrainableModel(
                name=combined_model_name,
                project=config["project"],
                base_model=config["base_model"],
            )
            await trained_combined_model.register(backend)
            trained_combined_step = await trained_combined_model.get_step()
            
            if trained_combined_step > 0:
                print(f"✓ Found combined-trained model at step {trained_combined_step}")
                trained_combined_model_wrapper = ArtQwenCombinedTrainedModelWrapper(
                    model=trained_combined_model,
                    model_name=f"{config['base_model']} (Combined @ step {trained_combined_step})",
                    correctness_judge_model=config["correctness_judge_model"],
                    tool_judge_model=config["tool_judge_model"]
                )
            else:
                print("⚠️  Combined model exists but is at step 0 (not trained yet)")
        except Exception as e:
            print(f"ℹ️  No combined-trained model found: {e}")
    else:
        print("\nℹ️  Skipping combined-trained model evaluation")
    
    # LR=1e-5 variants
    
    # Model 7: Try to load the RULER LR=1e-5 trained model if it exists
    trained_ruler_lr1e5_model_wrapper = None
    trained_ruler_lr1e5_step = 0
    if should_eval_ruler_lr1e5:
        try:
            print("\n🔄 Checking for RULER LR=1e-5 trained model...")
            ruler_lr1e5_model_name = format_model_name_with_lr(
                config["model_name_ruler"], 
                1e-5
            )
            trained_ruler_lr1e5_model = art.TrainableModel(
                name=ruler_lr1e5_model_name,
                project=config["project"],
                base_model=config["base_model"],
            )
            await trained_ruler_lr1e5_model.register(backend)
            trained_ruler_lr1e5_step = await trained_ruler_lr1e5_model.get_step()
            
            if trained_ruler_lr1e5_step > 0:
                print(f"✓ Found RULER LR=1e-5 trained model at step {trained_ruler_lr1e5_step}")
                trained_ruler_lr1e5_model_wrapper = ArtQwenRulerLR1e5TrainedModelWrapper(
                    model=trained_ruler_lr1e5_model,
                    model_name=f"{config['base_model']} (RULER LR=1e-5 @ step {trained_ruler_lr1e5_step})",
                    correctness_judge_model=config["correctness_judge_model"],
                    tool_judge_model=config["tool_judge_model"]
                )
            else:
                print("⚠️  RULER LR=1e-5 model exists but is at step 0 (not trained yet)")
        except Exception as e:
            print(f"ℹ️  No RULER LR=1e-5 trained model found: {e}")
    else:
        print("\nℹ️  Skipping RULER LR=1e-5 trained model evaluation")
    
    # Model 8: Try to load the independent LR=1e-5 trained model if it exists
    trained_independent_lr1e5_model_wrapper = None
    trained_independent_lr1e5_step = 0
    if should_eval_independent_lr1e5:
        try:
            print("\n🔄 Checking for independent LR=1e-5 trained model...")
            independent_lr1e5_model_name = format_model_name_with_lr(
                config["model_name_independent"], 
                1e-5
            )
            trained_independent_lr1e5_model = art.TrainableModel(
                name=independent_lr1e5_model_name,
                project=config["project"],
                base_model=config["base_model"],
            )
            await trained_independent_lr1e5_model.register(backend)
            trained_independent_lr1e5_step = await trained_independent_lr1e5_model.get_step()
            
            if trained_independent_lr1e5_step > 0:
                print(f"✓ Found independent LR=1e-5 trained model at step {trained_independent_lr1e5_step}")
                trained_independent_lr1e5_model_wrapper = ArtQwenIndependentLR1e5TrainedModelWrapper(
                    model=trained_independent_lr1e5_model,
                    model_name=f"{config['base_model']} (Independent LR=1e-5 @ step {trained_independent_lr1e5_step})",
                    correctness_judge_model=config["correctness_judge_model"],
                    tool_judge_model=config["tool_judge_model"]
                )
            else:
                print("⚠️  Independent LR=1e-5 model exists but is at step 0 (not trained yet)")
        except Exception as e:
            print(f"ℹ️  No independent LR=1e-5 trained model found: {e}")
    else:
        print("\nℹ️  Skipping independent LR=1e-5 trained model evaluation")
    
    # Model 9: Try to load the combined LR=1e-5 trained model if it exists
    trained_combined_lr1e5_model_wrapper = None
    trained_combined_lr1e5_step = 0
    if should_eval_combined_lr1e5:
        try:
            print("\n🔄 Checking for combined LR=1e-5 trained model...")
            combined_lr1e5_model_name = format_model_name_with_lr(
                config["model_name_combined"], 
                1e-5
            )
            trained_combined_lr1e5_model = art.TrainableModel(
                name=combined_lr1e5_model_name,
                project=config["project"],
                base_model=config["base_model"],
            )
            await trained_combined_lr1e5_model.register(backend)
            trained_combined_lr1e5_step = await trained_combined_lr1e5_model.get_step()
            
            if trained_combined_lr1e5_step > 0:
                print(f"✓ Found combined LR=1e-5 trained model at step {trained_combined_lr1e5_step}")
                trained_combined_lr1e5_model_wrapper = ArtQwenCombinedLR1e5TrainedModelWrapper(
                    model=trained_combined_lr1e5_model,
                    model_name=f"{config['base_model']} (Combined LR=1e-5 @ step {trained_combined_lr1e5_step})",
                    correctness_judge_model=config["correctness_judge_model"],
                    tool_judge_model=config["tool_judge_model"]
                )
            else:
                print("⚠️  Combined LR=1e-5 model exists but is at step 0 (not trained yet)")
        except Exception as e:
            print(f"ℹ️  No combined LR=1e-5 trained model found: {e}")
    else:
        print("\nℹ️  Skipping combined LR=1e-5 trained model evaluation")
    
    # Models list - only include models that were loaded
    models = []
    model_names = []
    display_names = []
    
    # Add OpenAI comparison model if it was loaded
    if comparison_model:
        models.append(comparison_model)
        model_names.append(comp_model_name)
        display_names.append(comp_model_name)
    
    # Add base model if it was loaded
    if qwen_base:
        models.append(qwen_base)
        model_names.append("qwen-base")
        display_names.append("OpenPipe/Qwen3-14B-Instruct base")
    
    # Add RULER-trained model if it exists
    if trained_ruler_model_wrapper:
        models.append(trained_ruler_model_wrapper)
        model_names.append("qwen-ruler-trained")
        display_names.append(f"OpenPipe/Qwen3-14B-Instruct (RULER @ step {trained_ruler_step})")
    
    # Add independent-trained model if it exists
    if trained_independent_model_wrapper:
        models.append(trained_independent_model_wrapper)
        model_names.append("qwen-independent-trained")
        display_names.append(f"OpenPipe/Qwen3-14B-Instruct (Independent @ step {trained_independent_step})")
    
    # Add combined-trained model if it exists
    if trained_combined_model_wrapper:
        models.append(trained_combined_model_wrapper)
        model_names.append("qwen-combined-trained")
        display_names.append(f"OpenPipe/Qwen3-14B-Instruct (Combined @ step {trained_combined_step})")
    
    # Add LR=1e-5 variants if they exist
    if trained_ruler_lr1e5_model_wrapper:
        models.append(trained_ruler_lr1e5_model_wrapper)
        model_names.append("qwen-ruler-lr1e5-trained")
        display_names.append(f"OpenPipe/Qwen3-14B-Instruct (RULER LR=1e-5 @ step {trained_ruler_lr1e5_step})")
    
    if trained_independent_lr1e5_model_wrapper:
        models.append(trained_independent_lr1e5_model_wrapper)
        model_names.append("qwen-independent-lr1e5-trained")
        display_names.append(f"OpenPipe/Qwen3-14B-Instruct (Independent LR=1e-5 @ step {trained_independent_lr1e5_step})")
    
    if trained_combined_lr1e5_model_wrapper:
        models.append(trained_combined_lr1e5_model_wrapper)
        model_names.append("qwen-combined-lr1e5-trained")
        display_names.append(f"OpenPipe/Qwen3-14B-Instruct (Combined LR=1e-5 @ step {trained_combined_lr1e5_step})")
    
    # Run evaluations - all models evaluated with the SAME evaluation object
    results = {}
    
    if not models:
        print("\n⚠️  No models to evaluate. Use --models to specify which models to evaluate.")
    else:
        for idx, (model, model_name, display_name) in enumerate(zip(models, model_names, display_names), 1):
            print(f"\n{'='*80}")
            print(f"🔄 Evaluating Model {idx}/{len(models)}: {display_name}")
            print(f"{'='*80}")
            
            # Run evaluation - all using the shared evaluation
            eval_result = await shared_evaluation.evaluate(
                model,
                __weave={"display_name": display_name}
            )
            results[model_name] = eval_result
            
            print(f"\n✅ Completed evaluation for: {display_name}")
            print(f"   Results summary: {eval_result}")
        
        print("\n✅ All evaluations complete!")
    
    # Create Weave Leaderboard object
    if not results:
        print("\n⏭️  Skipping leaderboard creation (no models were evaluated)")
        leaderboard_ref = None
    else:
        try:
            # Create leaderboard using the single shared Evaluation object
            # All models evaluated with this evaluation will appear as rows
            # Build dynamic description based on which models were found
            model_descriptions = [
                f"- **{comp_model_name}**: Comparison model (configurable OpenAI model)",
                "- **OpenPipe/Qwen3-14B-Instruct Base**: Untrained base model"
            ]
            
            if trained_ruler_model_wrapper:
                model_descriptions.append(f"- **OpenPipe/Qwen3-14B-Instruct (RULER)**: Fine-tuned with RULER rewards (step {trained_ruler_step})")
            
            if trained_independent_model_wrapper:
                model_descriptions.append(f"- **OpenPipe/Qwen3-14B-Instruct (Independent)**: Fine-tuned with independent rewards (step {trained_independent_step})")
            
            if trained_combined_model_wrapper:
                model_descriptions.append(f"- **OpenPipe/Qwen3-14B-Instruct (Combined)**: Fine-tuned with RULER + independent rewards (step {trained_combined_step})")
            
            if trained_ruler_lr1e5_model_wrapper:
                model_descriptions.append(f"- **OpenPipe/Qwen3-14B-Instruct (RULER LR=1e-5)**: Fine-tuned with RULER rewards at LR=1e-5 (step {trained_ruler_lr1e5_step})")
            
            if trained_independent_lr1e5_model_wrapper:
                model_descriptions.append(f"- **OpenPipe/Qwen3-14B-Instruct (Independent LR=1e-5)**: Fine-tuned with independent rewards at LR=1e-5 (step {trained_independent_lr1e5_step})")
            
            if trained_combined_lr1e5_model_wrapper:
                model_descriptions.append(f"- **OpenPipe/Qwen3-14B-Instruct (Combined LR=1e-5)**: Fine-tuned with RULER + independent rewards at LR=1e-5 (step {trained_combined_lr1e5_step})")
            
            leaderboard_spec = leaderboard.Leaderboard(
                name="Best Email Agent Models - Final Leaderboard",
                description=f"""
This leaderboard compares the performance of the best-performing models on the email search agent task.

### Models Being Compared
{chr(10).join(model_descriptions)}

### Metrics
1. **Correctness** (correct.mean): Whether the model's answer matches the expected answer (0-1 scale)
2. **Tool Optimal Rate** (tool_optimal_rate.mean): Percentage of tool calls that were optimal (0-1 scale)
3. **Retrieved Correct Sources** (retrieved_correct_sources.mean): Percentage of scenarios where correct source emails were retrieved (0-1 scale)

### Evaluation Dataset
- **Dataset**: enron-validation-scenarios-leaderboard (100 examples)
- **Judge Models**: {config['correctness_judge_model']} for correctness, {config['tool_judge_model']} for tool usage

### Note
All models are evaluated using the same shared evaluation object to ensure fair comparison.
Each trained model uses a unique wrapper class to display separately in the leaderboard.
""",
                columns=[
                    # Single set of columns for the shared evaluation
                    # All models appear as rows under these columns
                    leaderboard.LeaderboardColumn(
                        evaluation_object_ref=get_ref(shared_evaluation).uri(),
                        scorer_name="score_correctness",
                        summary_metric_path="correct.mean",
                    ),
                    leaderboard.LeaderboardColumn(
                        evaluation_object_ref=get_ref(shared_evaluation).uri(),
                        scorer_name="score_tool_usage",
                        summary_metric_path="tool_optimal_rate.mean",
                    ),
                    leaderboard.LeaderboardColumn(
                        evaluation_object_ref=get_ref(shared_evaluation).uri(),
                        scorer_name="score_source_retrieval",
                        summary_metric_path="retrieved_correct_sources.mean",
                    ),
                ],
            )
            
            if publish_leaderboard:
                # WARNING: Publishing overwrites the existing leaderboard definition
                # Use this only for initial creation or when updating the leaderboard structure
                leaderboard_ref = weave.publish(leaderboard_spec)
                print(f"\n📊 Leaderboard published: {leaderboard_ref}")
            else:
                # Not publishing - evaluation results are still saved and will appear as new rows
                # in the existing leaderboard (if it was published previously)
                print("\n⏭️  Leaderboard spec created but not published")
                print("    Evaluation results are saved and will appear in the existing leaderboard")
                print("    Use --publish-leaderboard only if you want to overwrite the leaderboard definition")
                leaderboard_ref = None
        except Exception as e:
            print(f"\n❌ Failed to create leaderboard: {e}")
            import traceback
            traceback.print_exc()
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
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["openai", "base", "ruler", "independent", "combined", "ruler_lr1e5", "independent_lr1e5", "combined_lr1e5", "all"],
        default=["all"],
        help="Which models to evaluate. Options: openai, base, ruler, independent, combined, ruler_lr1e5, independent_lr1e5, combined_lr1e5, all (default: all)"
    )
    parser.add_argument(
        "--publish-leaderboard",
        action="store_true",
        default=False,
        help="Publish/overwrite the leaderboard definition. Use ONLY for initial creation or to update the leaderboard structure. Omit this flag to simply add new model rows to existing leaderboard. (default: False)"
    )
    args = parser.parse_args()
    
    asyncio.run(main(
        config_path=args.config,
        models_to_eval=args.models,
        publish_leaderboard=args.publish_leaderboard
    ))

