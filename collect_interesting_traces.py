"""
Collect interesting traces for qualitative analysis.

Goal: Find specific examples where Combined and RULER differ,
with full trace details for manual inspection.
"""
import weave
from collections import defaultdict
import json
import pickle

def fetch_evaluation_traces(entity, project):
    """Fetch all rollout calls which contain the evaluation data."""
    print(f"🔄 Connecting to {entity}/{project}...")
    
    client = weave.init(f"{entity}/{project}")
    
    print("🔄 Fetching all calls (this may take a minute)...")
    calls = list(client.get_calls(limit=10000))
    
    print(f"✓ Fetched {len(calls)} total calls")
    
    # Filter for rollout calls - these are the agent execution traces
    rollout_calls = [c for c in calls if "rollout" in (c.op_name if hasattr(c, 'op_name') else "").lower()]
    
    print(f"✓ Found {len(rollout_calls)} rollout calls")
    
    return rollout_calls, client


def organize_by_model(calls):
    """Organize traces by model type."""
    print("\n🔍 Organizing by model type...")
    
    by_model = defaultdict(list)
    
    for call in calls:
        # Extract model info from inputs
        if not hasattr(call, 'inputs') or not call.inputs:
            continue
        
        inputs = call.inputs
        if not isinstance(inputs, dict) or 'model' not in inputs:
            continue
        
        model_obj = inputs['model']
        
        # Extract model name from the WeaveObject
        model_name = None
        if hasattr(model_obj, 'name'):
            model_name = model_obj.name
        elif isinstance(model_obj, dict) and 'name' in model_obj:
            model_name = model_obj['name']
        else:
            model_name = str(model_obj)[:100]
        
        # Categorize by model type based on name
        # Note: Order matters - check Combined first (has both "ruler" and "scorer")
        model_type = None
        if model_name and isinstance(model_name, str):
            if "ruler-and-scorer" in model_name.lower() or "ruler-and-weave-scorers" in model_name.lower():
                model_type = "Combined"
            elif "ruler" in model_name.lower():
                model_type = "RULER"
            elif "independent" in model_name.lower():
                model_type = "Independent"
            elif "qwen" in model_name.lower() or "Qwen" in model_name:
                model_type = "Base"
            elif "gpt" in model_name.lower() or "openai" in model_name.lower():
                model_type = "OpenAI"
        
        if model_type:
            by_model[model_type].append(call)
    
    print(f"\n✓ Found traces for:")
    for model, traces in by_model.items():
        print(f"  - {model}: {len(traces)} traces")
    
    return by_model


def extract_trace_details(call):
    """Extract all relevant details from a rollout call."""
    inputs = call.inputs if hasattr(call, 'inputs') else {}
    output = call.output if hasattr(call, 'output') else {}
    summary = call.summary if hasattr(call, 'summary') else {}
    
    # For rollout calls, scenario info is in email_scenario input
    email_scenario = inputs.get("email_scenario") if isinstance(inputs, dict) else None
    scenario_data = {}
    
    if email_scenario:
        if hasattr(email_scenario, 'scenario'):
            scenario = email_scenario.scenario
            scenario_data = {
                "id": scenario.id if hasattr(scenario, 'id') else None,
                "question": scenario.question if hasattr(scenario, 'question') else None,
                "answer": scenario.answer if hasattr(scenario, 'answer') else None,
                "message_ids": scenario.message_ids if hasattr(scenario, 'message_ids') else None,
            }
    
    # Extract trajectory from output
    trajectory = output if output else None
    trajectory_details = None
    model_answer = None
    source_ids_retrieved = None
    
    if trajectory and hasattr(trajectory, 'final_answer'):
        if trajectory.final_answer:
            model_answer = trajectory.final_answer.answer if hasattr(trajectory.final_answer, 'answer') else None
            source_ids_retrieved = trajectory.final_answer.source_ids if hasattr(trajectory.final_answer, 'source_ids') else None
        
        trajectory_details = {
            "has_final_answer": bool(trajectory.final_answer),
            "final_answer": {
                "answer": model_answer,
                "source_ids": source_ids_retrieved
            },
            "tool_evaluations": None,
            "num_steps": None,
            "metrics": {}
        }
        
        # Extract metrics from trajectory
        if hasattr(trajectory, 'metrics') and trajectory.metrics:
            metrics = trajectory.metrics
            trajectory_details["metrics"] = {
                "correct": metrics.get("correct") if isinstance(metrics, dict) else None,
                "retrieved_correct_sources": metrics.get("retrieved_correct_sources") if isinstance(metrics, dict) else None,
                "tool_optimal_rate": metrics.get("tool_optimal_rate") if isinstance(metrics, dict) else None,
            }
        
        # Extract tool evaluations
        if hasattr(trajectory, 'tool_evaluations') and trajectory.tool_evaluations:
            trajectory_details["tool_evaluations"] = [
                {
                    "label": e.get("label") if isinstance(e, dict) else None,
                    "tool_name": e.get("tool_name") if isinstance(e, dict) else None,
                }
                for e in trajectory.tool_evaluations
            ]
            trajectory_details["num_steps"] = len(trajectory.tool_evaluations)
    
    return {
        "call_id": call.id if hasattr(call, 'id') else None,
        "op_name": call.op_name if hasattr(call, 'op_name') else None,
        "scenario_id": scenario_data.get("id"),
        "question": scenario_data.get("question"),
        "expected_answer": scenario_data.get("answer"),
        "message_ids": scenario_data.get("message_ids"),
        "model_answer": model_answer,
        "source_ids_retrieved": source_ids_retrieved,
        "trajectory_details": trajectory_details,
        "scores": {
            "correctness": trajectory_details["metrics"].get("correct") if trajectory_details and "metrics" in trajectory_details else None,
            "correctness_reasoning": None,  # Not available in rollout output
            "tool_optimal_rate": trajectory_details["metrics"].get("tool_optimal_rate") if trajectory_details and "metrics" in trajectory_details else None,
            "source_retrieval": trajectory_details["metrics"].get("retrieved_correct_sources") if trajectory_details and "metrics" in trajectory_details else None,
        }
    }


def find_interesting_comparisons(by_model):
    """Find interesting cases where models differ."""
    print("\n🔍 Finding interesting comparative cases...")
    
    # Get Combined and RULER traces
    combined_calls = by_model.get("Combined", [])
    ruler_calls = by_model.get("RULER", [])
    
    print(f"  - Combined: {len(combined_calls)} traces")
    print(f"  - RULER: {len(ruler_calls)} traces")
    
    # Extract details and index by scenario
    print("\n🔄 Extracting trace details...")
    combined_traces = {}
    for call in combined_calls:
        details = extract_trace_details(call)
        scenario_id = details["scenario_id"]
        if scenario_id:
            combined_traces[scenario_id] = details
    
    ruler_traces = {}
    for call in ruler_calls:
        details = extract_trace_details(call)
        scenario_id = details["scenario_id"]
        if scenario_id:
            ruler_traces[scenario_id] = details
    
    # Find common scenarios
    common = set(combined_traces.keys()) & set(ruler_traces.keys())
    print(f"\n✓ Found {len(common)} scenarios evaluated by both models")
    
    # Categorize interesting cases
    cases = {
        "combined_correct_ruler_wrong": [],
        "ruler_correct_combined_wrong": [],
        "both_correct_different_paths": [],
        "both_wrong": [],
        "combined_better_tools": [],
        "ruler_better_tools": []
    }
    
    for scenario_id in common:
        combined = combined_traces[scenario_id]
        ruler = ruler_traces[scenario_id]
        
        c_correct = combined["scores"]["correctness"]
        r_correct = ruler["scores"]["correctness"]
        
        c_tool = combined["scores"]["tool_optimal_rate"]
        r_tool = ruler["scores"]["tool_optimal_rate"]
        
        comparison = {
            "scenario_id": scenario_id,
            "question": combined["question"],
            "expected_answer": combined["expected_answer"],
            "combined": combined,
            "ruler": ruler
        }
        
        # Categorize
        if c_correct == 1.0 and r_correct == 0.0:
            cases["combined_correct_ruler_wrong"].append(comparison)
        elif r_correct == 1.0 and c_correct == 0.0:
            cases["ruler_correct_combined_wrong"].append(comparison)
        elif c_correct == 1.0 and r_correct == 1.0:
            cases["both_correct_different_paths"].append(comparison)
        elif c_correct == 0.0 and r_correct == 0.0:
            cases["both_wrong"].append(comparison)
        
        # Tool usage comparisons
        if c_tool is not None and r_tool is not None:
            if c_tool > r_tool:
                cases["combined_better_tools"].append(comparison)
            elif r_tool > c_tool:
                cases["ruler_better_tools"].append(comparison)
    
    print("\n📊 Categorized cases:")
    for category, examples in cases.items():
        print(f"  - {category}: {len(examples)} cases")
    
    return cases


def save_results(cases, output_file="interesting_traces.json"):
    """Save interesting cases to JSON for analysis."""
    print(f"\n💾 Saving results to {output_file}...")
    
    # Convert to JSON-serializable format
    serializable = {}
    for category, examples in cases.items():
        serializable[category] = examples
    
    with open(output_file, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    
    print(f"✓ Saved {sum(len(v) for v in cases.values())} total examples")


def print_summary(cases):
    """Print a summary of interesting findings."""
    print("\n" + "="*80)
    print("SUMMARY OF INTERESTING CASES")
    print("="*80)
    
    combined_wins = cases["combined_correct_ruler_wrong"]
    ruler_wins = cases["ruler_correct_combined_wrong"]
    
    print(f"\n🎯 CORRECTNESS DIFFERENCES:")
    print(f"  Combined correct, RULER wrong: {len(combined_wins)} cases")
    print(f"  RULER correct, Combined wrong: {len(ruler_wins)} cases")
    print(f"  Advantage: {'Combined' if len(combined_wins) > len(ruler_wins) else 'RULER'} by {abs(len(combined_wins) - len(ruler_wins))} cases")
    
    if combined_wins:
        print(f"\n📝 Sample: Combined correct, RULER wrong")
        example = combined_wins[0]
        print(f"  Question: {example['question'][:100]}...")
        print(f"  Expected: {example['expected_answer'][:100]}...")
        print(f"  Combined answer: {example['combined']['model_answer'][:100]}...")
        print(f"  RULER answer: {example['ruler']['model_answer'][:100]}...")
    
    if ruler_wins:
        print(f"\n📝 Sample: RULER correct, Combined wrong")
        example = ruler_wins[0]
        print(f"  Question: {example['question'][:100]}...")
        print(f"  Expected: {example['expected_answer'][:100]}...")
        print(f"  RULER answer: {example['ruler']['model_answer'][:100]}...")
        print(f"  Combined answer: {example['combined']['model_answer'][:100]}...")
    
    print(f"\n🔧 TOOL USAGE PATTERNS:")
    print(f"  Both correct, different paths: {len(cases['both_correct_different_paths'])} cases")
    print(f"  Combined better tool usage: {len(cases['combined_better_tools'])} cases")
    print(f"  RULER better tool usage: {len(cases['ruler_better_tools'])} cases")
    
    print("\n" + "="*80)
    print("✅ Data collection complete!")
    print("Next steps:")
    print("  1. Review interesting_traces.json")
    print("  2. Manually inspect specific examples")
    print("  3. Identify patterns in the data")
    print("  4. Write analysis based on observations")
    print("="*80 + "\n")


def main():
    """Main data collection workflow."""
    print("="*80)
    print("INTERESTING TRACE COLLECTION")
    print("Collecting data for qualitative analysis")
    print("="*80 + "\n")
    
    # Fetch traces
    calls, client = fetch_evaluation_traces("wandb", "demo-project-qwen-email-agent-with-art-weave-models")
    
    # Organize by model
    by_model = organize_by_model(calls)
    
    # Find interesting cases
    cases = find_interesting_comparisons(by_model)
    
    # Save for analysis
    save_results(cases)
    
    # Print summary
    print_summary(cases)


if __name__ == "__main__":
    main()

