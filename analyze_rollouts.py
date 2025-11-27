"""
Analyze rollout data from Weave to find patterns and interesting examples.
"""
from dotenv import load_dotenv
import weave
import json

# Load environment variables
load_dotenv()

def main():
    client = weave.init('wandb/email-agent-openpipe-weave-models-demo')
    
    calls = client.get_calls(
        filter={
            'op_names': ['weave:///wandb/email-agent-openpipe-weave-models-demo/op/rollout:*'],
            'wb_run_ids': [
                'wandb/email-agent-openpipe-weave-models-demo/a8qs0y9o',
                'wandb/email-agent-openpipe-weave-models-demo/d05roj12'
            ]
        },
        sort_by=[{'field': 'started_at', 'direction': 'desc'}],
        limit=100
    )
    
    # Collect all calls
    calls_list = list(calls)
    print(f'Total calls fetched: {len(calls_list)}\n')
    
    # Debug: inspect first call structure
    if calls_list:
        first_call = calls_list[0]
        print("=" * 80)
        print("DEBUG: First call structure")
        print("=" * 80)
        if first_call.output:
            print(f"Output attributes: {dir(first_call.output)}")
            print(f"Output type: {type(first_call.output)}")
            if hasattr(first_call.output, '__dict__'):
                print(f"Output dict keys: {first_call.output.__dict__.keys()}")
        print()
    
    # Analyze patterns
    correct_count = 0
    incorrect_count = 0
    retrieved_source_count = 0
    
    tool_call_counts = []
    correct_with_few_tools = []
    incorrect_with_many_tools = []
    perfect_trajectories = []
    
    for call in calls_list:
        if not call.output:
            continue
            
        output = call.output
        
        # Get metrics
        correct = output.metrics.get('correct', 0) if hasattr(output, 'metrics') else 0
        retrieved = output.metrics.get('retrieved_correct_sources', 0) if hasattr(output, 'metrics') else 0
        tool_optimal = output.metrics.get('tool_optimal_rate', 0) if hasattr(output, 'metrics') else 0
        
        if correct == 1:
            correct_count += 1
        else:
            incorrect_count += 1
            
        if retrieved == 1:
            retrieved_source_count += 1
        
        # Count tool calls - use messages_and_choices or tool_evaluations
        num_tools = 0
        if hasattr(output, 'tool_evaluations') and output.tool_evaluations:
            num_tools = len(output.tool_evaluations)
        elif hasattr(output, 'messages_and_choices') and output.messages_and_choices:
            # Count tool calls in messages
            for msg in output.messages_and_choices:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    num_tools += len(msg.tool_calls)
        
        tool_call_counts.append(num_tools)
        
        # Extract question safely
        question = ''
        try:
            if hasattr(call, 'inputs') and call.inputs:
                if hasattr(call.inputs, 'scenario'):
                    question = call.inputs.scenario.scenario.question
                elif 'scenario' in call.inputs:
                    scenario_data = call.inputs['scenario']
                    if hasattr(scenario_data, 'scenario'):
                        question = scenario_data.scenario.question
        except:
            pass
        
        # Find interesting patterns
        # Pattern 1: Correct with few tools (efficient)
        if correct == 1 and num_tools <= 3:
            correct_with_few_tools.append({
                'call_id': call.id,
                'num_tools': num_tools,
                'tool_optimal': tool_optimal,
                'answer': output.final_answer.answer[:150] if hasattr(output, 'final_answer') and output.final_answer else '',
                'question': question
            })
        
        # Pattern 2: Incorrect with many tools (inefficient/struggling)
        if correct == 0 and num_tools >= 5:
            incorrect_with_many_tools.append({
                'call_id': call.id,
                'num_tools': num_tools,
                'tool_optimal': tool_optimal,
                'answer': output.final_answer.answer[:150] if hasattr(output, 'final_answer') and output.final_answer else '',
                'question': question
            })
        
        # Pattern 3: Perfect trajectories (correct + retrieved + high tool optimal)
        if correct == 1 and retrieved == 1 and tool_optimal >= 0.8:
            perfect_trajectories.append({
                'call_id': call.id,
                'num_tools': num_tools,
                'tool_optimal': tool_optimal,
                'answer': output.final_answer.answer[:150] if hasattr(output, 'final_answer') and output.final_answer else '',
                'question': question
            })
    
    # Print summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Correct answers: {correct_count} ({correct_count/len(calls_list)*100:.1f}%)")
    print(f"Incorrect answers: {incorrect_count} ({incorrect_count/len(calls_list)*100:.1f}%)")
    print(f"Retrieved correct sources: {retrieved_source_count} ({retrieved_source_count/len(calls_list)*100:.1f}%)")
    print(f"\nAverage tool calls per trajectory: {sum(tool_call_counts)/len(tool_call_counts):.2f}")
    print(f"Min tool calls: {min(tool_call_counts)}")
    print(f"Max tool calls: {max(tool_call_counts)}")
    print()
    
    # Print pattern examples
    print("=" * 80)
    print("PATTERN 1: EFFICIENT CORRECT ANSWERS (≤3 tools)")
    print("=" * 80)
    print(f"Found {len(correct_with_few_tools)} examples\n")
    for i, ex in enumerate(correct_with_few_tools[:3], 1):
        print(f"Example {i}:")
        print(f"  Tool calls: {ex['num_tools']}")
        print(f"  Tool optimal rate: {ex['tool_optimal']:.2f}")
        print(f"  Question: {ex['question'][:100]}...")
        print(f"  Answer: {ex['answer']}")
        print()
    
    print("=" * 80)
    print("PATTERN 2: STRUGGLING TRAJECTORIES (≥5 tools, incorrect)")
    print("=" * 80)
    print(f"Found {len(incorrect_with_many_tools)} examples\n")
    for i, ex in enumerate(incorrect_with_many_tools[:3], 1):
        print(f"Example {i}:")
        print(f"  Tool calls: {ex['num_tools']}")
        print(f"  Tool optimal rate: {ex['tool_optimal']:.2f}")
        print(f"  Question: {ex['question'][:100]}...")
        print(f"  Answer: {ex['answer']}")
        print()
    
    print("=" * 80)
    print("PATTERN 3: PERFECT TRAJECTORIES (correct + retrieved + tool_optimal ≥ 0.8)")
    print("=" * 80)
    print(f"Found {len(perfect_trajectories)} examples\n")
    for i, ex in enumerate(perfect_trajectories[:3], 1):
        print(f"Example {i}:")
        print(f"  Tool calls: {ex['num_tools']}")
        print(f"  Tool optimal rate: {ex['tool_optimal']:.2f}")
        print(f"  Question: {ex['question'][:100]}...")
        print(f"  Answer: {ex['answer']}")
        print()

if __name__ == "__main__":
    main()

