"""
Comprehensive analysis of rollout data from two training runs.
Focuses on real patterns, failure modes, and success patterns.
"""
from dotenv import load_dotenv
import weave
from collections import defaultdict

# Load environment variables
load_dotenv()

def extract_tool_sequence(output):
    """Extract the sequence of tool calls from a trajectory."""
    tools = []
    if hasattr(output, 'messages_and_choices') and output.messages_and_choices:
        for msg_choice in output.messages_and_choices:
            if hasattr(msg_choice, 'message'):
                msg = msg_choice.message
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if hasattr(tc, 'function'):
                            tools.append({
                                'name': tc.function.name if hasattr(tc.function, 'name') else 'unknown',
                                'arguments': tc.function.arguments if hasattr(tc.function, 'arguments') else {}
                            })
    return tools

def extract_question(call):
    """Extract question from call inputs."""
    try:
        if hasattr(call, 'inputs') and call.inputs:
            if hasattr(call.inputs, 'scenario'):
                return call.inputs.scenario.scenario.question
    except:
        pass
    return ""

def extract_answer(output):
    """Extract answer from output."""
    try:
        if hasattr(output, 'final_answer') and output.final_answer:
            return output.final_answer.answer
    except:
        pass
    return ""

def main():
    client = weave.init('wandb/email-agent-openpipe-weave-models-demo')
    
    # Fetch from both runs
    run_ids = [
        'wandb/email-agent-openpipe-weave-models-demo/a8qs0y9o',
        'wandb/email-agent-openpipe-weave-models-demo/d05roj12'
    ]
    
    # Store data by run
    run_data = defaultdict(lambda: {
        'calls': [],
        'correct': [],
        'incorrect': [],
        'tool_counts': [],
        'retrieved_sources': []
    })
    
    for run_id in run_ids:
        print(f"\n{'='*80}")
        print(f"Fetching data from run: {run_id.split('/')[-1]}")
        print(f"{'='*80}")
        
        calls = client.get_calls(
            filter={
                'op_names': ['weave:///wandb/email-agent-openpipe-weave-models-demo/op/rollout:*'],
                'wb_run_ids': [run_id]
            },
            sort_by=[{'field': 'started_at', 'direction': 'desc'}],
            limit=50
        )
        
        calls_list = list(calls)
        print(f"Fetched {len(calls_list)} calls")
        
        run_short = run_id.split('/')[-1]
        
        # Analyze each call
        for call in calls_list:
            if not call.output:
                continue
            
            output = call.output
            
            # Extract metrics
            correct = output.metrics.get('correct', 0) if hasattr(output, 'metrics') else 0
            retrieved = output.metrics.get('retrieved_correct_sources', 0) if hasattr(output, 'metrics') else 0
            tool_optimal = output.metrics.get('tool_optimal_rate', 0) if hasattr(output, 'metrics') else 0
            
            # Extract tool sequence
            tools = extract_tool_sequence(output)
            
            # Extract question and answer
            question = extract_question(call)
            answer = extract_answer(output)
            
            # Store trajectory data
            traj_data = {
                'call_id': call.id,
                'correct': correct,
                'retrieved': retrieved,
                'tool_optimal': tool_optimal,
                'num_tools': len(tools),
                'tools': tools,
                'question': question,
                'answer': answer,
                'tool_evaluations': output.tool_evaluations if hasattr(output, 'tool_evaluations') else []
            }
            
            run_data[run_short]['calls'].append(traj_data)
            run_data[run_short]['tool_counts'].append(len(tools))
            
            if correct == 1:
                run_data[run_short]['correct'].append(traj_data)
            else:
                run_data[run_short]['incorrect'].append(traj_data)
            
            if retrieved == 1:
                run_data[run_short]['retrieved_sources'].append(traj_data)
    
    # Now analyze patterns for each run
    print(f"\n\n{'='*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*80}\n")
    
    for run_short in run_data.keys():
        data = run_data[run_short]
        total = len(data['calls'])
        
        if total == 0:
            continue
        
        print(f"\n{'='*80}")
        print(f"RUN: {run_short}")
        print(f"{'='*80}")
        
        print(f"\nOverall Statistics:")
        print(f"  Total trajectories: {total}")
        print(f"  Correct: {len(data['correct'])} ({len(data['correct'])/total*100:.1f}%)")
        print(f"  Incorrect: {len(data['incorrect'])} ({len(data['incorrect'])/total*100:.1f}%)")
        print(f"  Retrieved sources: {len(data['retrieved_sources'])} ({len(data['retrieved_sources'])/total*100:.1f}%)")
        
        if data['tool_counts']:
            avg_tools = sum(data['tool_counts']) / len(data['tool_counts'])
            print(f"  Avg tool calls: {avg_tools:.2f}")
            print(f"  Max tool calls: {max(data['tool_counts'])}")
        
        # Analyze correct trajectories
        print(f"\n{'='*80}")
        print(f"SUCCESS PATTERNS (Run {run_short})")
        print(f"{'='*80}")
        
        if data['correct']:
            # Find efficient correct ones
            efficient = [t for t in data['correct'] if t['num_tools'] <= 3 and t['num_tools'] > 0]
            print(f"\nFound {len(efficient)} efficient successful trajectories (≤3 tools)")
            
            for i, traj in enumerate(efficient[:3], 1):
                print(f"\nExample {i}:")
                print(f"  Tool calls: {traj['num_tools']}")
                print(f"  Tool optimal: {traj['tool_optimal']:.2f}")
                print(f"  Retrieved sources: {'Yes' if traj['retrieved'] == 1 else 'No'}")
                print(f"  Question: {traj['question'][:120]}...")
                print(f"  Answer: {traj['answer'][:120]}...")
                
                if traj['tools']:
                    print(f"  Tool sequence:")
                    for j, tool in enumerate(traj['tools'], 1):
                        print(f"    {j}. {tool['name']}")
                
                # Show tool evaluations if available
                if traj['tool_evaluations']:
                    print(f"  Tool evaluations:")
                    eval_list = list(traj['tool_evaluations'])[:3]
                    for eval in eval_list:
                        label = eval.get('label', 'unknown') if hasattr(eval, 'get') else getattr(eval, 'label', 'unknown')
                        reasoning = eval.get('reasoning', '') if hasattr(eval, 'get') else getattr(eval, 'reasoning', '')
                        print(f"    - {label}: {str(reasoning)[:80]}...")
        
        # Analyze incorrect trajectories
        print(f"\n{'='*80}")
        print(f"FAILURE PATTERNS (Run {run_short})")
        print(f"{'='*80}")
        
        if data['incorrect']:
            # Categorize failures
            failed_retrieval = [t for t in data['incorrect'] if t['retrieved'] == 0]
            failed_despite_retrieval = [t for t in data['incorrect'] if t['retrieved'] == 1]
            many_tools = [t for t in data['incorrect'] if t['num_tools'] >= 5]
            
            print(f"\nFailure breakdown:")
            print(f"  Failed to retrieve sources: {len(failed_retrieval)} ({len(failed_retrieval)/len(data['incorrect'])*100:.1f}%)")
            print(f"  Retrieved but still wrong: {len(failed_despite_retrieval)} ({len(failed_despite_retrieval)/len(data['incorrect'])*100:.1f}%)")
            print(f"  Used many tools (≥5): {len(many_tools)}")
            
            # Show examples of each failure type
            print(f"\nPattern 1: Failed Retrieval")
            for i, traj in enumerate(failed_retrieval[:2], 1):
                print(f"\nExample {i}:")
                print(f"  Tool calls: {traj['num_tools']}")
                print(f"  Tool optimal: {traj['tool_optimal']:.2f}")
                print(f"  Question: {traj['question'][:120]}...")
                print(f"  Answer: {traj['answer'][:120]}...")
                
                if traj['tools']:
                    print(f"  Tool sequence:")
                    for j, tool in enumerate(traj['tools'][:5], 1):
                        print(f"    {j}. {tool['name']}")
                
                if traj['tool_evaluations']:
                    print(f"  Tool evaluations:")
                    eval_list = list(traj['tool_evaluations'])[:2]
                    for eval in eval_list:
                        label = eval.get('label', 'unknown') if hasattr(eval, 'get') else getattr(eval, 'label', 'unknown')
                        reasoning = eval.get('reasoning', '') if hasattr(eval, 'get') else getattr(eval, 'reasoning', '')
                        print(f"    - {label}: {str(reasoning)[:80]}...")
            
            if failed_despite_retrieval:
                print(f"\nPattern 2: Retrieved Sources But Still Incorrect")
                for i, traj in enumerate(failed_despite_retrieval[:2], 1):
                    print(f"\nExample {i}:")
                    print(f"  Tool calls: {traj['num_tools']}")
                    print(f"  Tool optimal: {traj['tool_optimal']:.2f}")
                    print(f"  Question: {traj['question'][:120]}...")
                    print(f"  Answer: {traj['answer'][:120]}...")
                    
                    if traj['tool_evaluations']:
                        print(f"  Tool evaluations:")
                        eval_list = list(traj['tool_evaluations'])[:2]
                        for eval in eval_list:
                            label = eval.get('label', 'unknown') if hasattr(eval, 'get') else getattr(eval, 'label', 'unknown')
                            reasoning = eval.get('reasoning', '') if hasattr(eval, 'get') else getattr(eval, 'reasoning', '')
                            print(f"    - {label}: {str(reasoning)[:80]}...")
    
    print(f"\n\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

