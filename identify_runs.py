"""
Identify which reward strategy each run used.
"""
from dotenv import load_dotenv
import wandb

load_dotenv()

def main():
    api = wandb.Api()
    
    run_ids = ['a8qs0y9o', 'd05roj12']
    
    for run_id in run_ids:
        run = api.run(f"wandb/email-agent-openpipe-weave-models-demo/{run_id}")
        
        print(f"\n{'='*80}")
        print(f"Run ID: {run_id}")
        print(f"{'='*80}")
        print(f"Name: {run.name}")
        print(f"Job Type: {run.config.get('wandb_job_type', 'N/A')}")
        print(f"Model Name: {run.config.get('model_name', 'N/A')}")
        
        # Check for RULER-specific config
        if 'ruler_judge_model' in run.config:
            print(f"RULER Judge Model: {run.config['ruler_judge_model']}")
            print("Reward Strategy: RULER (trains reward model to predict correctness)")
        
        # Check if model name contains 'independent'
        if 'independent' in run.config.get('model_name', '').lower():
            print("Reward Strategy: Independent Reward (weighted combination of metrics)")
        
        print(f"\nConfig keys: {list(run.config.keys())[:20]}")

if __name__ == "__main__":
    main()

