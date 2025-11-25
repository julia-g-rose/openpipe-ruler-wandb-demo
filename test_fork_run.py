import wandb
import math
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Test W&B run forking functionality.
    
    Creates a baseline run, then forks from it at step 200 to create
    a divergent path with modified metrics.
    """
    project = "test-fork-run"
    entity = "wandb"
    
    # Create baseline run with 300 steps
    print("Creating baseline run...")
    run1 = wandb.init(project=project, entity=entity, name="baseline-run")
    for i in range(300):
        run1.log({"metric": i})
    run1.finish()
    print(f"Baseline run completed: {run1.id}")
    
    # Fork from step 200 and create divergent behavior
    print(f"\nForking from run {run1.id} at step 200...")
    run2 = wandb.init(
        project=project, 
        entity=entity, 
        name="forked-run",
        fork_from=f"{run1.id}?_step=200"
    )
    
    # Continue from step 200 with modified behavior
    for i in range(200, 300):
        if i < 250:
            # Maintain baseline behavior until step 250
            run2.log({"metric": i})
        else:
            # Introduce spikey pattern from step 250
            subtle_spike = i + (2 * math.sin(i / 3.0))
            run2.log({"metric": subtle_spike})
        
        # Log additional metric throughout
        run2.log({"additional_metric": i * 1.1})
    
    run2.finish()
    print(f"Forked run completed: {run2.id}")
    print(f"\nView runs at: https://wandb.ai/{entity}/{project}")

if __name__ == "__main__":
    main()
