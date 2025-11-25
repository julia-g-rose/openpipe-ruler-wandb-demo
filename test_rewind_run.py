import wandb
import math
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Test W&B run rewind functionality using resume_from.
    
    Rewind allows you to "go back in time" to a specific step in a run's history,
    archive the original data, and log new data from that point forward.
    
    Key differences:
    - resume: Continue a crashed run with the same ID
    - fork_from: Create a NEW run branching off from an existing run
    - resume_from: Rewind the SAME run to a specific step (archives original)
    """
    project = "test-rewind-run"
    entity = "wandb"
    
    # Create initial run with 300 steps
    print("=== Part 1: Creating initial run with 300 steps ===")
    run1 = wandb.init(project=project, entity=entity, name="baseline-metrics")
    for i in range(300):
        run1.log({"metric": i})
        if i % 50 == 0:
            print(f"  Step {i}: metric={i}")
    run1.finish()
    print(f"✅ Initial run completed: {run1.id}")
    print(f"   Final metric value: 299")
    
    # Rewind from step 200 and log new data
    print(f"\n=== Part 2: Rewinding run {run1.id} to step 200 ===")
    print("This will:")
    print("  - Archive the original run (steps 200-299)")
    print("  - Reset the run history to step 200")
    print("  - Allow logging new data from step 200")
    
    run2 = wandb.init(
        project=project, 
        entity=entity, 
        name="rewound-with-spikes",
        resume_from=f"{run1.id}?_step=200"
    )
    
    print(f"✅ Rewound to step 200, now logging new data...")
    
    # Continue logging from step 200 with modified behavior
    for i in range(200, 300):
        if i < 250:
            # Keep baseline behavior until step 250
            run2.log({"metric": i, "step": i})
            if i % 25 == 0:
                print(f"  Step {i}: metric={i} (baseline)")
        else:
            # Introduce spikey pattern starting from step 250
            subtle_spike = i + (2 * math.sin(i / 3.0))
            run2.log({"metric": subtle_spike, "step": i})
            if i % 25 == 0:
                print(f"  Step {i}: metric={subtle_spike:.2f} (spikey)")
        
        # Additionally log a new metric at all steps
        run2.log({"additional_metric": i * 1.1, "step": i})
    
    run2.finish()
    print(f"\n✅ Rewind completed!")
    print(f"   Run ID: {run2.id}")
    print(f"   The run now shows:")
    print(f"   - Steps 0-199: Original data (preserved)")
    print(f"   - Steps 200-299: New data with spikes")
    print(f"   - Additional metric logged for steps 200-299")
    print(f"\n📊 View the run at: https://wandb.ai/{entity}/{project}/runs/{run2.id}")
    print(f"💡 Check the 'Overview' tab to see the archived original run")

if __name__ == "__main__":
    main()

