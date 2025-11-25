import wandb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Test forking from a rewound run.
    
    This demonstrates the combination of rewind and fork:
    1. Rewind modifies the run history (same run ID)
    2. Fork creates a NEW branch from the rewound run
    """
    project = "test-rewind-run"
    entity = "wandb"
    
    # The run ID from the previous rewind script
    # Replace this with the actual run ID from test_rewind_run.py output
    rewind_run_id = "5h6r2uqj"  # This should be the run ID from the rewound run
    
    print("=== Forking from rewound run ===")
    print(f"Source run: {rewind_run_id}")
    print(f"Forking from step 250 (where spikes began)")
    print(f"This creates a NEW run with a different trajectory\n")
    
    # Fork the run from step 250 (where the spikey pattern began)
    forked_run = wandb.init(
        project=project,
        entity=entity,
        name="forked-alternative-path",
        fork_from=f"{rewind_run_id}?_step=250",
    )
    
    print(f"✅ Forked run created: {forked_run.id}")
    print(f"   This is a NEW run branching from step 250")
    
    # Continue logging in the new run with a different pattern (multiplied by 3)
    print("\nLogging alternative trajectory:")
    for i in range(250, 400):
        # Different pattern: multiply by 3
        forked_run.log({"metric": i * 3})
        if i % 50 == 0:
            print(f"  Step {i}: metric={i * 3}")
    
    forked_run.finish()
    
    print(f"\n✅ Fork completed!")
    print(f"   Forked run ID: {forked_run.id} (NEW run)")
    print(f"   Source run ID: {rewind_run_id} (original)")
    print(f"\n📊 View the forked run at: https://wandb.ai/{entity}/{project}/runs/{forked_run.id}")
    print(f"💡 Compare runs:")
    print(f"   - Original rewound run: https://wandb.ai/{entity}/{project}/runs/{rewind_run_id}")
    print(f"   - Forked alternative: https://wandb.ai/{entity}/{project}/runs/{forked_run.id}")

if __name__ == "__main__":
    main()

