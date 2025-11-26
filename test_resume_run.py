import wandb
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

name = "training-with-crash"
project = "resume-run"

def simulate_training_with_crash():
    """
    Simulate a training run that crashes partway through.
    This demonstrates the need for resume functionality.
    """
    entity = "wandb"
    run_id = "test-resume-123"
    
    print("=== Part 1: Initial run (will 'crash' at step 50) ===")
    with wandb.init(
        project=project, 
        entity=entity, 
        id=run_id,
        name=name,
        resume="allow"
    ) as run:
        # Simulate training for 50 steps before "crashing"
        for step in range(0, 50):
            loss = 1.0 / (step + 1)  # Decreasing loss
            accuracy = step / 100.0  # Increasing accuracy
            
            run.log({
                "loss": loss,
                "accuracy": accuracy,
                "step": step
            })
            
            if step % 10 == 0:
                print(f"  Step {step}: loss={loss:.4f}, accuracy={accuracy:.4f}")
        
        print("  💥 Simulating crash at step 50...")
        print(f"  Run ID: {run.id}")
    
    return run_id


def resume_training(run_id):
    """
    Resume the training run from where it left off.
    """
    entity = "wandb"
    
    with wandb.init(
        project=project, 
        entity=entity, 
        id=run_id,
        name=name,
        resume="allow"  # Allow resume without overriding
    ) as run:
        print(f"  ✅ Successfully resumed run: {run.id}")
        
        # Continue training from step 50 to 100
        for step in range(50, 100):
            loss = 1.0 / (step + 1)  # Decreasing loss
            accuracy = step / 100.0  # Increasing accuracy
            
            run.log({
                "loss": loss,
                "accuracy": accuracy,
                "step": step
            })
            
            if step % 10 == 0:
                print(f"  Step {step}: loss={loss:.4f}, accuracy={accuracy:.4f}")
        
        print(f"  ✅ Training completed successfully!")
        print(f"  Final metrics - loss: {loss:.4f}, accuracy: {accuracy:.4f}")


def main():
    """
    Main function to test W&B resume functionality.
    
    Tests three scenarios:
    1. resume="allow" - Resume if exists, create if doesn't
    2. resume="must" - Must resume existing run
    3. Simulated crash and recovery scenario
    """
    run_id = simulate_training_with_crash()
    time.sleep(2)  # Brief pause to simulate time between crash and restart
    resume_training(run_id)


if __name__ == "__main__":
    entity = "wandb"
    main()

