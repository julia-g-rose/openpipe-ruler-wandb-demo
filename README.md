# Email Search Agent - OpenPipe ART Demo

This project demonstrates training an email search agent using OpenPipe's ART (Automated Reinforcement Learning Trainer) framework with the Enron email dataset.

## Project Structure

### Core Scripts

- **`train.py`** - Main training script
  - Initializes the model and serverless backend
  - Runs the training loop with RULER scoring
  - Performs validation at regular intervals
  - Configurable training parameters (epochs, learning rate, etc.)

- **`evaluate.py`** - Model evaluation script  
  - Tests trained models on validation and test scenarios
  - Provides detailed output of agent reasoning
  - Calculates accuracy metrics
  - Supports evaluation on subsets of data

- **`compare_models.py`** - Multi-model comparison using Weave evaluations
  - Compares 3 models: gpt-4o-mini, gpt-4o, and OpenPipe/Qwen base model
  - Uses all 3 Weave scorers (Correctness, Source Retrieval, Tool Usage)
  - Creates Weave leaderboards for easy comparison
  - Logs metrics for parallel coordinates visualization
  - Evaluates on full validation dataset

- **`helpers.py`** - Shared utilities and functions
  - `rollout()` - Core function that executes the agent on a scenario
  - `CorrectnessJudgeScorer` - Weave scorer for LLM-based correctness evaluation
  - `SourceRetrievalScorer` - Weave scorer for evaluating source email retrieval quality
  - `ToolUsageScorer` - Weave scorer for LLM-based evaluation of each tool call decision (applied at every step)
  - `print_trajectory()` - Pretty-prints agent trajectories
  - **Weave Prompts**: All LLM judge prompts are version-controlled using Weave's prompt management
  - Data models: `EmailScenario`, `ProjectTrajectory`, `CorrectnessJudgeResponse`, `ToolUsageJudgeResponse`

### Supporting Files

- **`enron_helpers.py`** - Enron dataset utilities and email search functions
- **`save_all_dataset_objects_to_wandb.py`** - ⭐ **Recommended**: All-in-one script to save datasets to W&B Artifacts, Weave Datasets, and W&B Tables
- **`save_datasets_to_weave.py`** - Save datasets to Weave for tracking
- **`save_artifacts_to_wandb.py`** - Save datasets as W&B artifacts
- **`save_artifacts_to_wandb_csv.py`** - Save datasets as W&B artifacts in CSV format
- **`log_artifacts_to_run.py`** - Log artifacts to specific W&B runs
- **`log_table_to_run.py`** - Log data tables to W&B runs

### Configuration Files

- **`config.yaml`** - Training configuration (hyperparameters, model settings, W&B config)

### Data Files

- `training_scenarios.json/csv` - Training dataset scenarios
- `validation_scenarios.json/csv` - Validation dataset scenarios

## Usage

### Prepare Dataset Artifacts (First Time Only)

Before training or evaluation, create the dataset artifacts:

```bash
# Recommended: Create all dataset objects at once (Artifacts, Weave Datasets, and Tables)
python save_all_dataset_objects_to_wandb.py

# Alternative: Create only W&B artifacts
python save_artifacts_to_wandb.py
```

The recommended script (`save_all_dataset_objects_to_wandb.py`) creates everything in a single W&B run:
- ✅ **W&B Artifacts** - Used by training and evaluation scripts
- ✅ **Weave Datasets** - Published to Weave for LLM observability
- ✅ **W&B Tables** - Visualize scenarios in the W&B UI
- ✅ **Summary Statistics** - Dataset counts and metadata

The number of scenarios loaded is controlled by `training_dataset_size` and `validation_dataset_size` in `config.yaml`.

**Note**: If you change the dataset configuration (size or seed), re-run the script to create new versions with the updated data.

### Training

```bash
# Use default config.yaml
python train.py

# Or specify a custom config file
python train.py --config my_custom_config.yaml
```

This will:
1. Load configuration from the YAML file
2. Download and load dataset artifacts from W&B
3. Initialize the email agent model
4. Register with the serverless backend
5. Run training for all configured epochs
6. Save checkpoint metadata as W&B model artifacts after each step
7. Log training metrics and validation results to W&B

**Note**: Training runs until all epochs complete based on the dataset size and `num_epochs` configuration. All checkpoints are preserved and logged to W&B.

### Evaluation

```bash
# Use default config.yaml
python evaluate.py

# Or specify a custom config file
python evaluate.py --config my_custom_config.yaml
```

This will:
1. Download and load dataset artifacts from W&B
2. Load the trained model
3. Evaluate on test scenarios
4. Print detailed trajectories and results
5. Calculate accuracy metrics

### Model Comparison with Weave Leaderboards

Compare multiple models using Weave evaluations and all three scorers:

```bash
# Use default config.yaml
python compare_models.py

# Or specify a custom config file
python compare_models.py --config my_custom_config.yaml
```

This will:
1. Load the validation dataset from W&B
2. Set up three models:
   - `gpt-4o-mini` (OpenAI)
   - `gpt-4o` (OpenAI)
   - OpenPipe/Qwen base model (before fine-tuning)
3. Run all 3 Weave scorers on each model:
   - **CorrectnessJudgeScorer** - LLM judge for answer correctness
   - **SourceRetrievalScorer** - Precision/recall for source emails
   - **ToolUsageScorer** - LLM judge for tool call appropriateness
4. Create a Weave leaderboard with all metrics
5. Log results to W&B for parallel coordinates visualization

**Viewing Results:**

1. **Weave Leaderboard**: Visit `https://wandb.ai/[entity]/[project]-model-comparison/weave/leaderboards`
2. **Parallel Coordinates**: 
   - Go to your W&B run page
   - Click "Add Panel" → "Parallel Coordinates"
   - Select metrics to compare across models
   - Color by model name to see patterns

**Key Metrics Tracked:**
- `correct` - Answer correctness (0.0-1.0)
- `retrieved_correct_sources` - Whether all expected sources were retrieved
- `tool_optimal_rate` - % of optimal tool decisions

## Configuration

All project configuration is stored in `config.yaml` and used across all scripts in the repository:

```yaml
# Model configuration
project: "julia-openpipe-wandb-email-agent-demo-v8"
base_model: "OpenPipe/Qwen3-14B-Instruct"

# Dataset configuration
training_dataset_size: 50  # Number of training scenarios to load
validation_dataset_size: 20  # Number of validation scenarios to load
dataset_seed: 42  # Random seed for dataset shuffling

# Dataset artifacts
training_dataset_artifact: "enron-training-scenarios:latest"
validation_dataset_artifact: "enron-validation-scenarios:latest"

# Judge models
ruler_judge_model: "openai/o4-mini"  # Used by RULER for scoring trajectories during training
correctness_judge_model: "openai/gpt-4o"  # Used for evaluating answer correctness
tool_judge_model: "openai/gpt-4o"  # Used for evaluating tool call appropriateness

# Training hyperparameters
groups_per_step: 2
num_epochs: 20
rollouts_per_group: 4
learning_rate: 1.0e-5

# Validation and evaluation
validation_step_interval: 5

# Checkpointing
save_checkpoint_artifact: true  # Save checkpoints as W&B artifacts

# Reproducibility
random_seed: 42

# W&B run configuration
wandb_run_name: "email-agent-training"
wandb_job_type: "train"
```

### Dataset Configuration

Dataset sizes and random seed are centrally configured:

- **`training_dataset_size`**: Number of training scenarios to load from the Enron dataset (default: 50)
- **`validation_dataset_size`**: Number of validation scenarios to load (default: 20)
- **`dataset_seed`**: Random seed for shuffling scenarios (default: 42)

These settings are used consistently across all dataset preparation scripts (`save_artifacts_to_wandb.py`, `save_datasets_to_weave.py`, etc.), ensuring that all artifact versions use the same data configuration.

### Training Hyperparameters

Training duration is determined by the dataset size and configuration:

- **Total training steps** = `(training_dataset_size / groups_per_step) * num_epochs`
- For the default config: `(50 / 2) * 20 = 500 steps`
- Training runs until all epochs are complete (no arbitrary max_steps limit)

Key parameters:
- **`groups_per_step`**: Number of scenario groups processed per training step
- **`num_epochs`**: Total number of passes through the training dataset
- **`rollouts_per_group`**: Number of trajectories generated per scenario for RULER comparison
- **`learning_rate`**: Learning rate for model updates

### Dataset Artifacts

The training and evaluation scripts use W&B dataset artifacts to ensure reproducibility and proper lineage tracking:

- **`training_dataset_artifact`**: W&B artifact containing training scenarios (e.g., `enron-training-scenarios:latest`)
- **`validation_dataset_artifact`**: W&B artifact containing validation scenarios (e.g., `enron-validation-scenarios:latest`)

During training and evaluation, these artifacts are automatically:
1. **Marked as inputs** using `run.use_artifact()` - Creates clear lineage showing which datasets were used
2. **Downloaded** to local cache - Ensures consistent data across runs
3. **Loaded into memory** - Scenarios are parsed from JSON and validated

This approach provides:
- **Reproducibility**: Exact dataset versions are tracked with each run
- **Lineage**: W&B shows which datasets produced which models
- **Versioning**: Use specific versions (`:v1`) or always latest (`:latest`)
- **Caching**: Artifacts are cached locally to speed up subsequent runs

To create/update dataset artifacts, run:
```bash
python save_all_dataset_objects_to_wandb.py
```

### Dataset Format Options

The project supports multiple dataset formats, each serving different purposes:

1. **W&B Artifacts** (Primary)
   - Used by training and evaluation scripts
   - Provides versioning and lineage tracking
   - Automatically downloaded and cached
   - **Required for training**

2. **Weave Datasets**
   - Published to W&B Weave for LLM observability
   - Enables advanced querying and filtering
   - Integrates with Weave trace data
   - **Optional but recommended**

3. **W&B Tables**
   - Interactive visualization in W&B UI
   - Easy browsing and filtering of scenarios
   - Useful for data exploration and QA
   - **Optional for visualization**

The unified script (`save_all_dataset_objects_to_wandb.py`) creates all three formats in one run, ensuring consistency across all representations.

### Checkpoint Management

The training script saves checkpoint metadata as W&B artifacts after each training step:

- **`save_checkpoint_artifact`**: When `true`, saves checkpoint metadata as W&B model artifacts
- Checkpoint artifacts include:
  - Training step and epoch information
  - All hyperparameters and configuration
  - Dataset artifact references used for training
  - Judge model configurations

This provides:
- **Full history**: All checkpoints are preserved (not deleted)
- **Reproducibility**: Complete training configuration saved with each checkpoint
- **Lineage**: Clear connection between datasets, checkpoints, and training runs
- **Rollback**: Easy to identify and return to any previous checkpoint

The actual checkpoint files are managed by the ART ServerlessBackend on W&B servers. The artifacts serve as metadata records linked to these checkpoints.

### Judge Models

The project uses two different judge models for different purposes:

- **`ruler_judge_model`**: Used during training by the RULER algorithm to assign relative scores to trajectory groups. This helps the model learn from comparative feedback.
- **`correctness_judge_model`**: Used to evaluate the correctness of final answers by comparing them against reference answers. This provides absolute accuracy metrics.

The configuration is automatically tracked with each training run in W&B, making it easy to:
- Compare experiments with different hyperparameters
- Reproduce training runs exactly
- Track which config produced which results
- Version control configurations separately from code
- Run multiple experiments with different config files

**Note:** All scripts in this repository now use `config.yaml` for the project name and other settings, replacing the previous `WANDB_PROJECT` environment variable. This ensures consistency across training, evaluation, and data preparation scripts.

### Creating Multiple Configurations

You can create different config files for different experiments:

```bash
# Create a config for a high learning rate experiment
cp config.yaml config_high_lr.yaml
# Edit config_high_lr.yaml to change learning_rate to 5e-5

# Run with the new config
python train.py --config config_high_lr.yaml
```

## Requirements

See `requirements.txt` for dependencies.

## Related Work

1. [ART Example Notebook](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/art-e.ipynb)
