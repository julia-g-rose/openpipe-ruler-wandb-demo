# Email Search Agent - OpenPipe ART Demo

This project demonstrates training an email search agent using [OpenPipe's ART (Automated Reinforcement Learning Trainer)](https://docs.openpipe.ai/features/art) framework with the [RULER](https://docs.openpipe.ai/features/art/reward-functions/ruler) reward function and the Enron email dataset. 

The project integrates with [W&B Models](https://docs.wandb.ai/guides/models) for experiment tracking and model management, and [W&B Weave](https://wandb.me/weave) for LLM observability, evaluation, and leaderboards.

## Usage

### Workflow Overview

1. **First Time Setup**: Run `upload_dataset_to_wandb.py` to prepare datasets (required before training)
2. **Training**: Run `train_ruler.py` or `train_independent_reward.py` to train models
3. **Benchmarking**: Run `create_leaderboard.py` at any time to:
   - Before training: Benchmark the untrained base model against OpenAI models
   - After training: Compare trained models against the base model and OpenAI models

### Step 1: Prepare Dataset (Required First)

**Important**: This must be run before training scripts will work.

```bash
python upload_dataset_to_wandb.py
```

This script creates everything in a single W&B run:
- ✅ **W&B Artifacts** - Used by training scripts (required for training)
- ✅ **Weave Datasets** - Published to Weave for LLM observability
- ✅ **W&B Tables** - Visualize scenarios in the W&B UI
- ✅ **Summary Statistics** - Dataset counts and metadata

The number of scenarios loaded is controlled by `training_dataset_size` and `validation_dataset_size` in `config.yaml`.

**Note**: If you change the dataset configuration (size or seed), re-run this script to create new versions with the updated data.

### Step 2: Training

Train using the RULER algorithm:

```bash
# Use default config.yaml
python train_ruler.py

# Or specify a custom config file
python train_ruler.py --config my_custom_config.yaml
```

Or train using independent reward scoring:

```bash
python train_independent_reward.py
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

### Step 3: Model Comparison with Weave Leaderboards (Flexible Timing)

**This can be run at any time** - before or after training:

- **Before training**: Benchmark the untrained base model (OpenPipe/Qwen) against OpenAI models
- **After training**: Compare trained models against the base model and OpenAI models

The script automatically detects which models are available and includes them in the comparison.

```bash
# Use default config.yaml
python create_leaderboard.py

# Or specify a custom config file
python create_leaderboard.py --config my_custom_config.yaml
```

This will:
1. Load the validation dataset from W&B
2. Set up models for comparison (includes trained models if they exist)
3. Run all Weave scorers on each model:
   - **CorrectnessJudgeScorer** - LLM judge for answer correctness
   - **SourceRetrievalScorer** - Precision/recall for source emails
   - **ToolUsageScorer** - LLM judge for tool call appropriateness
4. Create a Weave leaderboard with all metrics
5. Log results to W&B for parallel coordinates visualization

**Models included in comparison:**
- Comparison model (configurable in `config.yaml` as `comparison_model`)
- OpenPipe/Qwen base model (untrained)
- Trained RULER model (if exists - from `train_ruler.py`)
- Trained Independent model (if exists - from `train_independent_reward.py`)

**Viewing Results:**

1. **Weave Leaderboard**: Visit `https://wandb.ai/[entity]/[project]/weave/leaderboards`
2. **Parallel Coordinates**: 
   - Go to your W&B run page
   - Click "Add Panel" → "Parallel Coordinates"
   - Select metrics to compare across models
   - Color by model name to see patterns

**Key Metrics Tracked:**
- `correct` - Answer correctness (0.0-1.0)
- `retrieved_correct_sources` - Whether all expected sources were retrieved
- `tool_optimal_rate` - % of optimal tool decisions

## Project Structure

### Core Scripts

- **`train_ruler.py`** - Main training script using RULER algorithm
  - Initializes the model and serverless backend
  - Runs the training loop with RULER scoring
  - Performs validation at regular intervals
  - Configurable training parameters (epochs, learning rate, etc.)

- **`train_independent_reward.py`** - Alternative training script using independent reward scoring
  - Uses independent reward evaluation where each trajectory is scored individually on an absolute scale
  - Unlike RULER which compares trajectories to rank them relatively, this approach gives each trajectory its own independent score
  - Useful for comparing different reward strategies and their impact on model training

- **`create_leaderboard.py`** - Creates leaderboards for model comparison
  - Compares multiple models using Weave evaluations
  - Uses Weave scorers (Correctness, Source Retrieval, Tool Usage)
  - Creates Weave leaderboards for easy comparison
  - Logs metrics for parallel coordinates visualization

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
- **`upload_dataset_to_wandb.py`** - Upload datasets to W&B Artifacts, Weave Datasets, and W&B Tables

### Configuration Files

- **`config.yaml`** - Training configuration (hyperparameters, model settings, W&B config)

### Data Files

- `training_scenarios.json/csv` - Training dataset scenarios
- `validation_scenarios.json/csv` - Validation dataset scenarios
- `enron_emails.db` - SQLite database containing Enron email dataset

## Configuration

All project configuration is stored in `config.yaml` and used across all scripts in the repository. See the file for complete configuration options. Key configuration sections include:

- **Model configuration**: Project name, base model selection
- **Dataset configuration**: Training/validation sizes, random seed, artifact references
- **Judge models**: RULER judge, correctness judge, and tool usage judge models
- **Training hyperparameters**: Groups per step, epochs, rollouts, learning rate
- **Validation and evaluation**: Step intervals for validation
- **Checkpointing**: Artifact saving preferences
- **Reproducibility**: Random seeds
- **W&B run configuration**: Run names and job types

### Dataset Configuration

Dataset sizes and random seed are centrally configured in `config.yaml`:

- **`training_dataset_size`**: Number of training scenarios to load from the Enron dataset
- **`validation_dataset_size`**: Number of validation scenarios to load
- **`dataset_seed`**: Random seed for shuffling scenarios

These settings are used by the `upload_dataset_to_wandb.py` script, ensuring that all artifact versions use the same data configuration.

### Training Hyperparameters

Training duration is determined by the formula: **Total training steps** = `(training_dataset_size / groups_per_step) * num_epochs`

Training runs until all epochs are complete (no arbitrary max_steps limit).

Key parameters (see `config.yaml` for current values):
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
python upload_dataset_to_wandb.py
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

The `upload_dataset_to_wandb.py` script creates all three formats in one run, ensuring consistency across all representations.

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
python train_ruler.py --config config_high_lr.yaml
```

## Requirements

See `requirements.txt` for dependencies.

## Related Work

1. [ART Example Notebook](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/art-e.ipynb)
