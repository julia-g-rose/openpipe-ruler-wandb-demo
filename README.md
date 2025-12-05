# Email Search Agent - OpenPipe ART Demo

This project demonstrates training an LLM-powered email search agent using [OpenPipe's ART (Automated Reinforcement Learning Trainer)](https://docs.openpipe.ai/features/art) framework with different reward strategies.

## Overview

The agent searches through the Enron email dataset to answer user queries by:
1. Using tools to search and retrieve relevant emails
2. Analyzing email content to find answers
3. Citing source emails to support answers

This demo showcases three different training approaches:
- **RULER**: LLM-based comparative scoring
- **Weave Scorers**: Hand-crafted independent reward metrics
- **Combined**: Both RULER and Weave Scorers together

The project integrates:
- **[OpenPipe ART](https://docs.openpipe.ai/features/art)** - RL training framework
- **[W&B Models](https://docs.wandb.ai/guides/models)** - Experiment tracking and model management
- **[W&B Weave](https://wandb.me/weave)** - LLM observability, evaluation, and leaderboards

## Quick Start

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables by creating a `.env` file in the project root:
```bash
# Create .env file
cat > .env << EOF
WANDB_API_KEY=your_wandb_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
EOF
```

Get your API keys from:
- W&B: https://wandb.ai/authorize
- OpenAI: https://platform.openai.com/api-keys

### Workflow Overview

1. **Setup**: Run `upload_dataset_to_wandb.py` to prepare datasets (required before training)
2. **Training**: Choose one of three training approaches (see below)
3. **Evaluation**: Run `create_leaderboard.py` to benchmark and compare models

### Step 1: Prepare Dataset

**Required before training:**

```bash
python upload_dataset_to_wandb.py
```

This creates:
- ✅ **W&B Artifacts** - Used by training scripts (required)
- ✅ **Weave Datasets** - Published to Weave for LLM observability
- ✅ **W&B Tables** - Visualize scenarios in the W&B UI
- ✅ **Summary Statistics** - Dataset counts and metadata

Dataset size is controlled by `training_dataset_size` and `validation_dataset_size` in `config.yaml`.

**Note**: Re-run this script if you change dataset configuration (size or seed).

### Step 2: Training

Choose one of three training approaches:

#### Option A: RULER Only (LLM-Based Comparative Scoring)

Uses RULER algorithm with LLM judge to compare trajectories:

```bash
python train_with_ruler.py

# Or with custom config
python train_with_ruler.py --config my_config.yaml

# Resume training
python train_with_ruler.py --resume-auto
```

#### Option B: Weave Scorers Only (Independent Reward Metrics)

Uses hand-crafted metrics (correctness, source retrieval, tool usage) as independent rewards:

```bash
python train_with_weave_scorers.py

# Or with custom config
python train_with_weave_scorers.py --config my_config.yaml
```

#### Option C: Combined Approach (RULER + Weave Scorers)

Combines both RULER comparative scoring and independent reward metrics:

```bash
python train_with_ruler_and_weave_scorers.py

# Or with custom config
python train_with_ruler_and_weave_scorers.py --config my_config.yaml
```

All training scripts will:
1. Load configuration from YAML
2. Download dataset artifacts from W&B
3. Initialize the email agent model
4. Run training for configured epochs
5. Save checkpoints as W&B model artifacts
6. Log metrics and validation results to W&B

**Note**: Training runs until all epochs complete. All checkpoints are preserved.

### Step 3: Evaluation and Leaderboards

Run at any time to benchmark models:

```bash
python create_leaderboard.py

# Or with custom config
python create_leaderboard.py --config my_config.yaml
```

This will:
1. Load the validation dataset
2. Evaluate all available models (base + trained)
3. Run Weave scorers:
   - **CorrectnessJudgeScorer** - LLM-based answer correctness
   - **SourceRetrievalScorer** - Source email retrieval quality
   - **ToolUsageScorer** - Tool call appropriateness
4. Create a Weave leaderboard
5. Log results to W&B

**Available Models:**
- Comparison model (e.g., GPT-4o, configured in `config.yaml`)
- OpenPipe/Qwen base model (untrained)
- Trained models (automatically detected if they exist)

**Viewing Results:**

1. **Weave Leaderboard**: `https://wandb.ai/[your-entity]/email-search-agent-art-demo/weave/leaderboards`
2. **W&B Dashboard**: View training metrics, validation results, and comparisons
3. **Parallel Coordinates**: Add panel in W&B to compare metrics across models

Replace `[your-entity]` with your W&B username or team name.

**Key Metrics:**
- `correct` - Answer correctness (0.0-1.0)
- `retrieved_correct_sources` - Source retrieval accuracy
- `tool_optimal_rate` - Tool usage quality

## Project Structure

### Core Scripts

#### Training Scripts

- **`train_with_ruler.py`** - Training with RULER algorithm
  - Uses LLM judge to compare and rank trajectories
  - Provides relative scoring based on trajectory comparison
  - Best for learning from comparative feedback

- **`train_with_weave_scorers.py`** - Training with independent rewards
  - Uses hand-crafted metrics (correctness, source retrieval, tool usage)
  - Scores each trajectory independently on absolute scale
  - Best for training with explicit reward signals

- **`train_with_ruler_and_weave_scorers.py`** - Combined approach
  - Combines RULER comparative scoring with independent metrics
  - Best of both worlds: relative LLM feedback + explicit rewards

All training scripts support:
- Custom config files (`--config`)
- Automatic resumption (`--resume-auto`)
- Resume specific runs (`--resume-id`)

#### Evaluation Scripts

- **`create_leaderboard.py`** - Model benchmarking and comparison
  - Evaluates models using Weave scorers
  - Creates Weave leaderboards
  - Logs results to W&B for visualization
  - Automatically detects available trained models

- **`upload_dataset_to_wandb.py`** - Dataset preparation
  - Creates W&B Artifacts, Weave Datasets, and W&B Tables
  - Must be run before training
  - Ensures reproducibility with versioned datasets

#### Utility Files

- **`helpers.py`** - Shared utilities and Weave scorers
  - `rollout()` - Executes agent on a scenario
  - `CorrectnessJudgeScorer` - LLM-based answer correctness
  - `SourceRetrievalScorer` - Source email retrieval metrics
  - `ToolUsageScorer` - Tool call appropriateness evaluation
  - Data models: `EmailScenario`, `ProjectTrajectory`, etc.
  - All LLM judge prompts are version-controlled with Weave

- **`enron_helpers.py`** - Enron dataset utilities
  - Email search functions
  - Database access
  - Scenario management

#### Configuration

- **`config.yaml`** - Centralized configuration
  - Model settings (base model, project name)
  - Dataset configuration (sizes, seeds, artifacts)
  - Judge models (RULER, correctness, tool usage)
  - Training hyperparameters (epochs, learning rate, etc.)
  - Validation and checkpointing settings

#### Data Files

- `enron_emails.db` - SQLite database with Enron emails
- `training_scenarios.json` - Training scenarios
- `validation_scenarios.json` - Validation scenarios
- `artifacts/` - W&B dataset artifact cache

## Configuration

All configuration is centralized in `config.yaml`. Key sections:

### Model Configuration
- `project` - W&B project name
- `base_model` - Base model to train (e.g., `OpenPipe/Qwen3-14B-Instruct`)
- `model_name_ruler` - Name for RULER-trained model
- `model_name_independent` - Name for Weave scorer-trained model
- `model_name_combined` - Name for combined approach model

### Dataset Configuration
- `training_dataset_size` - Number of training scenarios (default: 500)
- `validation_dataset_size` - Number of validation scenarios (default: 100)
- `dataset_seed` - Random seed for reproducibility (default: 42)
- `training_dataset_artifact` - W&B artifact reference for training data
- `validation_dataset_artifact` - W&B artifact reference for validation data

### Judge Models
- `ruler_judge_model` - LLM judge for RULER scoring (e.g., `openai/gpt-5`)
- `correctness_judge_model` - LLM judge for answer correctness
- `tool_judge_model` - LLM judge for tool usage evaluation
- `comparison_model` - Model for leaderboard comparison (e.g., `gpt-4o`)

### Training Hyperparameters
- `groups_per_step` - Scenario groups per training step (default: 4)
- `num_epochs` - Training epochs (default: 1)
- `rollouts_per_group` - Trajectories per scenario for RULER (default: 4)
- `learning_rate` - Learning rate (default: 1.0e-5)

**Training duration**: Total steps = `(training_dataset_size / groups_per_step) * num_epochs`

### Validation & Checkpointing
- `validation_step_interval` - Steps between validations (default: 25)
- `save_checkpoint_artifact` - Save checkpoints to W&B (default: true)
- `log_correctness_correlation_plots` - Log correlation plots (default: true)

### Using Custom Configurations

Create different config files for different experiments:

```bash
# Create a custom config
cp config.yaml config_high_lr.yaml
# Edit config_high_lr.yaml to change settings

# Run with custom config
python train_with_ruler.py --config config_high_lr.yaml
```

## Key Features

### Dataset Management

**W&B Artifacts** (Required for training)
- Versioned dataset storage with lineage tracking
- Automatic caching and downloading
- Ensures reproducibility across runs

**Weave Datasets** (Optional, recommended)
- LLM observability and advanced querying
- Integrates with Weave trace data
- Useful for debugging and analysis

**W&B Tables** (Optional)
- Interactive visualization in W&B UI
- Easy browsing and filtering
- Helpful for data exploration

The `upload_dataset_to_wandb.py` script creates all three formats in one run.

### Checkpoint Management

When `save_checkpoint_artifact: true`:
- Checkpoint metadata saved as W&B artifacts after each step
- Includes training state, hyperparameters, and dataset references
- Full history preserved (no deletion)
- Clear lineage between datasets, checkpoints, and runs
- Easy rollback to any previous checkpoint

Actual checkpoint files are managed by ART ServerlessBackend on W&B servers.

### Judge Models

The project uses multiple LLM judges:

- **`ruler_judge_model`** - RULER algorithm scoring (comparative)
- **`correctness_judge_model`** - Answer correctness evaluation (absolute)
- **`tool_judge_model`** - Tool usage appropriateness evaluation

All judge prompts are version-controlled using Weave's prompt management.

## Requirements

### Dependencies

Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `openpipe-art>=0.5.2` - OpenPipe ART framework
- `wandb` - Experiment tracking
- `weave` - LLM observability
- `pydantic` - Data validation
- `litellm` - LLM API wrapper
- Additional utilities: `langchain-core`, `tenacity`, `datasets`, `tqdm`, `pyyaml`

### Environment Variables

Create a `.env` file with:

```bash
WANDB_API_KEY=your_wandb_key
OPENAI_API_KEY=your_openai_key  # For judge models
```

## Example Workflow

Here's a complete workflow from setup to evaluation:

```bash
# 1. Setup
pip install -r requirements.txt
# Create .env with your API keys

# 2. Prepare datasets
python upload_dataset_to_wandb.py

# 3. Train with RULER
python train_with_ruler.py

# 4. Evaluate and compare
python create_leaderboard.py

# 5. View results in W&B
# Visit: https://wandb.ai/[your-entity]/email-search-agent-art-demo
```

## Dataset

This project uses the [Enron Email Dataset](https://www.cs.cmu.edu/~enron/), a public corpus of emails made available by the Federal Energy Regulatory Commission during their investigation. The dataset is included in the repository as `enron_emails.db`.

## Resources

### OpenPipe
- [ART Documentation](https://docs.openpipe.ai/features/art)
- [RULER Reward Function](https://docs.openpipe.ai/features/art/reward-functions/ruler)
- [ART Example Notebook](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/art-e.ipynb)

### W&B
- [W&B Models Documentation](https://docs.wandb.ai/guides/models)
- [W&B Weave Documentation](https://wandb.me/weave)
- [Weave Evaluations Guide](https://wandb.github.io/weave/guides/evaluation/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT
