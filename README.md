# inVAE Implementation

Implementation of the **inVAE** (interpretable Variational Autoencoder) model for single-cell RNA-seq data analysis. This project provides a complete pipeline for training, evaluation, and hyperparameter optimization with W&B integration.

 I believe that I have successfully implemented a minimal but working pipeline, as shown in `example_notebook.ipynb`. The separation/mixing effects a not stunningly good, but they exist and could be improved by training longer (only did up to 500 epochs also due to time and hardware constraints) and looking for better hyperparameters (did not have time to run a proper search).

## DISCLAIMER

This repository is for illustration purposes. Many implementations could be more robust and sophisticated in production (e.g., proper test set handling, more comprehensive evaluations). See the "What Should Be Improved" section for details.

**Note:** The code has not been extensively tested across different setups. Bugs may occur with different OS, dependencies, or configurations.

## Features

### Core Components
- **Data preprocessing** and exploration (`data.py`, `data/visualize_raw_pancreas.py`, `describe_h5ad.py`)
- **inVAE model** implementation (`model.py`)
- **Training pipeline** with modular stages (`train.py`, `trainer.py`, `losses.py`, `pipeline.py`)
- **Evaluation pipeline** with visualization tools (`validation/` directory)

### Supporting Components
- **Configuration management** for data, training, sweeps, and model (`config/` directory)
- **Reproducibility utilities** for seeds, environment logging (`monitor_and_setup/reproducibility.py`)
- **Extensive W&B integration** for artifact tracking and checkpointing (`monitor_and_setup/`)
- **Checkpoint management** for saving/loading models and resuming training
- **Hyperparameter optimization** with W&B sweeps (`sweep/run_sweep.py`)
- **Unit tests** for key components (`tests/` directory)
- **Muon optimizer** implementation (felt like a good opportunity to try it)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/LudovicArnould1/invae_homework/ 
   cd scientalab
   ```

2. **Create a virtual environment** (recommended) (I used uv for this project)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

   For development with linting tools:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Prepare the data**
   
   Place your pancreas H5AD dataset in the `data/` directory:
   ```
   data/pancreas.h5ad
   ```

5. **Configure W&B** (optional, should run in offline mode)
   ```bash
   wandb login
   ```

   Or set offline mode:
   ```bash
   export WANDB_MODE=offline
   ```

## Usage

You can run the training in two ways: using the **simple script** (`train.py`) or the **modular pipeline** (`pipeline.py`).

### Option 1: Simple Training Script (Quick Start)

For quick experiments, use the standalone training script:

```python
from homework_scientalab.train import main

# Run complete training with default configs
main()
```

Or from the command line:
```bash
python -m homework_scientalab.train
```

This runs data loading, preprocessing, training, and evaluation in one go.

### Option 2: Modular Pipeline (Recommended)

For more control, use the modular pipeline that separates stages:

**Quick start:**
```python
from homework_scientalab.pipeline import run_pipeline

# Run full pipeline (data prep → training → evaluation)
outputs = run_pipeline()
```

**With configuration:**
```python
from homework_scientalab.config import load_data_config, load_train_config
from homework_scientalab.pipeline import Pipeline

# Load and customize configs
data_cfg = load_data_config()
train_cfg = load_train_config()
train_cfg.n_epochs = 100
train_cfg.z_i_dim = 40

# Run pipeline
pipeline = Pipeline(project="invae-pancreas", run_name="my-experiment")
outputs = pipeline.run(data_cfg=data_cfg, train_cfg=train_cfg)
```

The configs are also available in the `config/` directory as yaml files.

**Benefits of the modular pipeline:**
- Preprocess data once, train multiple models
- Run stages independently (data prep, training, evaluation)
- Reuse cached artifacts
- Better artifact tracking with W&B
- Enable parallel training

See [`examples/PIPELINE_GUIDE.md`](examples/PIPELINE_GUIDE.md) for detailed pipeline documentation and advanced usage patterns.

### Configuration

Edit configuration files in the `config/` directory:
- `data_config.yaml`: Data preprocessing settings
- `train_config.yaml`: Training hyperparameters and model architecture

Or override programmatically:
```python
data_cfg = load_data_config(overrides={"n_top_genes": 2000})
train_cfg = load_train_config(overrides={"learning_rate": 5e-4})
```

### Hyperparameter Sweeps

Run hyperparameter optimization with W&B sweeps:

```bash
python -m homework_scientalab.sweep.run_sweep
```


## What Should Be Improved

This is an illustrative implementation. Some features are simplified or omitted when they would duplicate similar functionality already demonstrated elsewhere in the codebase.

**Priority improvements for version 0.2.0:**

1. **Evaluation Process**
   - Currently minimal with basic visualizations only
   - Add proper test set (similar to validation set)
   - Implement additional metrics and visualizations from the paper
   - Add comprehensive benchmarking of different runs and models

2. **Data Preprocessing**
   - Improve data cleaning and normalization strategies
   - Fix gene count filters (current thresholds don't filter anything in practice)
   - Address stratified train/val split issues with rare cell types
   - Handle classes with too few samples more robustly

3. **Infrastructure & Scalability**
   - Add parallel sweep execution (multi-job)
   - Support multi-GPU training (data/model parallelism). I could not really work on this part as I did not have access to a GPU cluster for the project. I am familiar with torch DDP and deepspeed and could implement it if needed and if given access to a GPU cluster, for both inference and training acceleration.
   - Other GPU or multi-cpu acceleration (PCA, data preprocessing in parallel and streaming, precomputed HVGs, etc.). Similarly, due to my limited hardware ressources, I did not dive into this part.

4. **Setup & Developer Experience**
   - Add installation scripts for various environments
   - Improve cross-platform compatibility
   - Handling of offline W&B mode
   - Add helper scripts for common workflows
   - Automation

5. **Reporting and high level analysis/comparisons**

Each improvement should be divided into issues, linked to PR and milestones. The most crucial ones should be linked to version 0.2.0 if the current one is 0.1.0.
