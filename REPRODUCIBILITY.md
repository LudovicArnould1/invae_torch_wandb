# Reproducibility Guide

This document explains how reproducibility is implemented in the inVAE training pipeline.

## Overview

The pipeline now includes comprehensive reproducibility features:

1. **Deterministic behavior** via seed setting
2. **Environment tracking** (git, system, packages)
3. **Complete hyperparameter logging** to W&B
4. **Provenance tracking** (git commit, data versions)

## Configuration

### Setting Seeds

Seeds are configured in `train_config.yaml` or `TrainConfig`:

```yaml
# Reproducibility
seed: 42
deterministic: true  # use deterministic algorithms for full reproducibility
```

- `seed`: Random seed for Python, NumPy, and PyTorch
- `deterministic`: If `true`, uses deterministic algorithms (may be slower but fully reproducible)

### Deterministic Mode Trade-offs

**Enabled (`deterministic: true`)**:
- ✅ Fully reproducible results
- ✅ Same seed = same results across runs
- ❌ May be slower (10-30% overhead)
- ❌ Some operations may not have deterministic implementations

**Disabled (`deterministic: false`)**:
- ✅ Faster training
- ✅ Better GPU utilization
- ❌ Results may vary slightly between runs (even with same seed)

## What Gets Tracked

### 1. Seeds & Randomness
- Python `random` seed
- NumPy `np.random` seed
- PyTorch CPU seed
- PyTorch CUDA seed (all devices)
- CuDNN deterministic mode
- CuDNN benchmark mode

### 2. Git Provenance
Automatically logged to W&B:
- `git/commit`: Current git commit hash
- `git/branch`: Current branch name
- `git/is_dirty`: Whether there are uncommitted changes
- `git/remote_url`: Remote repository URL
- Tags: `uncommitted-changes` if repo is dirty

### 3. System Environment
Logged to W&B under `system/*`:
- Python version
- Platform (OS, kernel)
- Processor type
- Hostname
- PyTorch version
- CUDA availability, version, CuDNN version
- GPU count, names, memory

### 4. Package Versions
Logged to W&B under `packages/*`:
- scanpy
- scikit-learn
- anndata
- wandb
- (automatically extracts from installed packages)

### 5. Hyperparameters
**All** training and data configuration logged to W&B:
- Data preprocessing params (under `data_*`)
- Model architecture (under `model/*`)
- Training hyperparameters (under `train_*`)
- Optimizer configuration (under `optimizer/*`)
- Loss weights, warmup schedules, etc.

### 6. Model Metadata
- Total trainable parameters
- Parameters per optimizer (Muon vs AdamW)
- Architecture details (layer sizes, dropout, etc.)

## Usage

### Default Behavior

Just run training as usual - reproducibility features are automatic:

```python
from homework_scientalab.train import main

# Uses default seed=42, deterministic=True from config
main()
```

### Custom Seeds

Override seed via config:

```python
from homework_scientalab.config import load_train_config
from homework_scientalab.train import train

# Load config with custom seed
train_cfg = load_train_config(overrides={"seed": 123})
data_cfg = load_data_config()

train(train_cfg, data_cfg, use_wandb=True)
```

Or modify YAML:

```yaml
# train_config.yaml
seed: 123
deterministic: true
```

### Programmatic Seed Setting

For non-training scripts:

```python
from homework_scientalab import set_seed

# Set seed for data preprocessing, analysis, etc.
set_seed(42, deterministic=True)
```

### Environment Inspection

Get environment info without W&B:

```python
from homework_scientalab import get_environment_info

env_info = get_environment_info()
print(env_info["git"]["commit"])
print(env_info["system"]["gpu_names"])
print(env_info["packages"]["pytorch"])
```

## W&B Run Configuration

All configuration is logged to W&B and visible in:
- **Overview tab**: See all hyperparameters
- **Config tab**: Searchable, filterable configs
- **System tab**: Hardware metrics

### Example W&B Config Structure

```
config/
├── data_path: "data/pancreas.h5ad"
├── data_n_top_genes: 4000
├── data_val_size: 0.2
├── data_random_state: 42
├── train_seed: 42
├── train_deterministic: true
├── train_n_epochs: 500
├── train_learning_rate: 0.001
├── model/x_dim: 4000
├── model/z_i_dim: 30
├── model/z_s_dim: 5
├── model/n_params: 1234567
├── optimizer/type: "Muon+AdamW"
├── optimizer/muon_params: 1200000
├── optimizer/adamw_params: 34567
├── git/commit: "a1b2c3d4..."
├── git/branch: "main"
├── git/is_dirty: false
├── system/python_version: "3.11.5"
├── system/pytorch_version: "2.8.0"
├── system/cuda_version: "12.1"
├── system/gpu_names: ["NVIDIA A100"]
└── packages/scanpy: "1.10.0"
```

## Reproducing a Run

To exactly reproduce a W&B run:

1. **Check git commit**: Look at `config/git/commit`
2. **Checkout that commit**: `git checkout <commit>`
3. **Verify environment**: Compare `config/system/*` and `config/packages/*`
4. **Use same config**: Download run config from W&B or use same YAML
5. **Run training**: Use same seed and deterministic setting

```bash
# Checkout exact code version
git checkout <commit-hash>

# Install exact environment (if using uv)
uv sync

# Run with same config
python -m homework_scientalab.train
```

## Best Practices

### For Experiments
- ✅ Use `deterministic: true` for paper results
- ✅ Commit code before training runs
- ✅ Use descriptive W&B run names
- ✅ Add W&B tags for experiment type

### For Development
- ✅ Use `deterministic: false` for faster iteration
- ✅ It's OK to have uncommitted changes (will be tagged)
- ✅ Use different seeds to verify robustness

### For Production
- ✅ Always use `deterministic: true`
- ✅ Lock package versions in `pyproject.toml`
- ✅ Document any manual preprocessing steps
- ✅ Save W&B run ID with model checkpoints

## Verification

Test reproducibility:

```python
from homework_scientalab.train import main

# Run 1
main()  # Note the final loss

# Run 2 (same seed)
main()  # Should get identical loss (bit-for-bit)
```

If results differ:
1. Check if `deterministic: true`
2. Verify same git commit
3. Compare W&B configs
4. Check for non-deterministic data loading (e.g., `shuffle=True` with different workers)

## Known Limitations

1. **Multi-GPU training**: Some ops may not be fully deterministic
2. **CPU vs GPU**: Results may differ slightly between CPU and GPU (even with same seed)
3. **CUDA versions**: Different CUDA versions may give different results
4. **Hardware differences**: GPU architecture may affect numerics

For maximum reproducibility:
- Use same hardware (GPU model)
- Use same CUDA version
- Use same PyTorch version
- Use deterministic mode

