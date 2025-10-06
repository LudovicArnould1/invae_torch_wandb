# Configuration Module

This directory contains all configuration management for the inVAE training pipeline.

## Structure

```
config/
├── __init__.py           # Exports all configs and loaders
├── configs.py            # All dataclass definitions
├── loader.py             # YAML loading utilities
├── data_config.yaml      # Data preprocessing configuration
├── model_config.yaml     # Model architecture configuration
└── train_config.yaml     # Training hyperparameters
```

## Configuration Files

### `data_config.yaml`
Controls data loading and preprocessing:
- Dataset paths and backup URLs
- Quality control filters (min genes/cells, mitochondrial filtering)
- Feature selection (number of highly variable genes)
- Train/validation split parameters

### `model_config.yaml`
Defines model architecture:
- Latent space dimensions (z_i for biological, z_s for technical variation)
- Hidden layer sizes for encoder, decoder, and prior networks
- Regularization (dropout, batch normalization)

### `train_config.yaml`
Training hyperparameters:
- Number of epochs, batch size, learning rate
- Beta warmup schedule for KL annealing
- Independence penalty weight
- Checkpointing and W&B settings

## Usage

### Option 1: Load from default YAML files

```python
from homework_scientalab.train import main

# Uses config/*.yaml files automatically
main()
```

### Option 2: Load from custom YAML files

```python
from homework_scientalab.train import main

main(
    data_config_path="my_configs/data.yaml",
    train_config_path="my_configs/train.yaml"
)
```

### Option 3: Load with programmatic overrides

```python
from homework_scientalab.config import load_data_config, load_train_config
from homework_scientalab.train import train

# Load base config from YAML, override specific values
data_cfg = load_data_config(overrides={"n_top_genes": 2000, "batch_size": 128})
train_cfg = load_train_config(overrides={"n_epochs": 100, "learning_rate": 5e-4})

# Train with custom configs
train(train_cfg, data_cfg, use_wandb=True)
```

### Option 4: Pure Python (no YAML)

```python
from homework_scientalab.config import DataConfig, TrainConfig, WarmupSchedule
from homework_scientalab.train import train

# Create configs directly in Python
data_cfg = DataConfig(
    data_path="data/pancreas.h5ad",
    n_top_genes=2000,
    val_size=0.2
)

train_cfg = TrainConfig(
    n_epochs=50,
    batch_size=256,
    z_i_dim=30,
    z_s_dim=5,
    warmup_schedule=WarmupSchedule(
        beta_start=0.4,
        beta_end=1.0,
        warmup_steps=250
    )
)

train(train_cfg, data_cfg, use_wandb=True)
```

## Benefits

1. **Centralized Configuration**: All configs in one place (`configs.py`)
2. **Type Safety**: Dataclasses provide IDE support and type checking
3. **Flexibility**: Use YAML files or Python, or mix both
4. **Version Control**: YAML files can be versioned and tracked
5. **Easy Experimentation**: Modify YAML files without touching code
6. **Documentation**: YAML files serve as human-readable documentation

## Notes

- **Tuples in YAML**: Hidden layer sizes are lists in YAML (e.g., `[128, 128]`) but automatically converted to tuples in Python
- **Nested Configs**: `WarmupSchedule` is embedded in `TrainConfig` as a nested YAML structure
- **Required Parameters**: Some parameters like `x_dim`, `b_dim`, `t_dim` in `InVAEConfig` are set automatically from data dimensions

