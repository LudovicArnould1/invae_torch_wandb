# Pipeline Architecture Guide

## Overview

The inVAE training pipeline has been refactored into **modular, independent stages** that can be run separately or chained together. This design enables:

- **Independent execution**: Run data prep once, train multiple models
- **Artifact caching**: Reuse preprocessed data without reprocessing
- **Parallelization**: Train multiple models on the same dataset simultaneously
- **Resume from failures**: Restart from any stage
- **Better tracking**: Each stage logs artifacts to W&B for versioning

## Architecture

### Pipeline Stages

```
┌─────────────────────┐
│  1. Data Prep       │  → Dataset artifact (versioned)
│  - Load raw data    │
│  - QC filtering     │
│  - HVG selection    │
│  - Train/val split  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. Training        │  → Model artifact (versioned)
│  - Model init       │
│  - Training loop    │
│  - Validation       │
│  - Checkpointing    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. Evaluation      │  → Metrics & visualizations
│  - Load model       │
│  - Compute metrics  │
│  - Generate plots   │
└─────────────────────┘
```

### Key Design Principles

1. **Single W&B Run**: All stages share one W&B run for lineage tracking
2. **Local Caching**: Artifacts cached locally for fast access (with W&B as backup)
3. **StageOutput**: Standardized output format with artifacts, metadata, and local paths
4. **Artifact References**: Stages pass artifact references (not raw data) between them

## Usage

### Method 1: Full Pipeline (Recommended)

The simplest approach - run everything in one go:

```python
from homework_scientalab.pipeline import run_pipeline

# One-liner to run the complete pipeline
outputs = run_pipeline()
```

Or with more control:

```python
from homework_scientalab.config import load_data_config, load_train_config
from homework_scientalab.pipeline import Pipeline

data_cfg = load_data_config()
train_cfg = load_train_config()

# Optional: override parameters
train_cfg.n_epochs = 50
train_cfg.z_i_dim = 40

pipeline = Pipeline(
    project="invae-pancreas",
    entity="your-username",  # Optional
    run_name="my-experiment",
)

outputs = pipeline.run(
    data_cfg=data_cfg,
    train_cfg=train_cfg,
)

# Access results
print(f"Best val loss: {outputs['training'].metadata['best_val_loss']}")
print(f"Model path: {outputs['training'].local_paths['model']}")
```

### Method 2: Individual Stages

Run stages separately for maximum control:

```python
from homework_scientalab.pipeline import (
    DataPreparationStage,
    TrainingStage,
    EvaluationStage,
)
import wandb

# Initialize shared W&B run
run = wandb.init(project="invae-pancreas", name="my-run")

# Stage 1: Data Preparation
data_stage = DataPreparationStage(wandb_run=run)
data_output = data_stage.run(data_cfg=data_cfg)

# Stage 2: Training
training_stage = TrainingStage(wandb_run=run)
training_output = training_stage.run(
    train_cfg=train_cfg,
    data_output=data_output,
)

# Stage 3: Evaluation
eval_stage = EvaluationStage(wandb_run=run)
eval_output = eval_stage.run(
    data_output=data_output,
    training_output=training_output,
)

wandb.finish()
```

### Method 3: Skip Stages (Reuse Cached Artifacts)

Run training multiple times without reprocessing data:

```python
from homework_scientalab.pipeline import Pipeline, StageOutput
from pathlib import Path

# Create a mock data output pointing to cached data
data_output = StageOutput(
    artifacts={},
    metadata={
        "dims": {"x_dim": 2443, "b_dim": 23, "t_dim": 4},
    },
    local_paths={"dataset": Path("data/pancreas_processed.h5ad")},
)

# Run training only
pipeline = Pipeline(project="invae-pancreas")

outputs = pipeline.run(
    data_cfg=data_cfg,
    train_cfg=train_cfg,
    run_data_prep=False,  # Skip data prep!
    run_training=True,
    run_evaluation=True,
    data_output=data_output,  # Use cached data
)
```

## Use Cases

### 1. Hyperparameter Search

Train multiple models on the same preprocessed data:

```python
# Preprocess data once
data_stage = DataPreparationStage(wandb_run=run)
data_output = data_stage.run(data_cfg=data_cfg)

# Try different latent dimensions
for z_i_dim in [20, 30, 40, 50]:
    train_cfg.z_i_dim = z_i_dim
    train_cfg.save_dir = f"checkpoints/zi_{z_i_dim}"
    
    training_stage = TrainingStage(wandb_run=run)
    output = training_stage.run(train_cfg=train_cfg, data_output=data_output)
    
    print(f"z_i={z_i_dim}: val_loss={output.metadata['best_val_loss']:.2f}")
```

### 2. Resume Training

Load a checkpoint and continue training:

```python
# Load existing checkpoint
checkpoint_path = Path("checkpoints/best_model.pt")
model_output = StageOutput(
    artifacts={},
    metadata={},
    local_paths={"model": checkpoint_path},
)

# Evaluate existing model
eval_stage = EvaluationStage(wandb_run=run)
eval_output = eval_stage.run(
    data_output=data_output,
    training_output=model_output,
)
```

### 3. Parallel Training

Train multiple models simultaneously (e.g., using multiple GPUs):

```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python train_model_1.py

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python train_model_2.py

# Terminal 3 (GPU 2)
CUDA_VISIBLE_DEVICES=2 python train_model_3.py
```

All scripts reuse the same cached `data/pancreas_processed.h5ad`.

### 4. Evaluation Only

Evaluate multiple models without retraining:

```python
model_paths = [
    "checkpoints/model_v1/best_model.pt",
    "checkpoints/model_v2/best_model.pt",
    "checkpoints/model_v3/best_model.pt",
]

for path in model_paths:
    model_output = StageOutput(
        artifacts={},
        metadata={},
        local_paths={"model": Path(path)},
    )
    
    eval_stage = EvaluationStage(wandb_run=run)
    eval_output = eval_stage.run(
        data_output=data_output,
        training_output=model_output,
    )
```

## Configuration

### Data Configuration (`data_config.yaml`)

```yaml
data_path: data/pancreas.h5ad
n_top_genes: 4000
val_size: 0.2
min_genes: 200
min_cells: 10
batch_key: batch
celltype_key: celltype
```

### Training Configuration (`train_config.yaml`)

```yaml
# Model architecture
z_i_dim: 30  # Biological latent dimensions
z_s_dim: 5   # Technical latent dimensions
enc_hidden: [128, 128]
dec_hidden: [128, 128]

# Training
n_epochs: 50
batch_size: 256
learning_rate: 0.001

# W&B
project: invae-pancreas
run_name: null  # Auto-generated if not provided
```

### Override Configurations

```python
# Load with overrides
data_cfg = load_data_config(overrides={
    "n_top_genes": 2000,
    "val_size": 0.15,
})

train_cfg = load_train_config(overrides={
    "n_epochs": 100,
    "z_i_dim": 40,
    "learning_rate": 5e-4,
})
```

## W&B Integration

### Offline Mode

For testing or when working without internet:

```python
import os
os.environ["WANDB_MODE"] = "offline"

# Run pipeline normally
outputs = run_pipeline()

# Later, sync to cloud:
# wandb sync wandb/offline-run-XXXXXX
```

### Artifact Tracking

The pipeline automatically logs:
- **Dataset artifacts**: Versioned preprocessed data with metadata
- **Model artifacts**: Checkpoints with model configs and metrics
- **Metrics**: Training curves, validation metrics, evaluation results

```python
# Access artifacts in W&B UI
# Project → Artifacts → pancreas_processed:v0
# Project → Artifacts → invae_model:v0
```

## API Reference

### `StageOutput`

Standardized output from each stage:

```python
@dataclass
class StageOutput:
    artifacts: Dict[str, str]  # artifact_name → W&B reference
    metadata: Dict[str, Any]   # Stage-specific metadata
    local_paths: Dict[str, Path]  # artifact_name → local file path
```

### `PipelineStage`

Base class for all stages:

```python
class PipelineStage(ABC):
    def __init__(self, wandb_run: Optional[wandb.Run] = None)
    
    @abstractmethod
    def run(self, **kwargs) -> StageOutput
```

### `Pipeline`

Orchestrator for running multiple stages:

```python
class Pipeline:
    def __init__(
        self,
        project: str = "invae-pancreas",
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
    )
    
    def run(
        self,
        data_cfg: DataConfig,
        train_cfg: TrainConfig,
        run_data_prep: bool = True,
        run_training: bool = True,
        run_evaluation: bool = True,
        data_output: Optional[StageOutput] = None,
    ) -> Dict[str, StageOutput]
```

## Comparison: Old vs New

### Old Approach (Monolithic)

```python
from homework_scientalab.train import main

# Everything happens in one train() call
# - Data loading
# - Preprocessing
# - Training
# - Evaluation
main()

# To train another model → reprocess data from scratch 😞
```

### New Approach (Modular)

```python
from homework_scientalab.pipeline import Pipeline

# 1. Preprocess once
pipeline = Pipeline()
outputs = pipeline.run(run_training=False, run_evaluation=False)

# 2. Train multiple models (reuses preprocessed data) 🎉
for config in model_configs:
    outputs = pipeline.run(
        run_data_prep=False,
        data_output=outputs['data_preparation'],
        train_cfg=config,
    )
```

## Benefits

✅ **Faster iteration**: Reuse preprocessed data  
✅ **Parallelization**: Train multiple models simultaneously  
✅ **Better organization**: Each stage has clear responsibility  
✅ **Artifact versioning**: Track data and model versions in W&B  
✅ **Resume from failures**: Restart from any stage  
✅ **Testing**: Test stages independently  

## Examples

See `examples/run_pipeline_example.py` for comprehensive examples covering:
1. Full pipeline
2. Individual stages
3. Skipping stages
4. Training multiple models
5. Hyperparameter search

Run examples:
```bash
python examples/run_pipeline_example.py 1  # Full pipeline
python examples/run_pipeline_example.py 3  # Individual stages
python examples/run_pipeline_example.py 5  # Multiple models
```

## Legacy `train.py` Support

The old `train.py` is still available for quick experiments:

```python
from homework_scientalab.train import main

# Old approach still works
main()
```

However, **we recommend using the new pipeline** for all new work.

