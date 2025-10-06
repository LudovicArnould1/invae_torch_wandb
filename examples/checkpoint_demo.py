"""
Demo script showing checkpoint and resume functionality.

This demonstrates:
1. Training with automatic checkpointing
2. Resuming from checkpoint after interruption
3. Checkpoint cleanup (keeping best + last N)
4. W&B artifact upload
"""
from pathlib import Path
from homework_scientalab.pipeline import run_pipeline

# Example 1: Train from scratch with checkpointing
print("=" * 80)
print("Example 1: Train from scratch with automatic checkpointing")
print("=" * 80)

outputs = run_pipeline(
    data_config_path="src/homework_scientalab/config/data_config.yaml",
    train_config_path="src/homework_scientalab/config/train_config.yaml",
    use_wandb=True,
    resume_from_checkpoint=False,  # Start fresh
)

print("\nTraining complete!")
print(f"Best validation loss: {outputs['training'].metadata['best_val_loss']:.4f}")

# Checkpoints are saved in:
# - checkpoints/best_model.pt (best checkpoint)
# - checkpoints/checkpoint_epoch_X.pt (last N checkpoints)
# - checkpoints/resume_info.json (metadata for resumption)

# Example 2: Resume from checkpoint
print("\n" + "=" * 80)
print("Example 2: Resume from checkpoint")
print("=" * 80)

# This will:
# - Load the last checkpoint
# - Restore model, optimizer, epoch, global_step, RNG states
# - Continue the same W&B run
# - Continue training from where it left off

outputs_resumed = run_pipeline(
    data_config_path="src/homework_scientalab/config/data_config.yaml",
    train_config_path="src/homework_scientalab/config/train_config.yaml",
    use_wandb=True,
    resume_from_checkpoint=True,  # Resume from checkpoint
)

print("\nResumed training complete!")

# Example 3: Programmatic usage with custom configs
print("\n" + "=" * 80)
print("Example 3: Programmatic usage")
print("=" * 80)

from homework_scientalab.config import load_data_config, load_train_config
from homework_scientalab.pipeline import Pipeline

# Load configs with overrides
data_cfg = load_data_config(overrides={
    "n_top_genes": 2000,
})

train_cfg = load_train_config(overrides={
    "n_epochs": 20,
    "batch_size": 128,
    "save_every": 10,  # Save checkpoint every 10 epochs
    "save_dir": "checkpoints/custom_run",
})

# Create pipeline
pipeline = Pipeline(
    project="invae-checkpoint-demo",
    run_name="custom_run",
)

# Run with checkpointing
outputs_custom = pipeline.run(
    data_cfg=data_cfg,
    train_cfg=train_cfg,
    resume_from_checkpoint=False,
)

print("\nCustom run complete!")

# To resume this custom run later:
# outputs_resumed = pipeline.run(
#     data_cfg=data_cfg,
#     train_cfg=train_cfg,
#     resume_from_checkpoint=True,
# )

print("\n" + "=" * 80)
print("Demo complete!")
print("=" * 80)
print("\nKey features:")
print("✓ Automatic checkpoint saving every N epochs")
print("✓ Save best + last 2 checkpoints (disk space management)")
print("✓ Complete state preservation (model, optimizer, RNG)")
print("✓ W&B run resumption with same run ID")
print("✓ W&B artifact upload for best checkpoints")
print("✓ Exact continuation after crashes")

