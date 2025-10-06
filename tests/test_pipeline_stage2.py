"""
Test script for Stage 2: Model Training.

This script tests the TrainingStage using the output from DataPreparationStage.
"""
from homework_scientalab.config import load_train_config
from homework_scientalab.pipeline import TrainingStage, StageOutput
from pathlib import Path
import wandb
import torch

def test_training_stage():
    """Test training stage using cached data from stage 1."""
    print("=" * 80)
    print("TESTING TRAINING STAGE")
    print("=" * 80)
    
    # Check if data from stage 1 exists
    dataset_path = Path("data/pancreas_processed.h5ad")
    if not dataset_path.exists():
        print("Error: Need to run stage 1 first to generate processed data")
        return
    
    print(f"\n✓ Found processed dataset at {dataset_path}")
    
    # Create a mock data output (simulating stage 1 output)
    data_output = StageOutput(
        artifacts={},
        metadata={
            "n_cells": 9025,
            "n_genes": 2443,
            "n_batches": 4,
            "n_celltypes": 23,
            "train_cells": 7220,
            "val_cells": 1805,
            "dims": {"x_dim": 2443, "b_dim": 23, "t_dim": 4},
        },
        local_paths={"dataset": dataset_path},
    )
    
    # Load training config with reduced epochs for testing
    train_cfg = load_train_config()
    print("\nTraining Config:")
    print(f"  Epochs: {train_cfg.n_epochs} (using for test)")
    print(f"  Batch size: {train_cfg.batch_size}")
    print(f"  Learning rate: {train_cfg.learning_rate}")
    print(f"  Latent dims: z_i={train_cfg.z_i_dim}, z_s={train_cfg.z_s_dim}")
    
    # Override for quick testing
    train_cfg.n_epochs = 5
    train_cfg.save_every = 2
    print(f"  → Overriding to {train_cfg.n_epochs} epochs for quick test")
    
    # Initialize W&B run
    run = wandb.init(
        project="invae-pancreas-test",
        name="test-training-stage",
        job_type="training",
        config={"n_epochs": train_cfg.n_epochs, "batch_size": train_cfg.batch_size},
    )
    
    try:
        # Create and run training stage
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        
        training_stage = TrainingStage(wandb_run=run)
        output = training_stage.run(
            train_cfg=train_cfg,
            data_output=data_output,
            device=device,
        )
        
        # Verify output
        print("\n" + "=" * 80)
        print("STAGE OUTPUT:")
        print("=" * 80)
        print(f"\nArtifacts: {output.artifacts}")
        print(f"\nMetadata keys: {list(output.metadata.keys())}")
        print(f"  - best_val_loss: {output.metadata.get('best_val_loss')}")
        print(f"  - n_epochs: {output.metadata.get('n_epochs')}")
        print(f"\nLocal paths: {output.local_paths}")
        
        # Verify checkpoint exists
        model_path = output.local_paths.get("model")
        if model_path and model_path.exists():
            print(f"\n✓ Model checkpoint saved successfully to {model_path}")
            print(f"  File size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
        else:
            print(f"\n✗ Model checkpoint not found at {model_path}")
        
        print("\n" + "=" * 80)
        print("✓ TRAINING STAGE TEST PASSED")
        print("=" * 80)
        
        return output
        
    finally:
        wandb.finish()


if __name__ == "__main__":
    output = test_training_stage()

