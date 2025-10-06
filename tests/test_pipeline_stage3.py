"""
Test script for Stage 3: Model Evaluation.

This script tests the EvaluationStage using outputs from previous stages.
"""
from homework_scientalab.pipeline import EvaluationStage, StageOutput
from pathlib import Path
import wandb
import torch

def test_evaluation_stage():
    """Test evaluation stage using cached data and model from previous stages."""
    print("=" * 80)
    print("TESTING EVALUATION STAGE")
    print("=" * 80)
    
    # Check if required files exist
    dataset_path = Path("data/pancreas_processed.h5ad")
    model_path = Path("checkpoints/best_model.pt")
    
    if not dataset_path.exists():
        print(f"Error: Need dataset at {dataset_path}")
        print("Run test_pipeline_stage1.py first")
        return
    
    if not model_path.exists():
        print(f"Error: Need trained model at {model_path}")
        print("Run test_pipeline_stage2.py first")
        return
    
    print(f"✓ Found dataset at {dataset_path}")
    print(f"✓ Found model at {model_path}")
    
    # Create mock outputs from previous stages
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
    
    training_output = StageOutput(
        artifacts={},
        metadata={
            "best_val_loss": 1626.43,
            "n_epochs": 5,
        },
        local_paths={"model": model_path},
    )
    
    # Initialize W&B run
    run = wandb.init(
        project="invae-pancreas-test",
        name="test-evaluation-stage",
        job_type="evaluation",
    )
    
    try:
        # Create and run evaluation stage
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        
        eval_stage = EvaluationStage(wandb_run=run)
        output = eval_stage.run(
            data_output=data_output,
            training_output=training_output,
            device=device,
        )
        
        # Verify output
        print("\n" + "=" * 80)
        print("STAGE OUTPUT:")
        print("=" * 80)
        print(f"\nMetadata keys: {list(output.metadata.keys())}")
        print(f"  - val_loss: {output.metadata.get('val_loss')}")
        print(f"  - n_val_cells: {output.metadata.get('n_val_cells')}")
        print(f"  - model_epoch: {output.metadata.get('model_epoch')}")
        
        print("\n" + "=" * 80)
        print("✓ EVALUATION STAGE TEST PASSED")
        print("=" * 80)
        
        return output
        
    finally:
        wandb.finish()


if __name__ == "__main__":
    output = test_evaluation_stage()

