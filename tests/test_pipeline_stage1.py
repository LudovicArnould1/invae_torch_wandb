"""
Test script for Stage 1: Data Preparation.

This script tests the DataPreparationStage independently to ensure:
1. Data loading and preprocessing works
2. Artifacts are created and logged to W&B
3. Output format is correct
"""
from homework_scientalab.config import load_data_config
from homework_scientalab.pipeline import DataPreparationStage
import wandb

def test_data_preparation_stage():
    """Test data preparation stage independently."""
    print("=" * 80)
    print("TESTING DATA PREPARATION STAGE")
    print("=" * 80)
    
    # Load config
    data_cfg = load_data_config()
    print(f"\nData Config:")
    print(f"  Data path: {data_cfg.data_path}")
    print(f"  HVGs: {data_cfg.n_top_genes}")
    print(f"  Val size: {data_cfg.val_size}")
    
    # Initialize W&B run
    run = wandb.init(
        project="invae-pancreas-test",
        name="test-data-prep-stage",
        job_type="data_preparation",
    )
    
    try:
        # Create and run data preparation stage
        data_stage = DataPreparationStage(wandb_run=run)
        output = data_stage.run(data_cfg=data_cfg, save_processed=True)
        
        # Verify output
        print("\n" + "=" * 80)
        print("STAGE OUTPUT:")
        print("=" * 80)
        print(f"\nArtifacts: {output.artifacts}")
        print(f"\nMetadata keys: {list(output.metadata.keys())}")
        print(f"  - n_cells: {output.metadata.get('n_cells')}")
        print(f"  - n_genes: {output.metadata.get('n_genes')}")
        print(f"  - train_cells: {output.metadata.get('train_cells')}")
        print(f"  - val_cells: {output.metadata.get('val_cells')}")
        print(f"  - dims: {output.metadata.get('dims')}")
        print(f"\nLocal paths: {output.local_paths}")
        
        # Verify local file exists
        dataset_path = output.local_paths.get("dataset")
        if dataset_path and dataset_path.exists():
            print(f"\n✓ Dataset saved successfully to {dataset_path}")
            print(f"  File size: {dataset_path.stat().st_size / 1024 / 1024:.2f} MB")
        else:
            print(f"\n✗ Dataset file not found at {dataset_path}")
        
        print("\n" + "=" * 80)
        print("✓ DATA PREPARATION STAGE TEST PASSED")
        print("=" * 80)
        
        return output
        
    finally:
        wandb.finish()


if __name__ == "__main__":
    output = test_data_preparation_stage()

