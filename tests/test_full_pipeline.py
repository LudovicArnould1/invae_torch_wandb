"""
Test script for the complete pipeline.

This script tests the full Pipeline orchestrator that chains all stages together.
"""
from homework_scientalab.config import load_data_config, load_train_config
from homework_scientalab.pipeline import Pipeline

def test_full_pipeline():
    """Test the complete pipeline with all stages."""
    print("=" * 80)
    print("TESTING FULL PIPELINE")
    print("=" * 80)
    
    # Load configs
    data_cfg = load_data_config()
    train_cfg = load_train_config()
    
    # Override for quick testing
    train_cfg.n_epochs = 3
    train_cfg.save_every = 1
    
    print("\nPipeline Configuration:")
    print(f"  Data: {data_cfg.n_top_genes} HVGs, val_size={data_cfg.val_size}")
    print(f"  Training: {train_cfg.n_epochs} epochs, batch_size={train_cfg.batch_size}")
    print(f"  Latent dims: z_i={train_cfg.z_i_dim}, z_s={train_cfg.z_s_dim}")
    
    # Create and run pipeline
    pipeline = Pipeline(
        project="invae-pancreas-test",
        entity=None,
        run_name="test-full-pipeline",
    )
    
    print("\n" + "=" * 80)
    print("RUNNING FULL PIPELINE")
    print("=" * 80)
    
    outputs = pipeline.run(
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        run_data_prep=True,
        run_training=True,
        run_evaluation=True,
    )
    
    # Verify outputs
    print("\n" + "=" * 80)
    print("PIPELINE OUTPUTS")
    print("=" * 80)
    
    print(f"\nStages completed: {list(outputs.keys())}")
    
    if "data_preparation" in outputs:
        data_out = outputs["data_preparation"]
        print("\n✓ Data Preparation:")
        print(f"  - Cells: {data_out.metadata.get('n_cells')}")
        print(f"  - Genes: {data_out.metadata.get('n_genes')}")
        print(f"  - Train/Val: {data_out.metadata.get('train_cells')}/{data_out.metadata.get('val_cells')}")
        print(f"  - Local path: {data_out.local_paths.get('dataset')}")
    
    if "training" in outputs:
        train_out = outputs["training"]
        print("\n✓ Training:")
        print(f"  - Best val loss: {train_out.metadata.get('best_val_loss'):.2f}")
        print(f"  - Epochs: {train_out.metadata.get('n_epochs')}")
        print(f"  - Local path: {train_out.local_paths.get('model')}")
    
    if "evaluation" in outputs:
        eval_out = outputs["evaluation"]
        print("\n✓ Evaluation:")
        print(f"  - Val loss: {eval_out.metadata.get('val_loss'):.2f}")
        print(f"  - Val cells: {eval_out.metadata.get('n_val_cells')}")
    
    print("\n" + "=" * 80)
    print("✓ FULL PIPELINE TEST PASSED")
    print("=" * 80)
    
    return outputs


if __name__ == "__main__":
    outputs = test_full_pipeline()

