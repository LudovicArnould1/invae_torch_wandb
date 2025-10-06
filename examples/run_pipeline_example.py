"""
Example: How to use the modular pipeline for inVAE training.

This script demonstrates different ways to use the pipeline:
1. Running the full pipeline (all stages)
2. Running individual stages
3. Skipping stages and reusing cached artifacts
"""
import os
from homework_scientalab.config import load_data_config, load_train_config
from homework_scientalab.pipeline import (
    Pipeline,
    DataPreparationStage,
    TrainingStage,
    EvaluationStage,
    run_pipeline,
)
import wandb

# Set to offline mode if you want to test without W&B cloud sync
os.environ["WANDB_MODE"] = "offline"  # Remove this line for online mode


def example_1_full_pipeline():
    """
    Example 1: Run the complete pipeline (recommended for most users).
    
    This is the simplest way - just load configs and run everything.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: FULL PIPELINE")
    print("=" * 80)
    
    # Load configs from YAML files
    data_cfg = load_data_config()
    train_cfg = load_train_config()
    
    # Optional: Override specific parameters
    train_cfg.n_epochs = 20  # Reduce epochs for testing
    
    # Create and run pipeline
    pipeline = Pipeline(
        project="invae-pancreas",
        entity=None,  # Set to your W&B username/team
        run_name="full-pipeline-run",
    )
    
    outputs = pipeline.run(
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        run_data_prep=True,
        run_training=True,
        run_evaluation=True,
    )
    
    print("\n✓ Pipeline complete!")
    print(f"  - Dataset: {outputs['data_preparation'].local_paths['dataset']}")
    print(f"  - Model: {outputs['training'].local_paths['model']}")
    print(f"  - Best val loss: {outputs['training'].metadata['best_val_loss']:.2f}")
    
    return outputs


def example_2_convenience_function():
    """
    Example 2: Use the convenience function (even simpler).
    
    This is the one-liner approach.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: CONVENIENCE FUNCTION")
    print("=" * 80)
    
    # This does everything in one call
    outputs = run_pipeline(
        data_config_path=None,  # Uses default config
        train_config_path=None,  # Uses default config
        use_wandb=True,
    )
    
    print("\n✓ Pipeline complete!")
    return outputs


def example_3_individual_stages():
    """
    Example 3: Run stages individually for more control.
    
    This is useful when you want to:
    - Inspect outputs between stages
    - Debug specific stages
    - Rerun only certain stages
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: INDIVIDUAL STAGES")
    print("=" * 80)
    
    # Load configs
    data_cfg = load_data_config()
    train_cfg = load_train_config()
    train_cfg.n_epochs = 5
    
    # Initialize W&B run (shared across all stages)
    run = wandb.init(
        project="invae-pancreas",
        name="individual-stages-run",
        config={
            "n_epochs": train_cfg.n_epochs,
            "batch_size": train_cfg.batch_size,
        },
    )
    
    try:
        # Stage 1: Data Preparation
        print("\n>>> Running Stage 1: Data Preparation")
        data_stage = DataPreparationStage(wandb_run=run)
        data_output = data_stage.run(data_cfg=data_cfg)
        print(f"✓ Data ready: {data_output.metadata['n_cells']} cells, "
              f"{data_output.metadata['n_genes']} genes")
        
        # Stage 2: Training
        print("\n>>> Running Stage 2: Training")
        training_stage = TrainingStage(wandb_run=run)
        training_output = training_stage.run(
            train_cfg=train_cfg,
            data_output=data_output,
        )
        print(f"✓ Training complete: best val loss = {training_output.metadata['best_val_loss']:.2f}")
        
        # Stage 3: Evaluation
        print("\n>>> Running Stage 3: Evaluation")
        eval_stage = EvaluationStage(wandb_run=run)
        eval_output = eval_stage.run(
            data_output=data_output,
            training_output=training_output,
        )
        print(f"✓ Evaluation complete: {eval_output.metadata['n_val_cells']} cells evaluated")
        
        print("\n✓ All stages complete!")
        
    finally:
        wandb.finish()


def example_4_skip_data_prep():
    """
    Example 4: Skip data preparation and reuse cached data.
    
    Useful when you want to train multiple models on the same data.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: SKIP DATA PREP (REUSE CACHED DATA)")
    print("=" * 80)
    
    from pathlib import Path
    from homework_scientalab.pipeline import StageOutput
    
    # Check if we have cached data
    cached_data_path = Path("data/pancreas_processed.h5ad")
    if not cached_data_path.exists():
        print("Error: No cached data found. Run example 1 or 3 first.")
        return
    
    print(f"✓ Using cached data from {cached_data_path}")
    
    # Create mock data output
    data_output = StageOutput(
        artifacts={},
        metadata={
            "n_cells": 9025,
            "n_genes": 2443,
            "dims": {"x_dim": 2443, "b_dim": 23, "t_dim": 4},
        },
        local_paths={"dataset": cached_data_path},
    )
    
    # Load training config
    train_cfg = load_train_config()
    train_cfg.n_epochs = 5
    train_cfg.z_i_dim = 40  # Try different latent dimensions
    
    # Run pipeline without data prep
    pipeline = Pipeline(
        project="invae-pancreas",
        run_name="skip-data-prep-run",
    )
    
    data_cfg = load_data_config()  # Still needed for config logging
    
    outputs = pipeline.run(
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        run_data_prep=False,  # Skip data prep!
        run_training=True,
        run_evaluation=True,
        data_output=data_output,  # Use pre-existing data
    )
    
    print("\n✓ Training and evaluation complete without reprocessing data!")
    print(f"  - Best val loss: {outputs['training'].metadata['best_val_loss']:.2f}")


def example_5_train_multiple_models():
    """
    Example 5: Train multiple models on the same data.
    
    This demonstrates the power of stage separation - you can parallelize
    training multiple models without reprocessing the data each time.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: TRAIN MULTIPLE MODELS")
    print("=" * 80)
    
    from pathlib import Path
    from homework_scientalab.pipeline import StageOutput
    
    # Use cached data
    cached_data_path = Path("data/pancreas_processed.h5ad")
    if not cached_data_path.exists():
        print("Error: No cached data found. Run example 1 first.")
        return
    
    data_output = StageOutput(
        artifacts={},
        metadata={"dims": {"x_dim": 2443, "b_dim": 23, "t_dim": 4}},
        local_paths={"dataset": cached_data_path},
    )
    
    # Try different model configurations
    configs = [
        {"z_i_dim": 20, "z_s_dim": 3, "name": "small"},
        {"z_i_dim": 30, "z_s_dim": 5, "name": "medium"},
        {"z_i_dim": 40, "z_s_dim": 7, "name": "large"},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n>>> Training {config['name']} model (z_i={config['z_i_dim']}, z_s={config['z_s_dim']})")
        
        # Load and modify config
        train_cfg = load_train_config()
        train_cfg.n_epochs = 5
        train_cfg.z_i_dim = config["z_i_dim"]
        train_cfg.z_s_dim = config["z_s_dim"]
        train_cfg.save_dir = f"checkpoints/{config['name']}"
        
        # Run training only
        run = wandb.init(
            project="invae-pancreas",
            name=f"model-{config['name']}",
            config=config,
            reinit=True,
        )
        
        try:
            training_stage = TrainingStage(wandb_run=run)
            output = training_stage.run(
                train_cfg=train_cfg,
                data_output=data_output,
            )
            
            results.append({
                "name": config["name"],
                "val_loss": output.metadata["best_val_loss"],
                "config": config,
            })
            
            print(f"✓ {config['name']} model: val_loss = {output.metadata['best_val_loss']:.2f}")
            
        finally:
            wandb.finish()
    
    # Compare results
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON:")
    print("=" * 80)
    for result in results:
        print(f"  {result['name']:10s}: val_loss = {result['val_loss']:.2f} "
              f"(z_i={result['config']['z_i_dim']}, z_s={result['config']['z_s_dim']})")
    
    best = min(results, key=lambda x: x["val_loss"])
    print(f"\n✓ Best model: {best['name']} with val_loss = {best['val_loss']:.2f}")


if __name__ == "__main__":
    import sys
    
    # Run specific example or all of them
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = {
            1: example_1_full_pipeline,
            2: example_2_convenience_function,
            3: example_3_individual_stages,
            4: example_4_skip_data_prep,
            5: example_5_train_multiple_models,
        }
        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Unknown example {example_num}. Choose 1-5.")
    else:
        print("\n" + "=" * 80)
        print("inVAE PIPELINE EXAMPLES")
        print("=" * 80)
        print("\nAvailable examples:")
        print("  1. Full pipeline (recommended)")
        print("  2. Convenience function (simplest)")
        print("  3. Individual stages (most control)")
        print("  4. Skip data prep (reuse cached data)")
        print("  5. Train multiple models (hyperparameter search)")
        print("\nUsage: python run_pipeline_example.py <example_number>")
        print("\nRunning Example 1 (full pipeline)...")
        example_1_full_pipeline()

