#!/usr/bin/env python
"""
Quick-start script to run the inVAE training pipeline.

Usage:
    python run_pipeline.py                    # Run full pipeline with defaults
    python run_pipeline.py --epochs 100       # Override epochs
    python run_pipeline.py --offline          # Run in offline mode
    python run_pipeline.py --help             # Show all options
"""
import argparse
import os
from homework_scientalab.config import load_data_config, load_train_config
from homework_scientalab.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run the inVAE training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Data arguments
    parser.add_argument(
        "--data-config",
        type=str,
        default=None,
        help="Path to data config YAML (default: config/data_config.yaml)",
    )
    parser.add_argument(
        "--hvgs",
        type=int,
        default=None,
        help="Number of highly variable genes to select",
    )
    
    # Training arguments
    parser.add_argument(
        "--train-config",
        type=str,
        default=None,
        help="Path to train config YAML (default: config/train_config.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--z-i-dim",
        type=int,
        default=None,
        help="Biological latent dimensions",
    )
    parser.add_argument(
        "--z-s-dim",
        type=int,
        default=None,
        help="Technical latent dimensions",
    )
    
    # Pipeline control
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip data preparation (reuse cached data)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation",
    )
    
    # W&B arguments
    parser.add_argument(
        "--project",
        type=str,
        default="invae-pancreas",
        help="W&B project name",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity (username/team)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="W&B run name",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run W&B in offline mode",
    )
    
    args = parser.parse_args()
    
    # Set offline mode if requested
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        print("Running in OFFLINE mode - logs will be saved locally")
    
    # Load configurations
    print("Loading configurations...")
    
    data_overrides = {}
    if args.hvgs is not None:
        data_overrides["n_top_genes"] = args.hvgs
    data_cfg = load_data_config(yaml_path=args.data_config, overrides=data_overrides)
    
    train_overrides = {}
    if args.epochs is not None:
        train_overrides["n_epochs"] = args.epochs
    if args.batch_size is not None:
        train_overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        train_overrides["learning_rate"] = args.lr
    if args.z_i_dim is not None:
        train_overrides["z_i_dim"] = args.z_i_dim
    if args.z_s_dim is not None:
        train_overrides["z_s_dim"] = args.z_s_dim
    train_cfg = load_train_config(yaml_path=args.train_config, overrides=train_overrides)
    
    # Print configuration summary
    print("\n" + "=" * 80)
    print("PIPELINE CONFIGURATION")
    print("=" * 80)
    print(f"\nData:")
    print(f"  Path: {data_cfg.data_path}")
    print(f"  HVGs: {data_cfg.n_top_genes}")
    print(f"  Val size: {data_cfg.val_size}")
    print(f"\nTraining:")
    print(f"  Epochs: {train_cfg.n_epochs}")
    print(f"  Batch size: {train_cfg.batch_size}")
    print(f"  Learning rate: {train_cfg.learning_rate}")
    print(f"  Latent dims: z_i={train_cfg.z_i_dim}, z_s={train_cfg.z_s_dim}")
    print(f"\nW&B:")
    print(f"  Project: {args.project}")
    print(f"  Entity: {args.entity or '(default)'}")
    print(f"  Run name: {args.run_name or '(auto-generated)'}")
    print(f"\nStages:")
    print(f"  Data prep: {'SKIP' if args.skip_data_prep else 'RUN'}")
    print(f"  Training: {'SKIP' if args.skip_training else 'RUN'}")
    print(f"  Evaluation: {'SKIP' if args.skip_evaluation else 'RUN'}")
    print("=" * 80)
    
    # Handle skip data prep
    data_output = None
    if args.skip_data_prep:
        from pathlib import Path
        from homework_scientalab.pipeline import StageOutput
        
        cached_path = Path("data/pancreas_processed.h5ad")
        if not cached_path.exists():
            print(f"\nError: Cannot skip data prep - no cached data at {cached_path}")
            print("Run without --skip-data-prep first to create the cached data.")
            return
        
        print(f"\nUsing cached data from {cached_path}")
        data_output = StageOutput(
            artifacts={},
            metadata={
                "dims": {"x_dim": 2443, "b_dim": 23, "t_dim": 4},
            },
            local_paths={"dataset": cached_path},
        )
    
    # Create and run pipeline
    print("\n" + "=" * 80)
    print("STARTING PIPELINE")
    print("=" * 80 + "\n")
    
    pipeline = Pipeline(
        project=args.project,
        entity=args.entity,
        run_name=args.run_name,
    )
    
    outputs = pipeline.run(
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        run_data_prep=not args.skip_data_prep,
        run_training=not args.skip_training,
        run_evaluation=not args.skip_evaluation,
        data_output=data_output,
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    
    if "data_preparation" in outputs:
        data_out = outputs["data_preparation"]
        print(f"\n✓ Data Preparation:")
        print(f"  Cells: {data_out.metadata.get('n_cells')}")
        print(f"  Genes: {data_out.metadata.get('n_genes')}")
        print(f"  Train/Val: {data_out.metadata.get('train_cells')}/{data_out.metadata.get('val_cells')}")
        print(f"  Saved to: {data_out.local_paths.get('dataset')}")
    
    if "training" in outputs:
        train_out = outputs["training"]
        print(f"\n✓ Training:")
        print(f"  Best val loss: {train_out.metadata.get('best_val_loss'):.2f}")
        print(f"  Epochs: {train_out.metadata.get('n_epochs')}")
        print(f"  Model saved to: {train_out.local_paths.get('model')}")
    
    if "evaluation" in outputs:
        eval_out = outputs["evaluation"]
        print(f"\n✓ Evaluation:")
        print(f"  Val loss: {eval_out.metadata.get('val_loss'):.2f}")
        print(f"  Val cells: {eval_out.metadata.get('n_val_cells')}")
    
    print("\n" + "=" * 80)
    
    if args.offline:
        print("\nTo sync offline run to W&B cloud:")
        print("  wandb sync wandb/offline-run-<run_id>")


if __name__ == "__main__":
    main()

