#!/usr/bin/env python3
"""
Visualize train/val split performance of inVAE model.

Generates:
- Separate plots for train and val sets
- Metrics report comparing raw vs corrected representations
- Quantitative assessment of batch correction quality

Usage:
    python scripts/visualize_train_val.py
    python scripts/visualize_train_val.py --data data/pancreas_processed.h5ad --model checkpoints/best_model.pt
"""
import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from homework_scientalab.validation.visualize_latent import quick_visualize


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize train/val split performance of inVAE model"
    )
    parser.add_argument(
        "--data",
        default="data/pancreas_processed.h5ad",
        help="Path to preprocessed data with split info (default: data/pancreas_processed.h5ad)",
    )
    parser.add_argument(
        "--model",
        default="checkpoints/best_model.pt",
        help="Path to trained model checkpoint (default: checkpoints/best_model.pt)",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TRAIN/VAL SPLIT VISUALIZATION")
    print("=" * 80)
    print(f"\nData: {args.data}")
    print(f"Model: {args.model}")
    print()
    
    # Check if split column exists
    import scanpy as sc
    adata_check = sc.read(args.data)
    if "split" not in adata_check.obs.columns:
        print("⚠️  WARNING: No 'split' column found in data!")
        print("The preprocessed data needs to be regenerated with train/val split info.")
        print()
        print("To fix this, run:")
        print("  python -m homework_scientalab.train  # This will regenerate pancreas_processed.h5ad")
        print()
        sys.exit(1)
    
    print(f"✓ Found split info: {(adata_check.obs['split'] == 'train').sum()} train, "
          f"{(adata_check.obs['split'] == 'val').sum()} val cells\n")
    
    quick_visualize(
        raw_data_path=args.data,
        model_path=args.model,
    )
    
    print("\n" + "=" * 80)
    print("OUTPUTS GENERATED")
    print("=" * 80)
    print("\nPlots:")
    print("  - train_raw_batch.png, train_raw_celltype.png")
    print("  - train_invariant_batch.png, train_invariant_celltype.png")
    print("  - val_raw_batch.png, val_raw_celltype.png")
    print("  - val_invariant_batch.png, val_invariant_celltype.png")
    print("\nMetrics:")
    print("  - metrics_report.md (quantitative assessment)")
    print("\nAll saved in: validation_results/latent_vis/")

