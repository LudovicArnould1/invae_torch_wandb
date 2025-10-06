"""
Visualize inVAE latent representations to demonstrate batch correction.

Core visualizations:
1. Raw data: Shows batch effects
2. Invariant space (z_i): Batch-corrected, biology preserved
3. Spurious space (z_s): Captures batch effects only
4. Train/val comparison: Verify generalization
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import scanpy as sc
import matplotlib.pyplot as plt
from anndata import AnnData
from sklearn.metrics import silhouette_score

from homework_scientalab.model import InVAE
from homework_scientalab.config import InVAEConfig


def get_latent_representations(
    model: InVAE,
    adata: AnnData,
    batch_key: str = "batch",
    celltype_key: str = "celltype",
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract invariant (z_i) and spurious (z_s) latent representations.
    
    Args:
        model: Trained InVAE model
        adata: AnnData with raw counts in .layers['counts']
        batch_key: Column name for batch labels
        celltype_key: Column name for cell type labels
        batch_size: Batch size for inference
        
    Returns:
        z_i: Invariant latent representation (n_cells, z_i_dim)
        z_s: Spurious latent representation (n_cells, z_s_dim)
    """
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import DataLoader, TensorDataset
    
    model.eval()
    device = next(model.parameters()).device
    
    # Prepare data
    counts = adata.layers["counts"]
    if hasattr(counts, "toarray"):
        counts = counts.toarray()
    counts = torch.FloatTensor(counts)
    
    # One-hot encode covariates
    batch_enc = LabelEncoder()
    batch_labels = batch_enc.fit_transform(adata.obs[batch_key])
    batch_onehot = torch.FloatTensor(np.eye(len(batch_enc.classes_))[batch_labels])
    
    celltype_enc = LabelEncoder()
    celltype_labels = celltype_enc.fit_transform(adata.obs[celltype_key])
    celltype_onehot = torch.FloatTensor(np.eye(len(celltype_enc.classes_))[celltype_labels])
    
    # Compute size factors
    library_size = counts.sum(dim=1, keepdim=True)
    size_factors = torch.clamp(library_size, min=1.0)
    
    # Create dataloader
    dataset = TensorDataset(counts, celltype_onehot, batch_onehot, size_factors.squeeze())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Extract latents
    z_i_list, z_s_list = [], []
    
    with torch.no_grad():
        for x_batch, b_batch, t_batch, sf_batch in loader:
            x_batch = x_batch.to(device)
            b_batch = b_batch.to(device)
            t_batch = t_batch.to(device)
            sf_batch = sf_batch.to(device)
            
            out = model(x_batch, b_batch, t_batch, sf_batch, sample_z=False)
            z_i_list.append(out["z_i"].cpu().numpy())
            z_s_list.append(out["z_s"].cpu().numpy())
    
    z_i = np.vstack(z_i_list)
    z_s = np.vstack(z_s_list)
    
    return z_i, z_s


def compute_batch_correction_metrics(
    adata: AnnData,
    batch_key: str = "batch",
    celltype_key: str = "celltype",
    representation: str = "X_latent",
) -> Dict[str, float]:
    """
    Compute quantitative metrics for batch correction quality.
    
    Metrics:
    - Batch ASW (Average Silhouette Width): Lower is better (less batch separation)
    - Cell type ASW: Higher is better (more biological separation)
    - Batch mixing entropy: Higher is better (more batch mixing)
    
    Args:
        adata: AnnData with representation in .obsm[representation]
        batch_key: Column name for batch labels
        celltype_key: Column name for cell type labels
        representation: Key in adata.obsm for the representation to evaluate
        
    Returns:
        Dictionary with metric names and values
    """
    from sklearn.preprocessing import LabelEncoder
    from scipy.stats import entropy
    from sklearn.neighbors import NearestNeighbors
    
    # Get representation
    if representation in adata.obsm:
        X = adata.obsm[representation]
    elif representation == "X":
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
    else:
        raise ValueError(f"Representation {representation} not found in adata")
    
    # Encode labels
    batch_enc = LabelEncoder()
    batch_labels = batch_enc.fit_transform(adata.obs[batch_key])
    
    celltype_enc = LabelEncoder()
    celltype_labels = celltype_enc.fit_transform(adata.obs[celltype_key])
    
    metrics = {}
    
    # 1. Silhouette scores
    try:
        # Batch ASW (lower is better - we want batches mixed)
        batch_asw = silhouette_score(X, batch_labels, sample_size=min(5000, len(X)))
        metrics["batch_asw"] = float(batch_asw)
        
        # Cell type ASW (higher is better - we want cell types separated)
        celltype_asw = silhouette_score(X, celltype_labels, sample_size=min(5000, len(X)))
        metrics["celltype_asw"] = float(celltype_asw)
    except Exception as e:
        print(f"Warning: Could not compute silhouette scores: {e}")
        metrics["batch_asw"] = np.nan
        metrics["celltype_asw"] = np.nan
    
    # 2. Batch mixing entropy (k-NN based)
    try:
        k = min(50, len(X) // 10)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
        _, indices = nbrs.kneighbors(X)
        
        # For each cell, compute entropy of batch distribution in k-NN
        batch_entropies = []
        n_batches = len(batch_enc.classes_)
        
        for idx in indices:
            # Get batch distribution in neighborhood (excluding self)
            neighbor_batches = batch_labels[idx[1:]]
            batch_counts = np.bincount(neighbor_batches, minlength=n_batches)
            batch_probs = batch_counts / batch_counts.sum()
            batch_entropies.append(entropy(batch_probs))
        
        # Average entropy (higher is better - more batch mixing)
        metrics["batch_mixing_entropy"] = float(np.mean(batch_entropies))
        metrics["batch_mixing_entropy_std"] = float(np.std(batch_entropies))
    except Exception as e:
        print(f"Warning: Could not compute batch mixing entropy: {e}")
        metrics["batch_mixing_entropy"] = np.nan
        metrics["batch_mixing_entropy_std"] = np.nan
    
    return metrics


def visualize_train_val_split(
    adata_raw: AnnData,
    adata_inv: AnnData,
    output_dir: Path,
    split_key: str = "split",
    batch_key: str = "batch",
    celltype_key: str = "celltype",
) -> Dict[str, Dict[str, float]]:
    """
    Create separate visualizations for train and val sets.
    
    Generates individual plots:
    - train_raw_batch.png, train_raw_celltype.png
    - train_invariant_batch.png, train_invariant_celltype.png
    - val_raw_batch.png, val_raw_celltype.png
    - val_invariant_batch.png, val_invariant_celltype.png
    
    Args:
        adata_raw: Raw data with UMAP
        adata_inv: Invariant representation with UMAP
        output_dir: Directory to save plots
        split_key: Column for train/val split
        batch_key: Column for batch labels
        celltype_key: Column for cell type labels
        
    Returns:
        Dictionary with metrics for each split and representation
    """
    metrics_all = {}
    
    for split_name in ["train", "val"]:
        print(f"\nProcessing {split_name.upper()} set...")
        
        # Subset data
        mask = adata_raw.obs[split_key] == split_name
        adata_raw_sub = adata_raw[mask].copy()
        adata_inv_sub = adata_inv[mask].copy()
        
        n_cells = adata_raw_sub.n_obs
        print(f"  {n_cells} cells")
        
        # Raw data plots
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        sc.pl.umap(adata_raw_sub, color=batch_key, ax=ax, show=False,
                   title=f"{split_name.upper()} - Raw Data (Batch)", 
                   palette="Set2", frameon=False)
        plt.tight_layout()
        save_path = output_dir / f"{split_name}_raw_batch.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        sc.pl.umap(adata_raw_sub, color=celltype_key, ax=ax, show=False,
                   title=f"{split_name.upper()} - Raw Data (Cell Type)", 
                   legend_loc=None, frameon=False)
        plt.tight_layout()
        save_path = output_dir / f"{split_name}_raw_celltype.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()
        
        # Invariant plots
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        sc.pl.umap(adata_inv_sub, color=batch_key, ax=ax, show=False,
                   title=f"{split_name.upper()} - Invariant z_i (Batch Mixing)", 
                   palette="Set2", frameon=False)
        plt.tight_layout()
        save_path = output_dir / f"{split_name}_invariant_batch.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        sc.pl.umap(adata_inv_sub, color=celltype_key, ax=ax, show=False,
                   title=f"{split_name.upper()} - Invariant z_i (Cell Types)", 
                   legend_loc=None, frameon=False)
        plt.tight_layout()
        save_path = output_dir / f"{split_name}_invariant_celltype.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()
        
        # Compute metrics
        print(f"  Computing metrics...")
        raw_metrics = compute_batch_correction_metrics(
            adata_raw_sub, batch_key, celltype_key, representation="X_umap"
        )
        inv_metrics = compute_batch_correction_metrics(
            adata_inv_sub, batch_key, celltype_key, representation="X_latent"
        )
        
        metrics_all[f"{split_name}_raw"] = raw_metrics
        metrics_all[f"{split_name}_invariant"] = inv_metrics
    
    return metrics_all


def save_metrics_report(
    metrics: Dict[str, Dict[str, float]],
    output_path: Path,
    additional_info: Optional[Dict] = None,
) -> None:
    """
    Save metrics to a markdown file.
    
    Args:
        metrics: Dictionary of metrics for each representation
        output_path: Path to save the markdown file
        additional_info: Optional additional information to include
    """
    with open(output_path, "w") as f:
        f.write("# Batch Correction Metrics Report\n\n")
        
        if additional_info:
            f.write("## Model Information\n\n")
            for key, value in additional_info.items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")
        
        f.write("## Metrics Explanation\n\n")
        f.write("**Batch ASW (Average Silhouette Width)**:\n")
        f.write("- Measures batch separation in the latent space\n")
        f.write("- Range: [-1, 1]\n")
        f.write("- **Lower is better** (closer to 0 means less batch effect)\n")
        f.write("- Negative values indicate good batch mixing\n\n")
        
        f.write("**Cell Type ASW**:\n")
        f.write("- Measures biological (cell type) separation\n")
        f.write("- Range: [-1, 1]\n")
        f.write("- **Higher is better** (closer to 1 means well-separated cell types)\n\n")
        
        f.write("**Batch Mixing Entropy**:\n")
        f.write("- Measures diversity of batches in k-NN neighborhoods\n")
        f.write("- Range: [0, log(n_batches)]\n")
        f.write("- **Higher is better** (more batch mixing)\n\n")
        
        f.write("---\n\n")
        f.write("## Results\n\n")
        
        # Organize by split (train/val)
        for split in ["train", "val"]:
            f.write(f"### {split.upper()} Set\n\n")
            
            raw_key = f"{split}_raw"
            inv_key = f"{split}_invariant"
            
            if raw_key in metrics and inv_key in metrics:
                raw_metrics = metrics[raw_key]
                inv_metrics = metrics[inv_key]
                
                f.write("| Metric | Raw Data | Invariant (z_i) | Change | Interpretation |\n")
                f.write("|--------|----------|-----------------|--------|----------------|\n")
                
                # Batch ASW
                raw_basw = raw_metrics.get("batch_asw", np.nan)
                inv_basw = inv_metrics.get("batch_asw", np.nan)
                change_basw = inv_basw - raw_basw
                interp_basw = "✅ Improved" if change_basw < -0.05 else "⚠️ Marginal" if change_basw < 0 else "❌ Worse"
                f.write(f"| Batch ASW (↓ better) | {raw_basw:.4f} | {inv_basw:.4f} | {change_basw:+.4f} | {interp_basw} |\n")
                
                # Cell type ASW
                raw_casw = raw_metrics.get("celltype_asw", np.nan)
                inv_casw = inv_metrics.get("celltype_asw", np.nan)
                change_casw = inv_casw - raw_casw
                interp_casw = "✅ Preserved" if abs(change_casw) < 0.1 else "⚠️ Changed" if change_casw > -0.2 else "❌ Lost"
                f.write(f"| Cell Type ASW (preserve) | {raw_casw:.4f} | {inv_casw:.4f} | {change_casw:+.4f} | {interp_casw} |\n")
                
                # Batch mixing entropy
                raw_ent = raw_metrics.get("batch_mixing_entropy", np.nan)
                inv_ent = inv_metrics.get("batch_mixing_entropy", np.nan)
                change_ent = inv_ent - raw_ent
                interp_ent = "✅ Improved" if change_ent > 0.05 else "⚠️ Marginal" if change_ent > 0 else "❌ Worse"
                f.write(f"| Batch Mixing Entropy (↑ better) | {raw_ent:.4f} | {inv_ent:.4f} | {change_ent:+.4f} | {interp_ent} |\n")
                
                f.write("\n")
        
        f.write("---\n\n")
        f.write("## Summary\n\n")
        f.write("**Good batch correction should show**:\n")
        f.write("- ✅ Batch ASW decreases (closer to 0 or negative)\n")
        f.write("- ✅ Cell Type ASW remains high or increases\n")
        f.write("- ✅ Batch Mixing Entropy increases\n")
        f.write("- ✅ Similar metrics for train and val (good generalization)\n\n")
        
    print(f"✓ Saved metrics report: {output_path}")


def visualize_batch_correction(
    adata_raw: AnnData,
    model_path: str,
    config: InVAEConfig,
    output_dir: str = "validation_results/latent_vis",
    batch_key: str = "batch",
    celltype_key: str = "celltype",
    log_to_wandb: bool = False,
    ) -> None:
    """
    Create comprehensive visualization of batch correction.
    
    Args:
        adata_raw: Raw preprocessed AnnData
        model_path: Path to trained model checkpoint
        config: Model configuration
        output_dir: Directory to save plots
        batch_key: Batch covariate key
        celltype_key: Cell type covariate key
        log_to_wandb: If True, log plots and metrics as W&B artifacts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("LATENT SPACE VISUALIZATION")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = InVAE(config)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("✓ Model loaded")
    
    # Get latent representations
    print("\nExtracting latent representations...")
    z_i, z_s = get_latent_representations(model, adata_raw, batch_key, celltype_key)
    print(f"  z_i (invariant): {z_i.shape}")
    print(f"  z_s (spurious): {z_s.shape}")
    
    # Create copies for different representations
    adata_inv = adata_raw.copy()
    adata_spur = adata_raw.copy()
    
    adata_inv.obsm["X_latent"] = z_i
    adata_spur.obsm["X_latent"] = z_s
    
    # Compute UMAP embeddings
    print("\nComputing UMAP embeddings...")
    
    # Raw data
    if "X_umap" not in adata_raw.obsm:
        sc.pp.neighbors(adata_raw, n_neighbors=15)
        sc.tl.umap(adata_raw)
    
    # Invariant space
    sc.pp.neighbors(adata_inv, use_rep="X_latent", n_neighbors=15)
    sc.tl.umap(adata_inv)
    
    # Spurious space
    sc.pp.neighbors(adata_spur, use_rep="X_latent", n_neighbors=15)
    sc.tl.umap(adata_spur)
    
    print("✓ UMAP embeddings computed")
    
    # Create visualizations
    print("\nGenerating plots...")
    
    # 1. Main comparison: Raw vs Invariant
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Colored by batch
    sc.pl.umap(adata_raw, color=batch_key, ax=axes[0, 0], show=False,
               title="Raw Data (Batch)", palette="Set2", frameon=False)
    sc.pl.umap(adata_inv, color=batch_key, ax=axes[0, 1], show=False,
               title="Invariant z_i (Batch)", palette="Set2", frameon=False)
    sc.pl.umap(adata_spur, color=batch_key, ax=axes[0, 2], show=False,
               title="Spurious z_s (Batch)", palette="Set2", frameon=False)
    
    # Row 2: Colored by cell type
    sc.pl.umap(adata_raw, color=celltype_key, ax=axes[1, 0], show=False,
               title="Raw Data (Cell Type)", legend_loc=None, frameon=False)
    sc.pl.umap(adata_inv, color=celltype_key, ax=axes[1, 1], show=False,
               title="Invariant z_i (Cell Type)", legend_loc=None, frameon=False)
    sc.pl.umap(adata_spur, color=celltype_key, ax=axes[1, 2], show=False,
               title="Spurious z_s (Cell Type)", legend_loc=None, frameon=False)
    
    plt.tight_layout()
    save_path = output_path / "latent_comparison.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    # 2. Side-by-side: Raw vs Invariant (cleaner version)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    sc.pl.umap(adata_raw, color=batch_key, ax=axes[0, 0], show=False,
               title="Raw: Batch Effect Visible", palette="Set2", frameon=False)
    sc.pl.umap(adata_raw, color=celltype_key, ax=axes[1, 0], show=False,
               title="Raw: Cell Types", legend_loc=None, frameon=False)
    
    sc.pl.umap(adata_inv, color=batch_key, ax=axes[0, 1], show=False,
               title="Corrected: Batch Mixing", palette="Set2", frameon=False)
    sc.pl.umap(adata_inv, color=celltype_key, ax=axes[1, 1], show=False,
               title="Corrected: Cell Types Preserved", legend_loc=None, frameon=False)
    
    plt.tight_layout()
    save_path = output_path / "batch_correction_comparison.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    # Save latent representations
    print("\nSaving latent representations...")
    adata_inv.write(output_path / "adata_invariant.h5ad")
    adata_spur.write(output_path / "adata_spurious.h5ad")
    print(f"✓ Saved invariant and spurious AnnData objects")
    
    # Train/val split visualization (if split column exists)
    if "split" in adata_raw.obs.columns:
        print("\n" + "=" * 80)
        print("TRAIN/VAL SPLIT ANALYSIS")
        print("=" * 80)
        
        metrics_all = visualize_train_val_split(
            adata_raw, adata_inv, output_path,
            split_key="split", batch_key=batch_key, celltype_key=celltype_key
        )
        
        # Save metrics report
        model_info = {
            "z_i_dim": config.z_i_dim,
            "z_s_dim": config.z_s_dim,
            "n_cells": adata_raw.n_obs,
            "n_genes": adata_raw.n_vars,
            "n_batches": len(adata_raw.obs[batch_key].cat.categories),
            "n_celltypes": len(adata_raw.obs[celltype_key].cat.categories),
        }
        save_metrics_report(metrics_all, output_path / "metrics_report.md", model_info)
    else:
        print("\nNote: No 'split' column found in data. Skipping train/val analysis.")
    
    # Log visualizations to W&B
    if log_to_wandb:
        from homework_scientalab.artifacts import log_visualization_artifact, log_metrics_table
        import wandb
        
        print("\n" + "=" * 80)
        print("LOGGING TO WANDB")
        print("=" * 80)
        
        # Collect all plot files
        plot_files = [
            output_path / "latent_comparison.png",
            output_path / "batch_correction_comparison.png",
        ]
        
        # Add train/val plots if they exist
        if "split" in adata_raw.obs.columns:
            for split in ["train", "val"]:
                for plot_type in ["raw_batch", "raw_celltype", "invariant_batch", "invariant_celltype"]:
                    plot_file = output_path / f"{split}_{plot_type}.png"
                    if plot_file.exists():
                        plot_files.append(plot_file)
        
        # Log visualization artifact
        existing_plots = [str(p) for p in plot_files if p.exists()]
        if existing_plots:
            log_visualization_artifact(
                existing_plots,
                artifact_name="batch_correction_plots",
                description="UMAP visualizations showing batch correction quality",
                metadata={
                    "model_path": model_path,
                    "n_plots": len(existing_plots),
                },
            )
            print(f"✓ Logged {len(existing_plots)} plots to W&B")
        
        # Log metrics as W&B table
        if "split" in adata_raw.obs.columns:
            log_metrics_table(metrics_all, table_name="batch_correction_metrics")
            print(f"✓ Logged metrics table to W&B")
        
        # Log images directly to W&B for visualization in UI
        for plot_file in existing_plots:
            plot_name = Path(plot_file).stem
            try:
                wandb.log({f"visualization/{plot_name}": wandb.Image(plot_file)})
            except Exception:
                pass
        
        print("=" * 80)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_path}")
    print("\nKey observations:")
    print("  1. Raw data: Clear batch clusters")
    print("  2. Invariant (z_i): Batch mixing + cell type preservation")
    print("  3. Spurious (z_s): Captures batch effects only")
    if "split" in adata_raw.obs.columns:
        print("  4. Train/val analysis: Check metrics_report.md for generalization")


def quick_visualize(
    raw_data_path: str = "data/pancreas_processed.h5ad",
    model_path: str = "checkpoints/best_model.pt",
    config_path: Optional[str] = None,
) -> None:
    """
    Quick visualization using default paths.
    
    Args:
        raw_data_path: Path to preprocessed data
        model_path: Path to trained model
        config_path: Optional path to model config (will infer from checkpoint if None)
    """
    from homework_scientalab.config import load_model_config
    
    # Load data
    print(f"Loading data from {raw_data_path}...")
    adata = sc.read(raw_data_path)
    
    # Load or infer config
    if config_path is not None:
        config = load_model_config(config_path)
    else:
        # Infer from checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        if "config" in checkpoint:
            config = checkpoint["config"]
        else:
            # Use default config with inferred dimensions
            from homework_scientalab.config import InVAEConfig
            config = InVAEConfig(
                x_dim=adata.n_vars,
                b_dim=len(adata.obs["celltype"].cat.categories),
                t_dim=len(adata.obs["batch"].cat.categories),
            )
    
    # Run visualization
    visualize_batch_correction(adata, model_path, config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize inVAE latent representations")
    parser.add_argument("--data", default="data/pancreas_processed.h5ad",
                        help="Path to preprocessed data")
    parser.add_argument("--model", default="checkpoints/best_model.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--output", default="validation_results/latent_vis",
                        help="Output directory for plots")
    parser.add_argument("--no-rapids", action="store_true",
                        help="Disable rapids-singlecell (use standard scanpy)")
    
    args = parser.parse_args()
    
    quick_visualize(
        raw_data_path=args.data,
        model_path=args.model,
    )

