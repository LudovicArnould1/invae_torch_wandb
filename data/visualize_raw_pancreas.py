"""
Visualize raw pancreas dataset to understand structure and quality.

Quick exploration script to inspect:
- Cell type distribution
- Batch effects
- Sample distribution
- QC metrics (genes/counts per cell)
- UMAP embeddings
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import scanpy as sc

# Configure scanpy plotting
sc.set_figure_params(dpi=150, figsize=(6, 4), fontsize=10, frameon=False)
sc.settings.verbosity = 1


def visualize_raw_data(data_path: str = "data/pancreas.h5ad") -> None:
    """
    Load and visualize raw pancreas dataset.
    
    Args:
        data_path: Path to raw pancreas.h5ad file
    """
    print("=" * 80)
    print("PANCREAS DATASET VISUALIZATION")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    adata = sc.read(data_path)
    
    print(f"Shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
    print("\nAvailable metadata:")
    print(f"  obs columns: {list(adata.obs.columns)}")
    print(f"  var columns: {list(adata.var.columns)}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\nCell types ({len(adata.obs['celltype'].cat.categories)}):")
    print(adata.obs['celltype'].value_counts())
    
    print(f"\nBatches ({len(adata.obs['batch'].cat.categories)}):")
    print(adata.obs['batch'].value_counts())
    
    print(f"\nSamples ({len(adata.obs['sample'].cat.categories)}):")
    print(adata.obs['sample'].value_counts())
    
    print("\nQC metrics:")
    print(f"  Mean genes per cell: {adata.obs['n_genes'].mean():.1f}")
    print(f"  Median genes per cell: {adata.obs['n_genes'].median():.1f}")
    print(f"  Mean counts per cell: {adata.obs['n_counts'].mean():.1f}")
    print(f"  Median counts per cell: {adata.obs['n_counts'].median():.1f}")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    
    # 1. QC metrics histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(adata.obs['n_genes'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Number of genes')
    axes[0].set_ylabel('Number of cells')
    axes[0].set_title('Genes per cell distribution')
    axes[0].axvline(200, color='red', linestyle='--', label='min_genes=200')
    axes[0].legend()
    
    axes[1].hist(adata.obs['n_counts'], bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Total counts')
    axes[1].set_ylabel('Number of cells')
    axes[1].set_title('Counts per cell distribution')
    axes[1].axvline(100, color='red', linestyle='--', label='lower bound=100')
    axes[1].axvline(15000, color='red', linestyle='--', label='upper bound=15000')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/raw_pancreas_qc.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: visualizations/raw_pancreas_qc.png")
    plt.close()
    
    # 2. UMAP visualization (use existing if available)
    if 'X_umap' not in adata.obsm:
        print("\nComputing UMAP embedding...")
        sc.pp.neighbors(adata, use_rep='X_pca' if 'X_pca' in adata.obsm else 'X')
        sc.tl.umap(adata)
    
    # UMAP colored by different variables
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Plot UMAP for each covariate
    sc.pl.umap(adata, color='celltype', ax=axes[0], show=False, 
               title='Cell types', legend_loc='on data', legend_fontsize=6)
    sc.pl.umap(adata, color='batch', ax=axes[1], show=False, 
               title='Batch (technical)', palette='Set2')
    sc.pl.umap(adata, color='sample', ax=axes[2], show=False, 
               title='Sample (study)', palette='Set1')
    sc.pl.umap(adata, color='n_counts', ax=axes[3], show=False, 
               title='Total counts', cmap='viridis')
    
    plt.tight_layout()
    plt.savefig('visualizations/raw_pancreas_umap.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: visualizations/raw_pancreas_umap.png")
    plt.close()
    
    # 3. Batch composition by cell type
    import pandas as pd
    import seaborn as sns
    
    # Create contingency table
    ct_batch = pd.crosstab(
        adata.obs['celltype'], 
        adata.obs['batch'],
        normalize='columns'
    )
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(ct_batch, annot=False, fmt='.2f', cmap='Blues', 
                cbar_kws={'label': 'Proportion'}, ax=ax)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Cell type')
    ax.set_title('Cell type composition per batch')
    plt.tight_layout()
    plt.savefig('visualizations/raw_pancreas_batch_composition.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: visualizations/raw_pancreas_batch_composition.png")
    plt.close()
    
    # 4. Violin plots for QC metrics by batch
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sc.pl.violin(adata, keys='n_genes', groupby='batch', ax=axes[0], show=False)
    axes[0].set_title('Genes per cell by batch')
    
    sc.pl.violin(adata, keys='n_counts', groupby='batch', ax=axes[1], show=False)
    axes[1].set_title('Counts per cell by batch')
    
    plt.tight_layout()
    plt.savefig('visualizations/raw_pancreas_qc_by_batch.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: visualizations/raw_pancreas_qc_by_batch.png")
    plt.close()
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print("\nAll plots saved to visualizations/ directory:")
    print("  - raw_pancreas_qc.png (QC metrics)")
    print("  - raw_pancreas_umap.png (UMAP embeddings)")
    print("  - raw_pancreas_batch_composition.png (batch effects)")
    print("  - raw_pancreas_qc_by_batch.png (QC by batch)")


if __name__ == "__main__":
    import os
    
    # Create visualizations directory if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    
    # Run visualization
    visualize_raw_data()

