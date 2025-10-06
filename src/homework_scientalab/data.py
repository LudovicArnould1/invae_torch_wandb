"""
Data preparation and dataset utilities for inVAE training on single-cell RNA-seq data.
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import scanpy as sc
from anndata import AnnData
from scipy import sparse as sp

from homework_scientalab.config import DataConfig


class SingleCellDataset(Dataset):
    """
    PyTorch Dataset for single-cell count data.
    
    Returns batches with:
        - 'x': raw counts (B, G)
        - 'b': biological covariates (B, d_b) - one-hot encoded cell types
        - 't': technical covariates (B, d_t) - one-hot encoded batches
        - 'size_factor': library size normalization factors (B,)
    """
    
    def __init__(
        self,
        counts: np.ndarray,
        batch_onehot: np.ndarray,
        celltype_onehot: np.ndarray,
        size_factors: np.ndarray,
    ):
        """
        Args:
            counts: Raw count matrix (n_cells, n_genes)
            batch_onehot: One-hot encoded batch labels (n_cells, n_batches)
            celltype_onehot: One-hot encoded cell type labels (n_cells, n_celltypes)
            size_factors: Library size normalization factors (n_cells,)
        """
        self.counts = torch.FloatTensor(counts)
        self.batch_onehot = torch.FloatTensor(batch_onehot)
        self.celltype_onehot = torch.FloatTensor(celltype_onehot)
        self.size_factors = torch.FloatTensor(size_factors)
        
    def __len__(self) -> int:
        return len(self.counts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.counts[idx],
            "b": self.celltype_onehot[idx],  # biological
            "t": self.batch_onehot[idx],  # technical
            "size_factor": self.size_factors[idx],
        }


def load_and_preprocess_data(
    cfg: DataConfig,
    save_path: str | None = None
) -> AnnData:
    """
    Load and preprocess single-cell data.
    
    Steps:
    1. Load raw data
    2. QC filtering (min genes/cells, optional mito filtering, count filtering)
    3. Compute HVGs on log-normalized data for feature selection
    4. Store raw counts in adata.layers["counts"]
    5. Subset to HVGs if requested
    
    Args:
        cfg: DataConfig with preprocessing parameters
        save_path: Optional path to save processed AnnData
        
    Returns:
        AnnData object with raw counts in .layers["counts"] and metadata in .obs
    """
    print(f"Loading data from {cfg.data_path}...")
    adata = sc.read(cfg.data_path, backup_url=cfg.backup_url)
    
    print(f"Initial shape: {adata.shape}")
    print(f"Cell types:\n{adata.obs[cfg.celltype_key].value_counts()}")
    print(f"Batches:\n{adata.obs[cfg.batch_key].value_counts()}")
    
    # Store original raw counts
    # Prefer .raw.X only if it matches current genes; otherwise use current X.
    counts_source = adata.X
    try:
        if getattr(adata, "raw", None) is not None and getattr(adata.raw, "X", None) is not None:
            raw_X = adata.raw.X
            # Use .raw only if gene dimension matches current n_vars
            if getattr(raw_X, "shape", None) and raw_X.shape[1] == adata.n_vars:
                counts_source = raw_X
    except Exception:
        counts_source = adata.X

    # Coerce to non-negative integers while preserving sparsity structure
    if sp.issparse(counts_source):
        counts_csr = counts_source.tocsr(copy=True)
        counts_csr.data = np.rint(np.clip(counts_csr.data, 0, None)).astype(np.int64, copy=False)
        adata.layers["counts"] = counts_csr
    else:
        counts_arr = np.rint(np.clip(np.asarray(counts_source), 0, None)).astype(np.int64, copy=False)
        adata.layers["counts"] = counts_arr
    
    # 1. QC filtering
    print("\n=== Quality Control ===")
    sc.pp.filter_genes(adata, min_cells=cfg.min_cells)
    sc.pp.filter_cells(adata, min_genes=cfg.min_genes)
    
    # Optional: mitochondrial filtering
    if cfg.max_mito_pct is not None:
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
        adata = adata[adata.obs["pct_counts_mt"] < cfg.max_mito_pct * 100].copy()
        print(f"After mito filtering: {adata.shape}")
    
    print(f"After QC: {adata.shape}")
    
    counts_layer = adata.layers.get("counts", adata.X)
    if sp.issparse(counts_layer):
        cell_counts = np.asarray(counts_layer.sum(axis=1)).ravel()
    else:
        cell_counts = counts_layer.sum(axis=1)
    before_n = adata.n_obs
    mask = (cell_counts >= cfg.min_cell_counts) & (cell_counts <= cfg.max_cell_counts)
    adata = adata[mask].copy()
    removed_n = before_n - adata.n_obs
    print(f"After count filtering [{cfg.min_cell_counts}, {cfg.max_cell_counts}]: {adata.shape} (removed {removed_n} cells)")

    sc.pp.normalize_total(adata, target_sum=cfg.normalization_target_sum)
    sc.pp.log1p(adata)
    
    # 3. HVG selection
    if cfg.n_top_genes is not None:
        print(f"\n=== Selecting {cfg.n_top_genes} HVGs ===")
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=cfg.n_top_genes,
            flavor="seurat_v3",
            layer="counts",
            subset=False,
        )
        # Subset to HVGs
        adata = adata[:, adata.var["highly_variable"]].copy()
        print(f"After HVG selection: {adata.shape}")
    
    print(f"\nFinal dataset shape: {adata.shape}")
    print(f"Final cell types:\n{adata.obs[cfg.celltype_key].value_counts()}")
    print(f"Final batches:\n{adata.obs[cfg.batch_key].value_counts()}")
    
    # Save processed data if requested
    if save_path is not None:
        print("\n=== Saving Processed Data ===")
        print(f"Saving to: {save_path}")
        adata.write(save_path)
        print(f"Successfully saved processed AnnData to {save_path}")
    
    return adata


def compute_size_factors(
    counts: np.ndarray,
    clip_percentile: float = 99.5,
    min_clip: float = 1.0
) -> np.ndarray:
    """
    Compute library size (total counts per cell).
    
    Args:
        counts: Raw count matrix (n_cells, n_genes)
        clip_percentile: Clip extreme values at this percentile to avoid outliers
        min_clip: Minimum value to clip library sizes
        
    Returns:
        Library sizes (n_cells,) - raw count sums per cell
    """
    library_size = counts.sum(axis=1)
    upper_bound = np.percentile(library_size, clip_percentile)
    library_size = np.clip(library_size, min_clip, upper_bound)
    
    print("\nLibrary size statistics:")
    print(f"  Min: {library_size.min():.1f}")
    print(f"  Median: {np.median(library_size):.1f}")
    print(f"  Max: {library_size.max():.1f}")
    print(f"  Mean: {library_size.mean():.1f}")
    
    return library_size


def prepare_covariates(
    adata: AnnData,
    batch_key: str = "tech",
    celltype_key: str = "celltype"
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, LabelEncoder]:
    """
    Prepare one-hot encoded covariates.
    
    Args:
        adata: AnnData object with obs containing batch and celltype info
        batch_key: Column name in adata.obs for batch/technical covariate
        celltype_key: Column name in adata.obs for celltype/biological covariate
        
    Returns:
        batch_onehot: (n_cells, n_batches) one-hot encoded batches
        celltype_onehot: (n_cells, n_celltypes) one-hot encoded cell types
        batch_encoder: LabelEncoder for batches
        celltype_encoder: LabelEncoder for cell types
    """
    # Encode batches
    batch_encoder = LabelEncoder()
    batch_labels = batch_encoder.fit_transform(adata.obs[batch_key])
    n_batches = len(batch_encoder.classes_)
    batch_onehot = np.eye(n_batches)[batch_labels]
    
    # Encode cell types
    celltype_encoder = LabelEncoder()
    celltype_labels = celltype_encoder.fit_transform(adata.obs[celltype_key])
    n_celltypes = len(celltype_encoder.classes_)
    celltype_onehot = np.eye(n_celltypes)[celltype_labels]
    
    print("\nCovariate dimensions:")
    print(f"  Batches: {n_batches} ({batch_encoder.classes_})")
    print(f"  Cell types: {n_celltypes} ({celltype_encoder.classes_})")
    
    return batch_onehot, celltype_onehot, batch_encoder, celltype_encoder


def stratified_train_val_split(
    adata: AnnData,
    batch_key: str = "tech",
    celltype_key: str = "celltype",
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create stratified train/val split by batch and cell type.
    
    Args:
        adata: AnnData object
        batch_key: Column name for batch stratification
        celltype_key: Column name for celltype stratification
        val_size: Fraction of data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        train_idx: Indices for training set
        val_idx: Indices for validation set
    """
    # Create combined stratification key
    strat_labels = (
        adata.obs[batch_key].astype(str) + "_" + 
        adata.obs[celltype_key].astype(str)
    )
    
    indices = np.arange(len(adata))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        stratify=strat_labels,
        random_state=random_state,
    )
    
    print("\n=== Train/Val Split ===")
    print(f"Train: {len(train_idx)} cells ({len(train_idx)/len(adata)*100:.1f}%)")
    print(f"Val:   {len(val_idx)} cells ({len(val_idx)/len(adata)*100:.1f}%)")
    
    # Verify stratification
    print("\nTrain set distribution:")
    print(f"  Batches:\n{adata.obs.iloc[train_idx][batch_key].value_counts()}")
    print(f"  Cell types:\n{adata.obs.iloc[train_idx][celltype_key].value_counts()}")
    
    print("\nVal set distribution:")
    print(f"  Batches:\n{adata.obs.iloc[val_idx][batch_key].value_counts()}")
    print(f"  Cell types:\n{adata.obs.iloc[val_idx][celltype_key].value_counts()}")
    
    return train_idx, val_idx


def prepare_dataloaders(
    cfg: DataConfig,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    save_processed: bool = True,
    log_artifacts: bool = False,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Complete data preparation pipeline.
    
    Args:
        cfg: DataConfig with preprocessing parameters
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        pin_memory: Pin memory for faster GPU transfer
        save_processed: If True, saves processed data as pancreas_processed.h5ad
        log_artifacts: If True, logs raw and processed data as W&B artifacts
        
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        dims: Dictionary with 'x_dim' (n_genes), 'b_dim' (n_celltypes), 't_dim' (n_batches)
    """
    print("=" * 80)
    print("DATA PREPARATION PIPELINE")
    print("=" * 80)
    
    # 1. Load and preprocess (don't save yet - need to add split info first)
    adata = load_and_preprocess_data(cfg, save_path=None)
    
    # 2. Extract raw counts from the layer
    counts = adata.layers["counts"]
    if hasattr(counts, "toarray"):  # Handle sparse matrices
        counts = counts.toarray()
    counts = counts.astype(np.float32)
    
    # 3. Compute size factors
    size_factors = compute_size_factors(
        counts, 
        clip_percentile=cfg.clip_size_factor_percentile,
        min_clip=cfg.size_factor_min_clip
    )
    
    # 4. Prepare covariates
    batch_onehot, celltype_onehot, batch_enc, celltype_enc = prepare_covariates(
        adata, cfg.batch_key, cfg.celltype_key
    )
    
    # 5. Train/val split
    train_idx, val_idx = stratified_train_val_split(
        adata, cfg.batch_key, cfg.celltype_key, cfg.val_size, cfg.random_state
    )
    
    # Add split labels to adata for later visualization
    adata.obs["split"] = "train"
    adata.obs.iloc[val_idx, adata.obs.columns.get_loc("split")] = "val"
    print("\nAdded 'split' column to adata.obs:")
    print(f"  Train: {(adata.obs['split'] == 'train').sum()} cells")
    print(f"  Val: {(adata.obs['split'] == 'val').sum()} cells")
    
    # Save processed data with split info
    processed_path = None
    if save_processed:
        processed_path = "data/pancreas_processed.h5ad"
        print("\n=== Saving Processed Data ===")
        print(f"Saving to: {processed_path}")
        adata.write(processed_path)
        print(f"Successfully saved processed AnnData with split info to {processed_path}")
    
    # Log data artifacts to W&B
    if log_artifacts:
        from homework_scientalab.monitor_and_setup.artifacts import log_dataset_artifact
        from pathlib import Path
        
        print("\n=== Logging Data Artifacts to W&B ===")
        
        # Log raw data (if exists)
        if Path(cfg.data_path).exists():
            raw_metadata = {
                "source": cfg.data_path,
                "backup_url": cfg.backup_url,
            }
            log_dataset_artifact(
                cfg.data_path,
                artifact_name="pancreas_raw",
                description="Raw pancreas scRNA-seq data",
                metadata=raw_metadata,
            )
        
        # Log processed data
        if processed_path and Path(processed_path).exists():
            processed_metadata = {
                "n_cells": adata.n_obs,
                "n_genes": adata.n_vars,
                "n_batches": len(adata.obs[cfg.batch_key].cat.categories),
                "n_celltypes": len(adata.obs[cfg.celltype_key].cat.categories),
                "train_cells": (adata.obs["split"] == "train").sum(),
                "val_cells": (adata.obs["split"] == "val").sum(),
                "preprocessing": {
                    "min_genes": cfg.min_genes,
                    "min_cells": cfg.min_cells,
                    "n_top_genes": cfg.n_top_genes,
                    "val_size": cfg.val_size,
                    "random_state": cfg.random_state,
                },
            }
            log_dataset_artifact(
                processed_path,
                artifact_name="pancreas_processed",
                description="Preprocessed pancreas data with HVG selection and train/val split",
                metadata=processed_metadata,
            )
        
        print("✓ Data artifacts logged to W&B")
    
    # 6. Create datasets
    train_dataset = SingleCellDataset(
        counts=counts[train_idx],
        batch_onehot=batch_onehot[train_idx],
        celltype_onehot=celltype_onehot[train_idx],
        size_factors=size_factors[train_idx],
    )
    
    val_dataset = SingleCellDataset(
        counts=counts[val_idx],
        batch_onehot=batch_onehot[val_idx],
        celltype_onehot=celltype_onehot[val_idx],
        size_factors=size_factors[val_idx],
    )
    
    # 7. Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch for stable BatchNorm
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    # 8. Store dimensions
    dims = {
        "x_dim": counts.shape[1],  # n_genes
        "b_dim": celltype_onehot.shape[1],  # n_celltypes
        "t_dim": batch_onehot.shape[1],  # n_batches
    }
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
    print("Dimensions:")
    print(f"  Genes (x_dim): {dims['x_dim']}")
    print(f"  Cell types (b_dim): {dims['b_dim']}")
    print(f"  Batches (t_dim): {dims['t_dim']}")
    print("\nDataLoaders:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Batch size: {batch_size}")
    print("=" * 80 + "\n")
    
    return train_loader, val_loader, dims

