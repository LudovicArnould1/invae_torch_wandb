import os
import numpy as np
import scanpy as sc
from homework_scientalab.data import DataConfig, load_and_preprocess_data, stratified_train_val_split


def test_processing_and_stratified_split():
    # Configure to be reasonably fast while exercising the pipeline
    cfg = DataConfig(
        data_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "pancreas.h5ad")),
        n_top_genes=1000,
        batch_key="batch",
        celltype_key="celltype",
        val_size=0.2,
        random_state=0,
    )

    adata = load_and_preprocess_data(cfg)

    # counts layer exists and is non-negative
    assert "counts" in adata.layers
    counts = adata.layers["counts"]
    if hasattr(counts, "toarray"):
        counts = counts.toarray()
    assert np.all(counts >= 0)

    # After HVG selection, number of genes <= n_top_genes
    if cfg.n_top_genes is not None:
        assert adata.n_vars <= cfg.n_top_genes

    # Split
    train_idx, val_idx = stratified_train_val_split(
        adata, batch_key=cfg.batch_key, celltype_key=cfg.celltype_key, val_size=cfg.val_size, random_state=cfg.random_state
    )

    # Basic split properties
    assert len(train_idx) > 0 and len(val_idx) > 0
    assert set(train_idx).isdisjoint(set(val_idx))
    assert len(set(train_idx).union(set(val_idx))) == adata.n_obs

    # Each batch present in both splits
    batches_all = set(adata.obs[cfg.batch_key].astype(str))
    batches_train = set(adata.obs.iloc[train_idx][cfg.batch_key].astype(str))
    batches_val = set(adata.obs.iloc[val_idx][cfg.batch_key].astype(str))
    assert batches_all == batches_train == batches_val

    # For celltypes with sufficient support, expect presence in both splits
    celltype_counts = adata.obs[cfg.celltype_key].value_counts()
    celltypes_train = set(adata.obs.iloc[train_idx][cfg.celltype_key].astype(str))
    celltypes_val = set(adata.obs.iloc[val_idx][cfg.celltype_key].astype(str))
    well_supported = set(celltype_counts[celltype_counts >= 5].index.astype(str))
    assert well_supported.issubset(celltypes_train)
    assert well_supported.issubset(celltypes_val)

