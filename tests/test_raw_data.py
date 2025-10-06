import os
import anndata as ad


def test_h5ad_basic_properties():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "pancreas.h5ad")
    path = os.path.abspath(path)
    assert os.path.exists(path), f"Missing file: {path}"

    adata = ad.read_h5ad(path, backed="r")

    # Basic shape
    assert adata.n_obs > 1000
    assert adata.n_vars > 1000

    # Required obs columns (from summary)
    required_obs = {"celltype", "sample", "n_genes", "batch", "n_counts", "louvain"}
    assert required_obs.issubset(set(adata.obs.columns))

    # Common embeddings present
    for key in ("X_pca", "X_umap"):
        assert key in adata.obsm

    # Connectivity info moved to obsp; at least ensure structure keys exist
    for key in ("celltype_colors", "pca"):
        assert key in adata.uns

    # Close backed file if applicable
    try:
        adata.file.close()  # type: ignore[attr-defined]
    except Exception:
        pass

