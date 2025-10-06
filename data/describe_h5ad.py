#!/usr/bin/env python3
"""Describe an .h5ad file concisely.

This script prints a compact summary of an AnnData (.h5ad) dataset, including:
- path, backed mode, basic shape (n_obs x n_var)
- X type and density (if sparse)
- top-level keys in .obs, .var, .layers, .obsm, .varm, .uns
- dtypes of selected .obs / .var columns and category levels for categoricals
- approximate memory footprint of loaded object (best-effort)

Usage:
  python scripts/describe_h5ad.py data/pancreas.h5ad
  python scripts/describe_h5ad.py data/pancreas.h5ad --backed r

This avoids heavy computations and is safe for large datasets. For huge files,
prefer backed mode (read-only) to minimize memory usage.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Iterable, List, Optional, Tuple

import anndata as ad
import numpy as np


def _setup_logging(verbose: bool) -> None:
    """Configure logging level and format.

    Args:
        verbose: Whether to enable DEBUG-level logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _fmt_list(values: Iterable[str], limit: int = 10) -> str:
    """Format a list for compact display.

    Args:
        values: Iterable of strings to format.
        limit: Maximum number of items to display before truncation.

    Returns:
        A string like "a, b, c (n=...)".
    """
    items = list(values)
    if not items:
        return "-"
    if len(items) <= limit:
        return ", ".join(items)
    head = ", ".join(items[:limit])
    return f"{head}, ... (total={len(items)})"


def _safe_len(x: object) -> Optional[int]:
    """Return len(x) if available, else None.

    Args:
        x: Any object.

    Returns:
        Length or None if not available.
    """
    try:
        return len(x)  # type: ignore[arg-type]
    except Exception:
        return None


def _x_summary(X: object) -> Tuple[str, Optional[float]]:
    """Summarize the storage type and density of `X`.

    Args:
        X: The AnnData.X matrix (dense or sparse or backed).

    Returns:
        A tuple of (type_str, density) where density is None if unknown.
    """
    try:
        import scipy.sparse as sp  # local import to avoid hard dependency at import time
    except Exception:
        sp = None  # type: ignore[assignment]

    type_str = type(X).__name__
    density: Optional[float] = None

    if sp is not None and (isinstance(X, sp.spmatrix) or getattr(X, "__array_priority__", None) == 10.1):
        # Backed sparse may not be a standard spmatrix, so we try to access nnz safely.
        try:
            nnz = int(getattr(X, "nnz"))
            shape = tuple(getattr(X, "shape"))  # type: ignore[assignment]
            if len(shape) == 2 and shape[0] * shape[1] > 0:
                density = nnz / float(shape[0] * shape[1])
        except Exception:
            density = None
    elif isinstance(X, np.ndarray):
        density = 1.0

    return type_str, density


def _append_kv(lines: List[str], label: str, value: object) -> None:
    """Append a single key/value line to the provided list.

    Args:
        lines: Target list of lines to append to.
        label: Field label.
        value: Field value.
    """
    lines.append(f"- {label}: {value}")


def _summarize_categoricals(df, limit_cols: int = 8, limit_levels: int = 8) -> List[str]:
    """Build lines summarizing categoricals in a DataFrame-like object.

    Args:
        df: Pandas DataFrame with dtypes.
        limit_cols: Max number of columns to show.
        limit_levels: Max number of category levels to list.

    Returns:
        Lines describing categorical columns and their levels.
    """
    import pandas as pd

    lines: List[str] = []
    cat_cols = [c for c in df.columns if pd.api.types.is_categorical_dtype(df[c])]
    if not cat_cols:
        return lines
    shown_cols = cat_cols[:limit_cols]
    for col in shown_cols:
        cats = list(df[col].cat.categories)
        if len(cats) > limit_levels:
            levels = ", ".join(map(str, cats[:limit_levels])) + f", ... (total={len(cats)})"
        else:
            levels = ", ".join(map(str, cats))
        lines.append(f"  * {col}: {len(cats)} levels [{levels}]")
    if len(cat_cols) > len(shown_cols):
        lines.append(f"  * ... {len(cat_cols) - len(shown_cols)} more categorical columns")
    return lines


def describe_h5ad(
    path: str,
    backed: Optional[str] = None,
    preview_rows: int = 5,
    verbose: bool = False,
    out_path: Optional[str] = None,
) -> int:
    """Print a concise description of an .h5ad file.

    Args:
        path: Path to the .h5ad file.
        backed: Backed mode for AnnData (e.g., "r" for read-only) or None.
        preview_rows: Number of rows to preview for .obs and .var dtypes.
        verbose: If True, enable DEBUG logging.

    Returns:
        Process exit code (0 for success, non-zero for failure).
    """
    _setup_logging(verbose)

    if not os.path.exists(path):
        logging.error("File not found: %s", path)
        return 2

    try:
        adata = ad.read_h5ad(path, backed=backed)
    except Exception as exc:
        logging.exception("Failed to read .h5ad: %s", exc)
        return 3

    lines: List[str] = []

    # Basic
    n_obs, n_var = int(adata.n_obs), int(adata.n_vars)
    lines.append("AnnData Summary")
    _append_kv(lines, "path", os.path.abspath(path))
    _append_kv(lines, "backed", backed or "None")
    _append_kv(lines, "shape", f"{n_obs} x {n_var}")

    # X
    x_type, density = _x_summary(adata.X)
    dens_str = f" (density={density:.4f})" if density is not None else ""
    _append_kv(lines, "X", f"{x_type}{dens_str}")

    # Keys
    _append_kv(lines, "obs.columns", _fmt_list(map(str, list(adata.obs.columns))))
    _append_kv(lines, "var.columns", _fmt_list(map(str, list(adata.var.columns))))
    _append_kv(lines, "layers", _fmt_list(sorted(list(adata.layers.keys()))))
    _append_kv(lines, "obsm", _fmt_list(sorted(list(adata.obsm.keys()))))
    _append_kv(lines, "varm", _fmt_list(sorted(list(adata.varm.keys()))))
    _append_kv(lines, "uns", _fmt_list(sorted(list(adata.uns.keys()))))

    # Dtypes preview
    try:
        obs_dtypes = adata.obs.dtypes.astype(str).to_dict()
        var_dtypes = adata.var.dtypes.astype(str).to_dict()
        obs_head = list(adata.obs.head(preview_rows).columns)
        var_head = list(adata.var.head(preview_rows).columns)
        _append_kv(lines, "obs dtypes", _fmt_list([f"{c}:{obs_dtypes[c]}" for c in obs_head]))
        _append_kv(lines, "var dtypes", _fmt_list([f"{c}:{var_dtypes[c]}" for c in var_head]))
    except Exception as exc:
        logging.debug("Failed dtypes preview: %s", exc)

    # Categoricals
    try:
        cat_lines = _summarize_categoricals(adata.obs)
        if cat_lines:
            lines.append("- obs categoricals:")
            lines.extend(cat_lines)
    except Exception as exc:
        logging.debug("Failed categoricals summary: %s", exc)

    # Memory footprint (best-effort)
    try:
        import pympler.asizeof as asizeof  # type: ignore[import-not-found]  # optional

        size_bytes = asizeof.asizeof(adata)
        _append_kv(lines, "approx size", f"{size_bytes/1024/1024:.2f} MiB")
    except Exception:
        # Fallback: minimal info based on arrays we have direct access to
        try:
            bytes_x = getattr(adata.X, "data", adata.X)
            nbytes = int(getattr(bytes_x, "nbytes", 0))
            _append_kv(lines, "approx size", f">= {nbytes/1024/1024:.2f} MiB (X only)")
        except Exception:
            pass

    # Final hints
    lines.append("")
    lines.append("Tip: use --backed r for large files to reduce memory usage.")

    # Decide output path
    if out_path is None:
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(os.path.dirname(path), f"{base}.summary.txt")

    # Write to file
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as exc:
        logging.warning("Failed to write summary to %s: %s", out_path, exc)

    # Also print to stdout for convenience
    print("\n".join(lines))
    print(f"\nSaved summary to: {os.path.abspath(out_path)}")

    # Ensure clean close when backed
    try:
        adata.file.close()  # type: ignore[attr-defined]
    except Exception:
        pass

    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional list of CLI arguments, defaults to sys.argv[1:].

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Describe an .h5ad file concisely")
    parser.add_argument("path", type=str, help="Path to .h5ad file")
    parser.add_argument("--backed", type=str, default=None, help='Backed mode (e.g., "r")')
    parser.add_argument("--preview-rows", type=int, default=5, help="Rows to preview for dtypes")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--out", type=str, default=None, help="Path to save summary .txt")
    return parser.parse_args(argv)


def main() -> None:
    """Entry point for the CLI."""
    args = parse_args()
    code = describe_h5ad(
        path=args.path,
        backed=args.backed,
        preview_rows=args.preview_rows,
        verbose=args.verbose,
        out_path=args.out,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()


