"""
W&B artifact management for datasets, models, and visualizations.

Provides utilities to:
- Log and version datasets (raw and processed)
- Log and version model checkpoints
- Log visualization artifacts (plots, metrics)
- Track artifact lineage (model → dataset → preprocessing)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import wandb

logger = logging.getLogger(__name__)


def log_dataset_artifact(
    dataset_path: str,
    artifact_name: str,
    artifact_type: str = "dataset",
    description: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    run: Optional[wandb.sdk.wandb_run.Run] = None,
) -> Optional[wandb.Artifact]:
    """
    Log a dataset as a W&B artifact with metadata.
    
    Args:
        dataset_path: Path to dataset file (e.g., .h5ad)
        artifact_name: Name for the artifact (e.g., "pancreas_raw", "pancreas_processed")
        artifact_type: Type of artifact (default: "dataset")
        description: Human-readable description
        metadata: Dictionary with preprocessing parameters, statistics, etc.
        run: Active W&B run (if None, uses wandb.run)
        
    Returns:
        Logged artifact object, or None if no active run
        
    Example:
        >>> log_dataset_artifact(
        ...     "data/pancreas_processed.h5ad",
        ...     "pancreas_processed",
        ...     description="Processed pancreas data with HVG selection",
        ...     metadata={"n_cells": 3500, "n_genes": 2000, "preprocessing": "..."}
        ... )
    """
    if run is None:
        run = wandb.run
    
    if run is None:
        logger.warning("No active W&B run; skipping dataset artifact logging")
        return None
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        logger.warning(f"Dataset path {dataset_path} does not exist; skipping artifact logging")
        return None
    
    # Create artifact
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=description,
        metadata=metadata or {},
    )
    
    # Add file
    artifact.add_file(str(dataset_path))
    
    # Log artifact
    run.log_artifact(artifact)
    logger.info(f"Logged dataset artifact: {artifact_name} ({dataset_path})")
    
    return artifact


def log_model_artifact(
    checkpoint_path: str,
    artifact_name: str,
    model_config: Dict[str, Any],
    metrics: Optional[Dict[str, float]] = None,
    description: str = "",
    aliases: Optional[list[str]] = None,
    run: Optional[wandb.sdk.wandb_run.Run] = None,
) -> Optional[wandb.Artifact]:
    """
    Log a model checkpoint as a W&B artifact with metadata.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        artifact_name: Name for the artifact (e.g., "invae_model")
        model_config: Model configuration dictionary
        metrics: Training/validation metrics to store with model
        description: Human-readable description
        aliases: Tags for this version (e.g., ["best", "latest", "production"])
        run: Active W&B run (if None, uses wandb.run)
        
    Returns:
        Logged artifact object, or None if no active run
        
    Example:
        >>> log_model_artifact(
        ...     "checkpoints/best_model.pt",
        ...     "invae_model",
        ...     model_config={"z_i_dim": 30, "z_s_dim": 5},
        ...     metrics={"val_loss": 1234.5},
        ...     aliases=["best", "v1"]
        ... )
    """
    if run is None:
        run = wandb.run
    
    if run is None:
        logger.warning("No active W&B run; skipping model artifact logging")
        return None
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint path {checkpoint_path} does not exist; skipping artifact logging")
        return None
    
    # Prepare metadata
    metadata = {
        "model_config": model_config,
        "checkpoint_file": checkpoint_path.name,
    }
    if metrics:
        metadata["metrics"] = metrics
    
    # Create artifact
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description=description,
        metadata=metadata,
    )
    
    # Add checkpoint file
    artifact.add_file(str(checkpoint_path))
    
    # Log artifact with aliases
    run.log_artifact(artifact, aliases=aliases or [])
    logger.info(f"Logged model artifact: {artifact_name} with aliases {aliases} ({checkpoint_path})")
    
    return artifact


def use_dataset_artifact(
    artifact_name: str,
    version: str = "latest",
    download_dir: Optional[str] = None,
) -> Path:
    """
    Download and use a dataset artifact from W&B.
    
    Args:
        artifact_name: Full artifact name (e.g., "user/project/pancreas_processed:latest")
        version: Artifact version or alias (default: "latest")
        download_dir: Directory to download to (default: W&B cache)
        
    Returns:
        Path to downloaded dataset file
        
    Example:
        >>> data_path = use_dataset_artifact("user/project/pancreas_processed:v2")
        >>> adata = sc.read(data_path / "pancreas_processed.h5ad")
    """
    if not artifact_name.count(":"):
        artifact_name = f"{artifact_name}:{version}"
    
    artifact = wandb.use_artifact(artifact_name)
    artifact_dir = artifact.download(root=download_dir)
    
    logger.info(f"Downloaded dataset artifact: {artifact_name} to {artifact_dir}")
    return Path(artifact_dir)


def use_model_artifact(
    artifact_name: str,
    version: str = "latest",
    download_dir: Optional[str] = None,
) -> tuple[Path, Dict[str, Any]]:
    """
    Download and use a model artifact from W&B.
    
    Args:
        artifact_name: Full artifact name (e.g., "user/project/invae_model:best")
        version: Artifact version or alias (default: "latest")
        download_dir: Directory to download to (default: W&B cache)
        
    Returns:
        Tuple of (checkpoint_path, metadata)
        
    Example:
        >>> ckpt_path, metadata = use_model_artifact("user/project/invae_model:best")
        >>> checkpoint = torch.load(ckpt_path / "best_model.pt")
        >>> model_config = metadata["model_config"]
    """
    if not artifact_name.count(":"):
        artifact_name = f"{artifact_name}:{version}"
    
    artifact = wandb.use_artifact(artifact_name)
    artifact_dir = artifact.download(root=download_dir)
    metadata = artifact.metadata
    
    logger.info(f"Downloaded model artifact: {artifact_name} to {artifact_dir}")
    return Path(artifact_dir), metadata


def log_visualization_artifact(
    plot_paths: list[str] | str,
    artifact_name: str,
    description: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    run: Optional[wandb.sdk.wandb_run.Run] = None,
) -> Optional[wandb.Artifact]:
    """
    Log visualization plots as a W&B artifact.
    
    Args:
        plot_paths: Path(s) to plot files (PNG, PDF, etc.)
        artifact_name: Name for the artifact (e.g., "umap_plots")
        description: Human-readable description
        metadata: Dictionary with plot metadata
        run: Active W&B run (if None, uses wandb.run)
        
    Returns:
        Logged artifact object, or None if no active run
        
    Example:
        >>> log_visualization_artifact(
        ...     ["plots/umap_batch.png", "plots/umap_celltype.png"],
        ...     "latent_visualizations",
        ...     description="UMAP plots showing batch correction"
        ... )
    """
    if run is None:
        run = wandb.run
    
    if run is None:
        logger.warning("No active W&B run; skipping visualization artifact logging")
        return None
    
    # Handle single path or list
    if isinstance(plot_paths, str):
        plot_paths = [plot_paths]
    
    # Create artifact
    artifact = wandb.Artifact(
        name=artifact_name,
        type="visualization",
        description=description,
        metadata=metadata or {},
    )
    
    # Add all plot files
    for plot_path in plot_paths:
        plot_path = Path(plot_path)
        if plot_path.exists():
            artifact.add_file(str(plot_path))
        else:
            logger.warning(f"Plot path {plot_path} does not exist; skipping")
    
    # Log artifact
    run.log_artifact(artifact)
    logger.info(f"Logged visualization artifact: {artifact_name} with {len(plot_paths)} files")
    
    return artifact


def log_metrics_table(
    metrics: Dict[str, Any],
    table_name: str = "metrics_summary",
    run: Optional[wandb.sdk.wandb_run.Run] = None,
) -> None:
    """
    Log metrics as a W&B Table for easy comparison.
    
    Args:
        metrics: Dictionary of metrics to log
        table_name: Name for the table
        run: Active W&B run (if None, uses wandb.run)
        
    Example:
        >>> log_metrics_table({
        ...     "train_loss": 1234.5,
        ...     "val_loss": 1456.7,
        ...     "batch_asw": -0.12,
        ...     "celltype_asw": 0.65,
        ... })
    """
    if run is None:
        run = wandb.run
    
    if run is None:
        logger.warning("No active W&B run; skipping metrics table logging")
        return
    
    # Flatten nested dicts
    flat_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat_metrics[f"{key}/{subkey}"] = subvalue
        else:
            flat_metrics[key] = value
    
    # Create table
    table = wandb.Table(
        columns=["metric", "value"],
        data=[[k, v] for k, v in flat_metrics.items()]
    )
    
    run.log({table_name: table})
    logger.info(f"Logged metrics table: {table_name} with {len(flat_metrics)} entries")

