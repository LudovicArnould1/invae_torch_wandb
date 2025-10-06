"""
Pipeline stage separation for inVAE training.

This module provides a modular pipeline with distinct stages:
1. Data Preparation: Load, preprocess, and log data artifacts
2. Training: Train model on versioned data artifact
3. Evaluation: Evaluate model and generate visualizations
4. Analysis: Detailed analysis and reports

Each stage can be run independently or as part of a full pipeline.
All stages share a single W&B run for tracking lineage.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from homework_scientalab.config import DataConfig, TrainConfig, InVAEConfig
from homework_scientalab.data import (
    load_and_preprocess_data,
    compute_size_factors,
    prepare_covariates,
    stratified_train_val_split,
    SingleCellDataset,
)
from homework_scientalab.model import InVAE
from homework_scientalab.trainer import InVAETrainer
from homework_scientalab.muon_optimizer import get_muon_optimizer
from homework_scientalab.monitor_and_setup.reproducibility import set_seed, log_environment_to_wandb
from homework_scientalab.monitor_and_setup.artifacts import (
    log_dataset_artifact,
    log_model_artifact,
    use_dataset_artifact,
    use_model_artifact,
)

logger = logging.getLogger(__name__)


@dataclass
class StageOutput:
    """
    Standard output format for pipeline stages.
    
    Attributes:
        artifacts: Dictionary mapping artifact names to their W&B references
        metadata: Additional metadata from the stage
        local_paths: Dictionary mapping artifact names to local file paths (for caching)
    """
    artifacts: Dict[str, str]  # artifact_name -> "entity/project/name:version"
    metadata: Dict[str, Any]
    local_paths: Dict[str, Path]  # artifact_name -> local path


class PipelineStage(ABC):
    """
    Base class for pipeline stages.
    
    Each stage:
    - Takes specific inputs (configs, artifact references)
    - Produces outputs (artifacts, metrics, visualizations)
    - Logs everything to shared W&B run
    - Returns StageOutput for next stage
    """
    
    def __init__(self, wandb_run: Optional[wandb.sdk.wandb_run.Run] = None):
        """
        Initialize pipeline stage.
        
        Args:
            wandb_run: Active W&B run (shared across all stages)
        """
        self.wandb_run = wandb_run or wandb.run
        
    @abstractmethod
    def run(self, **kwargs) -> StageOutput:
        """
        Execute the stage.
        
        Returns:
            StageOutput with artifacts and metadata
        """
        pass
    
    def _get_artifact_path(
        self, 
        artifact_ref: Optional[str] = None,
        local_path: Optional[str] = None,
    ) -> Path:
        """
        Get path to artifact, checking local cache first, then downloading from W&B.
        
        Args:
            artifact_ref: W&B artifact reference (e.g., "user/project/name:version")
            local_path: Local path to check first
            
        Returns:
            Path to artifact file
        """
        # Check local cache first
        if local_path and Path(local_path).exists():
            logger.info(f"Using cached artifact from {local_path}")
            return Path(local_path)
        
        # Download from W&B if artifact ref provided
        if artifact_ref:
            logger.info(f"Downloading artifact {artifact_ref} from W&B")
            artifact_dir = use_dataset_artifact(artifact_ref)
            # Return first file in artifact directory
            files = list(artifact_dir.glob("*"))
            if files:
                return files[0]
            raise FileNotFoundError(f"No files found in artifact {artifact_ref}")
        
        raise ValueError("Must provide either artifact_ref or local_path")


class DataPreparationStage(PipelineStage):
    """
    Stage 1: Data Preparation
    
    Input: DataConfig
    Output: Processed dataset artifact with train/val split
    Logs: Data statistics, QC plots, split information
    """
    
    def run(
        self,
        data_cfg: DataConfig,
        save_processed: bool = True,
    ) -> StageOutput:
        """
        Execute data preparation pipeline.
        
        Args:
            data_cfg: Data preprocessing configuration
            save_processed: Whether to save processed data locally
            
        Returns:
            StageOutput with processed dataset artifact
        """
        logger.info("=" * 80)
        logger.info("STAGE 1: DATA PREPARATION")
        logger.info("=" * 80)
        
        # 1. Load and preprocess data
        adata = load_and_preprocess_data(data_cfg, save_path=None)
        
        # 2. Extract raw counts
        counts = adata.layers["counts"]
        if hasattr(counts, "toarray"):
            counts = counts.toarray()
        counts = counts.astype("float32")
        
        # 3. Compute size factors
        size_factors = compute_size_factors(
            counts,
            clip_percentile=data_cfg.clip_size_factor_percentile,
            min_clip=data_cfg.size_factor_min_clip,
        )
        
        # 4. Prepare covariates
        batch_onehot, celltype_onehot, batch_enc, celltype_enc = prepare_covariates(
            adata, data_cfg.batch_key, data_cfg.celltype_key
        )
        
        # 5. Train/val split
        train_idx, val_idx = stratified_train_val_split(
            adata, data_cfg.batch_key, data_cfg.celltype_key, 
            data_cfg.val_size, data_cfg.random_state
        )
        
        # Add split labels to adata
        adata.obs["split"] = "train"
        adata.obs.iloc[val_idx, adata.obs.columns.get_loc("split")] = "val"
        
        # Store config keys in uns for later retrieval
        adata.uns["batch_key"] = data_cfg.batch_key
        adata.uns["celltype_key"] = data_cfg.celltype_key
        
        # 6. Save processed data locally
        processed_path = Path("data/pancreas_processed.h5ad")
        if save_processed:
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving processed data to {processed_path}")
            adata.write(str(processed_path))
        
        # 7. Log dataset artifact to W&B
        processed_metadata = {
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "n_batches": len(batch_onehot[0]),
            "n_celltypes": len(celltype_onehot[0]),
            "train_cells": len(train_idx),
            "val_cells": len(val_idx),
            "preprocessing": {
                "min_genes": data_cfg.min_genes,
                "min_cells": data_cfg.min_cells,
                "n_top_genes": data_cfg.n_top_genes,
                "val_size": data_cfg.val_size,
                "random_state": data_cfg.random_state,
            },
        }
        
        artifact = None
        artifact_ref = None
        if self.wandb_run:
            artifact = log_dataset_artifact(
                str(processed_path),
                artifact_name="pancreas_processed",
                description="Preprocessed pancreas data with HVG selection and train/val split",
                metadata=processed_metadata,
                run=self.wandb_run,
            )
            if artifact:
                # In offline mode, artifact properties are not available until synced
                # Use a placeholder reference or skip
                try:
                    artifact_ref = f"{artifact.entity}/{artifact.project}/{artifact.name}:{artifact.version}"
                except Exception:
                    # Offline mode - artifact ref not available
                    logger.warning("Running in offline mode - artifact reference not available")
                    artifact_ref = None
        
        # 8. Prepare output
        dims = {
            "x_dim": counts.shape[1],
            "b_dim": celltype_onehot.shape[1],
            "t_dim": batch_onehot.shape[1],
        }
        
        metadata = {
            **processed_metadata,
            "dims": dims,
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
        }
        
        logger.info("=" * 80)
        logger.info("DATA PREPARATION COMPLETE")
        logger.info(f"Dataset shape: {adata.shape}")
        logger.info(f"Dimensions: {dims}")
        logger.info("=" * 80)
        
        return StageOutput(
            artifacts={"dataset": artifact_ref} if artifact_ref else {},
            metadata=metadata,
            local_paths={"dataset": processed_path},
        )


class TrainingStage(PipelineStage):
    """
    Stage 2: Model Training
    
    Input: Dataset artifact (or local path), TrainConfig
    Output: Trained model artifact, training curves
    Logs: Metrics, gradients, model snapshots
    """
    
    def run(
        self,
        train_cfg: TrainConfig,
        data_output: StageOutput,
        device: Optional[torch.device] = None,
    ) -> StageOutput:
        """
        Execute training pipeline.
        
        Args:
            train_cfg: Training configuration
            data_output: Output from DataPreparationStage
            device: Device to train on (defaults to cuda if available)
            
        Returns:
            StageOutput with trained model artifact
        """
        logger.info("=" * 80)
        logger.info("STAGE 2: MODEL TRAINING")
        logger.info("=" * 80)
        
        # 1. Set seeds for reproducibility
        logger.info(f"Setting random seed to {train_cfg.seed} (deterministic={train_cfg.deterministic})")
        set_seed(train_cfg.seed, train_cfg.deterministic)
        
        # 2. Setup device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # 3. Load data (from cache or download)
        dataset_path = self._get_artifact_path(
            artifact_ref=data_output.artifacts.get("dataset"),
            local_path=str(data_output.local_paths.get("dataset", "")),
        )
        
        logger.info(f"Loading processed data from {dataset_path}")
        import scanpy as sc
        adata = sc.read(str(dataset_path))
        
        # Extract data components
        counts = adata.layers["counts"]
        if hasattr(counts, "toarray"):
            counts = counts.toarray()
        counts = counts.astype("float32")
        
        # Get indices from metadata or adata
        if "train_indices" in data_output.metadata:
            train_idx = np.array(data_output.metadata["train_indices"])
            val_idx = np.array(data_output.metadata["val_indices"])
        else:
            # Fallback: extract from adata split column
            train_idx = np.where(adata.obs["split"] == "train")[0]
            val_idx = np.where(adata.obs["split"] == "val")[0]
        
        # Prepare covariates
        from homework_scientalab.data import prepare_covariates
        batch_onehot, celltype_onehot, _, _ = prepare_covariates(
            adata, 
            adata.uns.get("batch_key", "batch"),
            adata.uns.get("celltype_key", "celltype"),
        )
        
        # Compute size factors
        size_factors = compute_size_factors(counts)
        
        # 4. Create datasets and dataloaders
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
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=train_cfg.num_workers,
            pin_memory=train_cfg.pin_memory,
            drop_last=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=train_cfg.pin_memory,
            drop_last=False,
        )
        
        # 5. Build model
        dims = data_output.metadata.get("dims", {
            "x_dim": counts.shape[1],
            "b_dim": celltype_onehot.shape[1],
            "t_dim": batch_onehot.shape[1],
        })
        
        model_cfg = InVAEConfig(
            x_dim=dims["x_dim"],
            b_dim=dims["b_dim"],
            t_dim=dims["t_dim"],
            z_i_dim=train_cfg.z_i_dim,
            z_s_dim=train_cfg.z_s_dim,
            enc_hidden=train_cfg.enc_hidden,
            dec_hidden=train_cfg.dec_hidden,
            prior_hidden=train_cfg.prior_hidden,
            dropout=train_cfg.dropout,
            batchnorm=train_cfg.batchnorm,
            use_library_size=train_cfg.use_library_size,
        )
        model = InVAE(model_cfg)
        
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model initialized with {n_params:,} trainable parameters")
        
        # 6. Log model config to W&B
        if self.wandb_run:
            self.wandb_run.config.update(
                {f"model/{k}": v for k, v in asdict(model_cfg).items()},
                allow_val_change=True,
            )
            self.wandb_run.config.update({"model/n_params": n_params}, allow_val_change=True)
            self.wandb_run.watch(model, log="gradients", log_freq=100)
        
        # 7. Setup optimizer
        params_2d = [p for p in model.parameters() if p.ndim == 2 and p.requires_grad]
        params_other = [p for p in model.parameters() if p.ndim != 2 and p.requires_grad]
        
        optimizer = get_muon_optimizer(
            params_2d,
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        
        optimizer_other = None
        if params_other:
            optimizer_other = torch.optim.AdamW(
                params_other,
                lr=train_cfg.learning_rate,
                weight_decay=train_cfg.weight_decay,
            )
        
        # 8. Setup trainer
        trainer = InVAETrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            beta_schedule=train_cfg.warmup_schedule,
            lambda_indep=train_cfg.lambda_indep,
            grad_clip=train_cfg.grad_clip,
            amp=train_cfg.amp,
            optimizer_other=optimizer_other,
        )
        
        # 9. Training loop
        save_dir = Path(train_cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float("inf")
        best_checkpoint_path = save_dir / "best_model.pt"
        
        logger.info("Starting training...")
        for epoch in range(1, train_cfg.n_epochs + 1):
            # Train
            train_metrics = trainer.train_epoch(train_loader)
            
            # Validate
            val_metrics = trainer.eval_epoch(val_loader)
            
            # Print summary
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch:3d}/{train_cfg.n_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.2f} | "
                    f"Val Loss: {val_metrics['loss']:.2f} | "
                    f"Beta: {train_metrics['beta']:.3f}"
                )
            
            # Log to W&B
            if self.wandb_run:
                self.wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_metrics["loss"],
                        "train/recon_loss": train_metrics["recon"],
                        "train/kl_i": train_metrics["kl_i"],
                        "train/kl_s": train_metrics["kl_s"],
                        "val/loss": val_metrics["loss"],
                        "val/recon_loss": val_metrics["recon"],
                        "val/kl_i": val_metrics["kl_i"],
                        "val/kl_s": val_metrics["kl_s"],
                        "train/beta": train_metrics["beta"],
                    },
                    step=trainer.global_step,
                )
            
            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "model_config": asdict(model_cfg),
                    },
                    best_checkpoint_path,
                )
                
                # Log to W&B
                if self.wandb_run:
                    self.wandb_run.summary["best_val_loss"] = best_val_loss
                    self.wandb_run.summary["best_epoch"] = epoch
        
        # 10. Log best model artifact
        artifact = None
        artifact_ref = None
        if self.wandb_run:
            artifact = log_model_artifact(
                str(best_checkpoint_path),
                artifact_name="invae_model",
                model_config=asdict(model_cfg),
                metrics={
                    "val_loss": float(best_val_loss),
                    "best_epoch": epoch,
                },
                description=f"Best model with val_loss={best_val_loss:.2f}",
                aliases=["best", "latest"],
                run=self.wandb_run,
            )
            if artifact:
                try:
                    artifact_ref = f"{artifact.entity}/{artifact.project}/{artifact.name}:{artifact.version}"
                except Exception:
                    # Offline mode - artifact ref not available
                    logger.warning("Running in offline mode - artifact reference not available")
                    artifact_ref = None
        
        logger.info("=" * 80)
        logger.info(f"TRAINING COMPLETE - Best val loss: {best_val_loss:.2f}")
        logger.info("=" * 80)
        
        return StageOutput(
            artifacts={"model": artifact_ref} if artifact_ref else {},
            metadata={
                "best_val_loss": float(best_val_loss),
                "n_epochs": train_cfg.n_epochs,
                "model_config": asdict(model_cfg),
            },
            local_paths={"model": best_checkpoint_path},
        )


class EvaluationStage(PipelineStage):
    """
    Stage 3: Model Evaluation
    
    Input: Model artifact, Dataset artifact
    Output: Evaluation metrics, basic visualizations
    Logs: Metrics tables, evaluation results
    """
    
    def run(
        self,
        data_output: StageOutput,
        training_output: StageOutput,
        device: Optional[torch.device] = None,
    ) -> StageOutput:
        """
        Execute evaluation pipeline.
        
        Args:
            data_output: Output from DataPreparationStage
            training_output: Output from TrainingStage
            device: Device to run evaluation on
            
        Returns:
            StageOutput with evaluation metrics
        """
        logger.info("=" * 80)
        logger.info("STAGE 3: MODEL EVALUATION")
        logger.info("=" * 80)
        
        # Setup device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        model_path = self._get_artifact_path(
            artifact_ref=training_output.artifacts.get("model"),
            local_path=str(training_output.local_paths.get("model", "")),
        )
        
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Reconstruct model
        model_cfg_dict = checkpoint.get("model_config", training_output.metadata.get("model_config"))
        model_cfg = InVAEConfig(**model_cfg_dict)
        model = InVAE(model_cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
        
        # Load dataset
        dataset_path = self._get_artifact_path(
            artifact_ref=data_output.artifacts.get("dataset"),
            local_path=str(data_output.local_paths.get("dataset", "")),
        )
        
        logger.info(f"Loading dataset from {dataset_path}")
        import scanpy as sc
        adata = sc.read(str(dataset_path))
        
        # Extract validation set
        val_mask = adata.obs["split"] == "val"
        logger.info(f"Evaluating on {val_mask.sum()} validation cells")
        
        # Compute basic evaluation metrics
        val_loss = checkpoint.get("val_loss", "N/A")
        
        evaluation_metrics = {
            "val_loss": val_loss,
            "n_val_cells": int(val_mask.sum()),
            "model_epoch": checkpoint.get("epoch", "unknown"),
        }
        
        # Log to W&B
        if self.wandb_run:
            from homework_scientalab.monitor_and_setup.artifacts import log_metrics_table
            log_metrics_table(
                evaluation_metrics,
                table_name="evaluation_metrics",
                run=self.wandb_run,
            )
        
        logger.info("=" * 80)
        logger.info("EVALUATION COMPLETE")
        logger.info(f"Validation Loss: {val_loss}")
        logger.info("=" * 80)
        
        return StageOutput(
            artifacts={},
            metadata=evaluation_metrics,
            local_paths={},
        )


class Pipeline:
    """
    Orchestrates the full training pipeline with artifact tracking.
    
    Stages:
    1. Data Preparation
    2. Training
    3. Evaluation
    4. (Future) Analysis
    """
    
    def __init__(
        self,
        project: str = "invae-pancreas",
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        """
        Initialize pipeline.
        
        Args:
            project: W&B project name
            entity: W&B entity (username/team)
            run_name: Optional run name
        """
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.wandb_run = None
        
    def run(
        self,
        data_cfg: DataConfig,
        train_cfg: TrainConfig,
        device: Optional[torch.device] = None,
        run_data_prep: bool = True,
        run_training: bool = True,
        run_evaluation: bool = True,
        data_output: Optional[StageOutput] = None,
    ) -> Dict[str, StageOutput]:
        """
        Run the full pipeline or specific stages.
        
        Args:
            data_cfg: Data configuration
            train_cfg: Training configuration
            device: Device to use for training/evaluation
            run_data_prep: Whether to run data preparation stage
            run_training: Whether to run training stage
            run_evaluation: Whether to run evaluation stage
            data_output: Pre-existing data output to skip data prep
            
        Returns:
            Dictionary mapping stage names to their outputs
        """
        # Initialize W&B run (shared across all stages)
        self.wandb_run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            config={**asdict(data_cfg), **asdict(train_cfg)},
        )
        
        # Log environment
        log_environment_to_wandb(self.wandb_run)
        
        outputs = {}
        
        try:
            # Stage 1: Data Preparation
            if run_data_prep:
                data_stage = DataPreparationStage(wandb_run=self.wandb_run)
                data_output = data_stage.run(data_cfg=data_cfg)
                outputs["data_preparation"] = data_output
            elif data_output is None:
                raise ValueError("Must provide data_output if run_data_prep=False")
            else:
                outputs["data_preparation"] = data_output
            
            # Stage 2: Training
            if run_training:
                training_stage = TrainingStage(wandb_run=self.wandb_run)
                training_output = training_stage.run(
                    train_cfg=train_cfg,
                    data_output=data_output,
                    device=device,
                )
                outputs["training"] = training_output
            
            # Stage 3: Evaluation
            if run_evaluation and run_training:
                eval_stage = EvaluationStage(wandb_run=self.wandb_run)
                eval_output = eval_stage.run(
                    data_output=data_output,
                    training_output=training_output,
                    device=device,
                )
                outputs["evaluation"] = eval_output
            
        finally:
            # Finish W&B run
            if self.wandb_run:
                wandb.finish()
        
        return outputs


def run_pipeline(
    data_config_path: Optional[str] = None,
    train_config_path: Optional[str] = None,
    use_wandb: bool = True,
) -> Dict[str, StageOutput]:
    """
    Convenience function to run the full pipeline.
    
    Args:
        data_config_path: Path to data config YAML
        train_config_path: Path to train config YAML
        use_wandb: Whether to use W&B logging
        
    Returns:
        Dictionary of stage outputs
        
    Example:
        >>> from homework_scientalab.pipeline import run_pipeline
        >>> outputs = run_pipeline()
        >>> # Access specific outputs
        >>> model_artifact = outputs["training"].artifacts["model"]
        >>> dataset_path = outputs["data_preparation"].local_paths["dataset"]
    """
    from homework_scientalab.config import load_data_config, load_train_config
    
    # Load configs
    data_cfg = load_data_config(yaml_path=data_config_path)
    train_cfg = load_train_config(yaml_path=train_config_path)
    
    if not use_wandb:
        raise ValueError("Pipeline requires W&B for artifact tracking. Set use_wandb=True")
    
    # Run pipeline
    pipeline = Pipeline(
        project=train_cfg.project,
        entity=train_cfg.entity,
        run_name=train_cfg.run_name,
    )
    
    return pipeline.run(
        data_cfg=data_cfg,
        train_cfg=train_cfg,
    )

