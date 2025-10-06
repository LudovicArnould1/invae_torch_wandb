"""
Configuration dataclasses for inVAE training.

All configuration parameters are centralized here for easy management.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class DataConfig:
    """Configuration for data preparation and preprocessing."""
    data_path: str = "data/pancreas.h5ad"
    backup_url: str = "https://www.dropbox.com/s/qj1jlm9w10wmt0u/pancreas.h5ad?dl=1"
    min_genes: int = 200
    min_cells: int = 10
    max_mito_pct: Optional[float] = None
    n_top_genes: Optional[int] = 4000
    batch_key: str = "batch"
    celltype_key: str = "celltype"
    val_size: float = 0.2
    random_state: int = 42
    clip_size_factor_percentile: float = 99.5
    min_cell_counts: int = 100
    max_cell_counts: int = 15000
    normalization_target_sum: float = 1e4
    size_factor_min_clip: float = 1.0


@dataclass
class InVAEConfig:
    """Configuration for inVAE model architecture."""
    x_dim: int
    b_dim: int
    t_dim: int
    z_i_dim: int = 30
    z_s_dim: int = 5
    enc_hidden: Tuple[int, ...] = (128, 128)
    dec_hidden: Tuple[int, ...] = (128, 128)
    prior_hidden: Tuple[int, ...] = (128, 128)
    dropout: float = 0.1
    batchnorm: bool = True
    use_library_size: bool = True


@dataclass
class WarmupSchedule:
    """
    Configuration for beta warmup schedule.
    
    Linear warm-up of beta from beta_start to beta_end over warmup_steps.
    After warmup, beta stays at beta_end.
    """
    beta_start: float = 0.0
    beta_end: float = 1.0
    warmup_steps: int = 10_000

    def value(self, global_step: int) -> float:
        """
        Compute beta value at given global step.
        
        Args:
            global_step: Current training step
            
        Returns:
            Beta value for KL weighting
        """
        if self.warmup_steps <= 0:
            return self.beta_end
        alpha = min(1.0, max(0.0, global_step / float(self.warmup_steps)))
        return (1 - alpha) * self.beta_start + alpha * self.beta_end


@dataclass
class TrainConfig:
    """Configuration for training hyperparameters and settings."""
    
    # Model architecture
    z_i_dim: int = 30
    z_s_dim: int = 5
    enc_hidden: Tuple[int, ...] = (128, 128)
    dec_hidden: Tuple[int, ...] = (128, 128)
    prior_hidden: Tuple[int, ...] = (128, 128)
    dropout: float = 0.1
    batchnorm: bool = True
    use_library_size: bool = True
    
    # Training
    n_epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    amp: bool = True
    
    # Loss
    lambda_indep: float = 0.1
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Warmup schedule (embedded)
    warmup_schedule: WarmupSchedule = field(default_factory=lambda: WarmupSchedule(
        beta_start=0.4,
        beta_end=1.0,
        warmup_steps=250
    ))
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing
    save_dir: str = "checkpoints"
    save_every: int = 50
    
    # Wandb
    project: str = "invae-pancreas"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    artifact_save_every: int = 50  # Save model artifacts to W&B every N epochs

