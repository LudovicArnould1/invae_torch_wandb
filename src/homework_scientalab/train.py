"""
Training script for inVAE model with Weights & Biases monitoring.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Optional
import torch
import wandb

from homework_scientalab.config import (
    DataConfig,
    InVAEConfig,
    TrainConfig,
    load_data_config,
    load_train_config,
)
from homework_scientalab.data import prepare_dataloaders
from homework_scientalab.model import InVAE
from homework_scientalab.muon_optimizer import get_muon_optimizer
from homework_scientalab.trainer import InVAETrainer
from homework_scientalab.monitor_and_setup.reproducibility import (
    set_seed,
    log_environment_to_wandb,
)
from homework_scientalab.monitor_and_setup.artifacts import log_model_artifact

# Training constants
WANDB_LOG_FREQ = 100  # How often to log gradients to wandb


def train(
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    device: Optional[torch.device] = None,
    use_wandb: bool = True,
) -> InVAE:
    """
    Main training function.
    
    Args:
        train_cfg: Training configuration
        data_cfg: Data preprocessing configuration
        device: Device to train on (defaults to cuda if available)
        use_wandb: Whether to use Weights & Biases logging
        
    Returns:
        Trained InVAE model
    """
    # Set seeds for reproducibility
    print(f"Setting random seed to {train_cfg.seed} (deterministic={train_cfg.deterministic})")
    set_seed(train_cfg.seed, train_cfg.deterministic)
    
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb
    if use_wandb:
        run = wandb.init(
            project=train_cfg.project,
            entity=train_cfg.entity,
            name=train_cfg.run_name,
            config={**asdict(train_cfg), **asdict(data_cfg)},
        )
        
        # Log comprehensive environment info
        log_environment_to_wandb(run)
    
    # Prepare data
    train_loader, val_loader, dims = prepare_dataloaders(
        cfg=data_cfg,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        log_artifacts=use_wandb,  # Log data artifacts if using W&B
    )
    
    # Build model
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
    print(f"\nModel initialized with {n_params:,} trainable parameters")
    
    # Log model architecture to wandb
    if use_wandb:
        wandb.config.update(
            {f"model/{k}": v for k, v in asdict(model_cfg).items()},
            allow_val_change=True,
        )
        wandb.config.update({"model/n_params": n_params}, allow_val_change=True)
        wandb.watch(model, log="gradients", log_freq=WANDB_LOG_FREQ)
    
    # Setup optimizer: Muon for 2D parameters, AdamW for others
    params_2d = [p for p in model.parameters() if p.ndim == 2 and p.requires_grad]
    params_other = [p for p in model.parameters() if p.ndim != 2 and p.requires_grad]
    
    n_params_2d = sum(p.numel() for p in params_2d)
    n_params_other = sum(p.numel() for p in params_other)
    
    print(f"Parameters: {len(params_2d)} 2D matrices ({n_params_2d:,} params) for Muon, "
          f"{len(params_other)} others ({n_params_other:,} params) for AdamW")
    
    optimizer = get_muon_optimizer(
        params_2d,
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    
    # If there are non-2D parameters, use a separate AdamW for them
    if params_other:
        optimizer_other = torch.optim.AdamW(
            params_other,
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
    
    # Log optimizer info to wandb
    if use_wandb:
        wandb.config.update(
            {
                "optimizer/type": "Muon+AdamW",
                "optimizer/muon_params": n_params_2d,
                "optimizer/adamw_params": n_params_other,
            },
            allow_val_change=True,
        )
    
    # Setup trainer
    trainer = InVAETrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        beta_schedule=train_cfg.warmup_schedule,
        lambda_indep=train_cfg.lambda_indep,
        grad_clip=train_cfg.grad_clip,
        amp=train_cfg.amp,
        optimizer_other=optimizer_other if params_other else None,
    )
    
    # Create checkpoint directory
    save_dir = Path(train_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    best_val_loss = float("inf")
    
    for epoch in range(1, train_cfg.n_epochs + 1):
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Validate
        val_metrics = trainer.eval_epoch(val_loader)
        
        # Print summary
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}/{train_cfg.n_epochs} | "
                f"Train Loss: {train_metrics['loss']:.2f} "
                f"(recon: {train_metrics['recon']:.2f}, "
                f"kl_i: {train_metrics['kl_i']:.2f}, "
                f"kl_s: {train_metrics['kl_s']:.2f}) | "
                f"Val Loss: {val_metrics['loss']:.2f} | "
                f"Beta: {train_metrics['beta']:.3f}"
            )
        
        # Log to wandb
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/recon_loss": train_metrics["recon"],
                    "train/kl_i": train_metrics["kl_i"],
                    "train/kl_s": train_metrics["kl_s"],
                    "train/recon_ll": train_metrics["recon_ll"],
                    "train/indep_penalty": train_metrics["indep"],
                    "val/loss": val_metrics["loss"],
                    "val/recon_loss": val_metrics["recon"],
                    "val/kl_i": val_metrics["kl_i"],
                    "val/kl_s": val_metrics["kl_s"],
                    "val/recon_ll": val_metrics["recon_ll"],
                    "val/indep_penalty": val_metrics["indep"],
                    "train/beta": train_metrics["beta"],
                    "train/lambda_indep": train_metrics["lambda_indep"],
                },
                step=trainer.global_step,
            )
        
        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint_path = save_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "model_config": asdict(model_cfg),
                },
                checkpoint_path,
            )
            if use_wandb:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch"] = epoch
                
                # Log best model as W&B artifact
                log_model_artifact(
                    str(checkpoint_path),
                    artifact_name="invae_model",
                    model_config=asdict(model_cfg),
                    metrics={
                        "val_loss": float(best_val_loss),
                        "epoch": epoch,
                        "train_loss": train_metrics["loss"],
                        "val_recon_loss": val_metrics["recon"],
                        "val_kl_i": val_metrics["kl_i"],
                        "val_kl_s": val_metrics["kl_s"],
                    },
                    description=f"Best model at epoch {epoch} with val_loss={best_val_loss:.2f}",
                    aliases=["best", "latest"],
                )
        
        # Periodic checkpoint
        if epoch % train_cfg.save_every == 0:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "model_config": asdict(model_cfg),
                },
                checkpoint_path,
            )
            
            # Log periodic checkpoint as W&B artifact
            if use_wandb:
                log_model_artifact(
                    str(checkpoint_path),
                    artifact_name="invae_model_checkpoints",
                    model_config=asdict(model_cfg),
                    metrics={
                        "val_loss": float(val_metrics["loss"]),
                        "epoch": epoch,
                    },
                    description=f"Checkpoint at epoch {epoch}",
                    aliases=[f"epoch_{epoch}"],
                )
    
    print("\n" + "=" * 80)
    print(f"Training complete! Best validation loss: {best_val_loss:.2f}")
    print(f"Best model saved to: {save_dir / 'best_model.pt'}")
    print("=" * 80 + "\n")
    
    if use_wandb:
        wandb.finish()
    return model


def main(
    data_config_path: Optional[str] = None,
    train_config_path: Optional[str] = None,
    use_wandb: bool = True,
) -> None:
    """
    Main entry point for training script.
    
    Loads configuration from YAML files by default, with option to override.
    
    Args:
        data_config_path: Path to data config YAML (defaults to config/data_config.yaml)
        train_config_path: Path to train config YAML (defaults to config/train_config.yaml)
        use_wandb: Whether to use Weights & Biases logging
        
    Example:
        >>> # Use default configs
        >>> main()
        
        >>> # Use custom configs
        >>> main(data_config_path="my_data.yaml", train_config_path="my_train.yaml")
        
        >>> # Override specific parameters
        >>> data_cfg = load_data_config(overrides={"n_top_genes": 2000})
        >>> train_cfg = load_train_config(overrides={"batch_size": 128})
        >>> train(train_cfg, data_cfg, use_wandb=True)
    """
    # Load configuration from YAML files
    print("Loading configuration from YAML files...")
    data_cfg = load_data_config(yaml_path=data_config_path)
    train_cfg = load_train_config(yaml_path=train_config_path)
    
    print(f"\nData Config:")
    print(f"  Data path: {data_cfg.data_path}")
    print(f"  HVGs: {data_cfg.n_top_genes}")
    print(f"  Val size: {data_cfg.val_size}")
    
    print(f"\nTraining Config:")
    print(f"  Epochs: {train_cfg.n_epochs}")
    print(f"  Batch size: {train_cfg.batch_size}")
    print(f"  Learning rate: {train_cfg.learning_rate}")
    print(f"  Latent dims: z_i={train_cfg.z_i_dim}, z_s={train_cfg.z_s_dim}")
    print(f"  Warmup: {train_cfg.warmup_schedule.warmup_steps} steps")
    
    # Train
    model = train(
        train_cfg,
        data_cfg,
        use_wandb=use_wandb,
    )

if __name__ == "__main__":
    main()

