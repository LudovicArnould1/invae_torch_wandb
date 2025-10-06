"""
Sweep-compatible training script for W&B hyperparameter optimization.

This script is designed to work with W&B Sweeps for automated hyperparameter search.
It accepts sweep parameters from wandb.config and runs training accordingly.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
import wandb

from homework_scientalab.config import (
    load_data_config,
    load_train_config,
)


def sweep_train(
    data_config_path: Optional[str] = None,
    train_config_path: Optional[str] = None,
) -> None:
    """
    Training function for W&B sweeps.
    
    Loads base configuration from YAML files, then overrides with sweep parameters
    from wandb.config. This allows the sweep to control specific hyperparameters
    while keeping other settings fixed.
    
    Args:
        data_config_path: Path to data config YAML (defaults to config/data_config.yaml)
        train_config_path: Path to train config YAML (defaults to config/train_config.yaml)
    """
    # Initialize W&B run (will be controlled by sweep)
    run = wandb.init()
    
    # Load base configurations from YAML
    data_cfg = load_data_config(yaml_path=data_config_path)
    train_cfg = load_train_config(yaml_path=train_config_path)
    
    # Override with sweep parameters
    # wandb.config contains the hyperparameters chosen by the sweep
    sweep_config = wandb.config
    
    # Update training config with sweep parameters
    if "z_i_dim" in sweep_config:
        train_cfg.z_i_dim = sweep_config.z_i_dim
    if "z_s_dim" in sweep_config:
        train_cfg.z_s_dim = sweep_config.z_s_dim
    if "learning_rate" in sweep_config:
        train_cfg.learning_rate = sweep_config.learning_rate
    if "lambda_indep" in sweep_config:
        train_cfg.lambda_indep = sweep_config.lambda_indep
    if "warmup_steps" in sweep_config:
        train_cfg.warmup_schedule.warmup_steps = sweep_config.warmup_steps
    
    # Create unique checkpoint directory for this sweep run
    train_cfg.save_dir = f"checkpoints/sweep_{run.id}"
    train_cfg.run_name = f"sweep_{run.id}"
    
    # Log the final configuration
    print("\n" + "=" * 80)
    print("SWEEP RUN CONFIGURATION")
    print("=" * 80)
    print(f"Run ID: {run.id}")
    print(f"z_i_dim: {train_cfg.z_i_dim}")
    print(f"z_s_dim: {train_cfg.z_s_dim}")
    print(f"learning_rate: {train_cfg.learning_rate:.6f}")
    print(f"lambda_indep: {train_cfg.lambda_indep:.4f}")
    print(f"warmup_steps: {train_cfg.warmup_schedule.warmup_steps}")
    print("=" * 80 + "\n")
    
    # Run training (W&B is already initialized by wandb.init())
    # We pass use_wandb=False to prevent re-initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train model - the train function will log to the already initialized W&B run
    try:
        # We need to use the existing wandb run, so we'll call the core training logic
        # but manage wandb ourselves
        from homework_scientalab.data import prepare_dataloaders
        from homework_scientalab.model import InVAE
        from homework_scientalab.muon_optimizer import get_muon_optimizer
        from homework_scientalab.trainer import InVAETrainer
        from homework_scientalab.monitor_and_setup.reproducibility import (
            set_seed,
            log_environment_to_wandb,
        )
        from homework_scientalab.config import InVAEConfig
        from dataclasses import asdict
        
        # Set seeds
        set_seed(train_cfg.seed, train_cfg.deterministic)
        
        # Log environment
        log_environment_to_wandb(run)
        
        # Prepare data
        train_loader, val_loader, dims = prepare_dataloaders(
            cfg=data_cfg,
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.num_workers,
            pin_memory=train_cfg.pin_memory,
            log_artifacts=False,  # Skip artifact logging in sweeps
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
        print(f"Model initialized with {n_params:,} trainable parameters")
        
        # Setup optimizer
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
        
        # Setup trainer
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
        
        # Create checkpoint directory
        save_dir = Path(train_cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        best_val_loss = float("inf")
        
        for epoch in range(1, train_cfg.n_epochs + 1):
            # Train
            train_metrics = trainer.train_epoch(train_loader)
            
            # Validate
            val_metrics = trainer.eval_epoch(val_loader)
            
            # Print summary every 10 epochs
            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:3d}/{train_cfg.n_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.2f} | "
                    f"Val Loss: {val_metrics['loss']:.2f} | "
                    f"Beta: {train_metrics['beta']:.3f}"
                )
            
            # Log to wandb
            wandb.log(
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
                    "train/lambda_indep": train_metrics["lambda_indep"],
                },
                step=trainer.global_step,
            )
            
            # Track best model
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
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch"] = epoch
        
        print(f"\nSweep run complete! Best validation loss: {best_val_loss:.2f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        # W&B will automatically finish the run when the script exits
        pass


if __name__ == "__main__":
    # When run as part of a sweep, wandb.init() will be called automatically
    # and wandb.config will contain the sweep parameters
    sweep_train()

