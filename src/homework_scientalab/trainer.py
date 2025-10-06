"""
Training utilities for inVAE model.
"""
from __future__ import annotations
from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader

from homework_scientalab.config import WarmupSchedule
from homework_scientalab.model import InVAE
from homework_scientalab.losses import InVAELoss


class InVAETrainer:
    """
    Trainer for inVAE model with support for AMP and gradient clipping.
    
    Args:
        model: InVAE model instance
        optimizer: PyTorch optimizer
        device: Device to train on
        beta_schedule: Beta warmup schedule for KL annealing
        lambda_indep: Weight for independence penalty
        grad_clip: Gradient clipping value (None to disable)
        amp: Whether to use automatic mixed precision
        optimizer_other: Optional second optimizer for non-2D parameters
    """
    def __init__(
        self,
        model: InVAE,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        beta_schedule: Optional[WarmupSchedule] = None,
        lambda_indep: float = 0.0,
        grad_clip: Optional[float] = 1.0,
        amp: bool = True,
        optimizer_other: Optional[torch.optim.Optimizer] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.optimizer_other = optimizer_other
        self.device = device
        self.beta_schedule = beta_schedule or WarmupSchedule()
        self.lambda_indep = lambda_indep
        self.grad_clip = grad_clip
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)

        # initialize loss (beta gets updated each step)
        self.loss_obj = InVAELoss(beta=1.0, lambda_indep=lambda_indep)

        # running step counter
        self.global_step = 0

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        moved = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device)
            else:
                moved[k] = v
        return moved

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        meters = {"loss": 0.0, "recon": 0.0, "kl_i": 0.0, "kl_s": 0.0, "indep": 0.0, "recon_ll": 0.0}

        for batch in loader:
            batch = self._move_batch(batch)
            size_factor = batch.get("size_factor", None)

            beta = self.beta_schedule.value(self.global_step)
            self.loss_obj.beta = beta

            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                out = self.model(
                    x=batch["x"],
                    b=batch["b"],
                    t=batch["t"],
                    size_factor=size_factor,
                    sample_z=True,
                )
                losses = self.loss_obj(batch, out)
                loss = losses["loss"]

            self.optimizer.zero_grad(set_to_none=True)
            if self.optimizer_other is not None:
                self.optimizer_other.zero_grad(set_to_none=True)
            
            self.scaler.scale(loss).backward()
            
            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                if self.optimizer_other is not None:
                    self.scaler.unscale_(self.optimizer_other)

            self.scaler.step(self.optimizer)
            if self.optimizer_other is not None:
                self.scaler.step(self.optimizer_other)
            self.scaler.update()

            meters["loss"] += losses["loss"].item()
            meters["recon"] += losses["recon_loss"].item()
            meters["kl_i"] += losses["kl_i"].item()
            meters["kl_s"] += losses["kl_s"].item()
            meters["indep"] += losses["indep_pen"].item()
            meters["recon_ll"] += losses["recon_ll"].item()

            self.global_step += 1

        for k in meters:
            meters[k] /= max(1, len(loader))
        meters["beta"] = self.loss_obj.beta
        meters["lambda_indep"] = self.lambda_indep
        return meters

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        meters = {"loss": 0.0, "recon": 0.0, "kl_i": 0.0, "kl_s": 0.0, "indep": 0.0, "recon_ll": 0.0}
        for batch in loader:
            batch = self._move_batch(batch)
            size_factor = batch.get("size_factor", None)
            out = self.model(
                x=batch["x"],
                b=batch["b"],
                t=batch["t"],
                size_factor=size_factor,
                sample_z=False,  # use posterior mean
            )
            losses = self.loss_obj(batch, out)
            meters["loss"] += losses["loss"].item()
            meters["recon"] += losses["recon_loss"].item()
            meters["kl_i"] += losses["kl_i"].item()
            meters["kl_s"] += losses["kl_s"].item()
            meters["indep"] += losses["indep_pen"].item()
            meters["recon_ll"] += losses["recon_ll"].item()

        for k in meters:
            meters[k] /= max(1, len(loader))
        meters["beta"] = self.loss_obj.beta
        meters["lambda_indep"] = self.lambda_indep
        return meters
