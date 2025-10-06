"""Toy implementation of the Muon optimizer.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _adjust_lr(
    lr: float, adjust_lr_fn: Optional[str], param_shape: torch.Size
) -> float:
    """Adjust learning rate per Muon recommendations.

    Args:
        lr: Base learning rate.
        adjust_lr_fn: One of None, "original", or "match_rms_adamw".
        param_shape: Parameter shape; first two dims used.

    Returns:
        Adjusted learning rate.
    """
    if len(param_shape) < 2:
        return lr
    a_dim, b_dim = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1.0, float(a_dim) / float(b_dim)))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(float(max(a_dim, b_dim)))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


class _LocalMuon(torch.optim.Optimizer):
    """Minimal Muon optimizer fallback.

    Differences vs reference:
    - Implements only single-tensor path (no foreach).
    - Requires all optimized params to be 2D matrices.
    - Uses decoupled weight decay like AdamW.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: Optional[str] = None,
    ) -> None:
        if lr < 0:
            raise ValueError("Learning rate must be non-negative")
        if momentum < 0:
            raise ValueError("Momentum must be non-negative")
        if weight_decay < 0:
            raise ValueError("Weight decay must be non-negative")
        if adjust_lr_fn is not None and adjust_lr_fn not in (
            "original",
            "match_rms_adamw",
        ):
            raise ValueError("Unsupported adjust_lr_fn")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)

        # Validate shapes once
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only supports 2D parameters; got {tuple(p.shape)}"
                    )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            weight_decay: float = group["weight_decay"]
            momentum: float = group["momentum"]
            nesterov: bool = group["nesterov"]
            ns_coefficients: Tuple[float, float, float] = group["ns_coefficients"]
            eps: float = group["eps"]
            ns_steps: int = group["ns_steps"]
            adjust_lr_fn: Optional[str] = group["adjust_lr_fn"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                if grad.ndim != 2:
                    raise ValueError("Param gradient must be a 2D matrix")

                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(grad, memory_format=torch.preserve_format)
                    state["momentum_buffer"] = buf

                # Momentum + optional Nesterov
                buf.lerp_(grad, 1 - momentum)
                update = grad.lerp(buf, momentum) if nesterov else buf

                # Newton–Schulz orthogonalization (quintic polynomial iteration)
                update = self._zeropower_via_newtonschulz(update, ns_coefficients, ns_steps, eps)

                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, p.shape)

                # Decoupled weight decay then Muon update
                p.mul_(1 - lr * weight_decay)
                p.add_(update, alpha=-adjusted_lr)

        return loss

    @staticmethod
    def _zeropower_via_newtonschulz(
        grad: Tensor, ns_coefficients: Tuple[float, float, float], ns_steps: int, eps: float
    ) -> Tensor:
        if ns_steps >= 100:
            raise ValueError("ns_steps must be < 100")
        if grad.ndim != 2:
            raise ValueError("Input gradient must be 2D")
        if len(ns_coefficients) != 3:
            raise ValueError("ns_coefficients must have 3 values")

        a, b, c = ns_coefficients
        ortho_grad = grad.bfloat16()
        if grad.size(0) > grad.size(1):
            ortho_grad = ortho_grad.T
        # Normalize spectral norm <= 1
        ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
        for _ in range(ns_steps):
            gram = ortho_grad @ ortho_grad.T
            gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
            ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
        if grad.size(0) > grad.size(1):
            ortho_grad = ortho_grad.T
        return ortho_grad.to(dtype=grad.dtype)


def get_muon_optimizer(params: Iterable[Tensor], **kwargs) -> torch.optim.Optimizer:
    """Return torch Muon if available, else local fallback.

    Args:
        params: Parameters to optimize.
        **kwargs: Muon keyword arguments.

    Returns:
        An optimizer instance implementing Muon behavior.
    """
    try:
        # Attempt to import Muon from torch (newer nightlies may include it)
        from torch.optim import Muon as TorchMuon  # type: ignore

        logging.info("Using torch.optim.Muon")
        return TorchMuon(params, **kwargs)  # type: ignore[arg-type]
    except Exception:
        logging.info("torch.optim.Muon not found; using local fallback implementation")
        return _LocalMuon(params, **kwargs)

