"""
Loss functions for inVAE training.

Includes KL divergence, Negative Binomial likelihood, and independence penalty.
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

# Numerical stability constants
NB_EPSILON = 1e-8          # Small constant for numerical stability in NB likelihood
NB_FALLBACK_LL = -1e10     # Fallback log-likelihood for non-finite values


def independence_penalty_cov(z_i: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
    """
    Independence penalty based on cross-covariance between z_i and z_s.
    
    Computes the squared Frobenius norm of the cross-covariance matrix
    as a surrogate for total correlation.
    
    Args:
        z_i: Biological latent representation (batch_size, z_i_dim)
        z_s: Spurious latent representation (batch_size, z_s_dim)
        
    Returns:
        Scalar penalty value
    """
    zi = z_i - z_i.mean(dim=0, keepdim=True)
    zs = z_s - z_s.mean(dim=0, keepdim=True)
    B = z_i.size(0)
    C = (zi.T @ zs) / (B - 1)
    return (C ** 2).sum()


def gaussian_kl(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
    reduction: str = "sum"
) -> torch.Tensor:
    """
    KL divergence between two diagonal Gaussian distributions.
    
    Computes KL(q || p) where:
    - q = N(mu_q, diag(exp(logvar_q)))
    - p = N(mu_p, diag(exp(logvar_p)))
    
    Args:
        mu_q: Mean of q
        logvar_q: Log variance of q
        mu_p: Mean of p
        logvar_p: Log variance of p
        reduction: How to reduce over batch ("sum", "mean", or "none")
        
    Returns:
        KL divergence (scalar if reduction != "none", else per-sample)
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    term = (logvar_p - logvar_q) + (var_q + (mu_q - mu_p) ** 2) / (var_p + NB_EPSILON) - 1.0
    kl_per_sample = 0.5 * term.sum(dim=-1)
    
    if reduction == "sum":
        return kl_per_sample.sum()
    elif reduction == "mean":
        return kl_per_sample.mean()
    elif reduction == "none":
        return kl_per_sample
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def nb_log_likelihood(
    x: torch.Tensor,
    mu: torch.Tensor,
    alpha: torch.Tensor,
    eps: float = NB_EPSILON,
    reduction: str = "sum"
) -> torch.Tensor:
    """
    Numerically stable Negative Binomial log-likelihood.
    
    Parameterization: x ~ NB(mean=mu, inv_dispersion=alpha)
    where r = 1/alpha is the dispersion parameter.
    
    Args:
        x: Observed counts (batch_size, n_genes)
        mu: Mean parameters (batch_size, n_genes)
        alpha: Inverse dispersion parameters (n_genes,)
        eps: Small constant for numerical stability
        reduction: How to reduce over batch and genes
        
    Returns:
        Log-likelihood (scalar if reduction != "none")
    """
    B, G = x.shape
    # Broadcast alpha to (B,G)
    alpha_bg = alpha.unsqueeze(0).expand(B, G)
    
    # Compute r = 1/alpha with numerical stability
    r = 1.0 / (alpha_bg + eps)
    
    # Stabilize mu to avoid division issues
    mu = torch.clamp(mu, min=eps)
    
    # Compute log terms more stably
    log_theta_mu_eps = torch.log(r + mu + eps)
    
    # More stable NB likelihood following old inVAE implementation
    ll = (
        r * (torch.log(r + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + r)
        - torch.lgamma(r)
        - torch.lgamma(x + 1.0)
    )
    
    ll = torch.where(torch.isfinite(ll), ll, torch.tensor(NB_FALLBACK_LL, device=ll.device, dtype=ll.dtype))
    
    if reduction == "sum":
        return ll.sum()
    elif reduction == "mean":
        return ll.mean()
    elif reduction == "none":
        return ll
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


class InVAELoss(nn.Module):
    """
    Evidence Lower Bound (ELBO) loss for inVAE.
    
    Components:
    - Reconstruction: E_q[log p(x|z)] using Negative Binomial likelihood
    - KL divergence: KL(q(z|x,b,t) || p(z|b,t)) = KL_i + KL_s
    - Independence penalty: Optional penalty on cross-covariance between z_i and z_s
    
    Args:
        beta: Weight for KL divergence (for warmup/annealing)
        lambda_indep: Weight for independence penalty
    """
    def __init__(self, beta: float = 1.0, lambda_indep: float = 0.0):
        super().__init__()
        self.beta = beta
        self.lambda_indep = lambda_indep

    def forward(self, batch: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["x"]  # (B, G)
        mu_x = out["mu_x"]  # (B, G)
        alpha = out["alpha"]  # (G,)
        enc = out["enc"]
        prior = out["prior"]


        recon_ll_per_sample = nb_log_likelihood(x=x, mu=mu_x, alpha=alpha, reduction="none")  # (B, G)
        

        recon_ll_per_sample = recon_ll_per_sample.sum(dim=-1)  # (B,) - sum over genes
        recon_ll = recon_ll_per_sample.mean()  # scalar - mean over batch
        recon_loss = -recon_ll
        

        # KL terms: sum over latent dims, mean over batch
        kl_i_per_sample = gaussian_kl(enc["mu_i"], enc["logvar_i"], prior["mu_i"], prior["logvar_i"], reduction="none")  # (B,)
        kl_s_per_sample = gaussian_kl(enc["mu_s"], enc["logvar_s"], prior["mu_s"], prior["logvar_s"], reduction="none")  # (B,)
        kl_i = kl_i_per_sample.mean()
        kl_s = kl_s_per_sample.mean()
        kl = kl_i + kl_s

        # Independence penalty (TC surrogate)
        indep_pen = torch.tensor(0.0, device=x.device)
        if self.lambda_indep > 0:
            indep_pen = independence_penalty_cov(out["z_i"], out["z_s"])

        # Total loss
        total = recon_loss + self.beta * kl + self.lambda_indep * indep_pen

        return {
            "loss": total,
            "recon_loss": recon_loss.detach(),
            "kl_i": kl_i.detach(),
            "kl_s": kl_s.detach(),
            "indep_pen": indep_pen.detach(),
            "recon_ll": recon_ll.detach(),
        }