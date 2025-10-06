"""
Neural network modules for the inVAE model.

Includes encoder, decoder, prior networks, and supporting components.
"""
from __future__ import annotations
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn

from homework_scientalab.config import InVAEConfig


# Numerical stability constants
THETA_CLAMP_MIN = -5.0  # Min value for decoder raw theta before exp
THETA_CLAMP_MAX = 5.0   # Max value for decoder raw theta before exp
ALPHA_EPSILON = 1e-6    # Small constant added to alpha for numerical stability


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable architecture.
    
    Args:
        in_dim: Input dimension
        hidden_dims: Tuple of hidden layer dimensions
        out_dim: Output dimension (if None, no output head is added)
        activation: Activation function (default: ReLU)
        dropout: Dropout probability
        batchnorm: Whether to use batch normalization
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128),
        out_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.1,
        batchnorm: bool = True,
    ):
        super().__init__()
        dims = (in_dim,) + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            if batchnorm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(activation)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.body = nn.Sequential(*layers)
        self.head = None
        if out_dim is not None:
            self.head = nn.Linear(dims[-1], out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        if self.head is not None:
            return self.head(h)
        return h



class GaussianHead(nn.Module):
    """
    Output layer for diagonal Gaussian distribution parameters.
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension (dimension of the Gaussian variable)
        
    Returns:
        Tuple of (mu, logvar) tensors with shape (..., out_dim)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mu = nn.Linear(in_dim, out_dim, bias=False)
        self.logvar = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mu(h), self.logvar(h)


class Encoder(nn.Module):
    """
    Posterior network q(z | x, b, t) = N(mu, exp(logvar)).
    
    Outputs joint Gaussian over z = [z_i, z_s], which is then split into
    biological (z_i) and spurious (z_s) components.
    
    Args:
        x_dim: Gene expression dimension
        b_dim: Biological covariate dimension
        t_dim: Technical covariate dimension
        z_i_dim: Biological latent dimension
        z_s_dim: Spurious latent dimension
        hidden: Hidden layer dimensions
        dropout: Dropout probability
        batchnorm: Whether to use batch normalization
    """
    def __init__(
        self,
        x_dim: int,
        b_dim: int,
        t_dim: int,
        z_i_dim: int,
        z_s_dim: int,
        hidden: Tuple[int, ...] = (128, 128),
        dropout: float = 0.1,
        batchnorm: bool = True,
    ):
        super().__init__()
        in_dim = x_dim + b_dim + t_dim
        self.z_i_dim = z_i_dim
        self.z_s_dim = z_s_dim
        self.backbone = MLP(in_dim, hidden, out_dim=None, dropout=dropout, batchnorm=batchnorm)
        self.head = GaussianHead(hidden[-1], z_i_dim + z_s_dim)

    def forward(self, x: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Apply log transform to raw counts
        x_log = torch.log(x + 1)
        h = self.backbone(torch.cat([x_log, b, t], dim=-1))
        mu, logvar = self.head(h)
        mu_i, mu_s = mu[..., :self.z_i_dim], mu[..., self.z_i_dim:]
        lv_i, lv_s = logvar[..., :self.z_i_dim], logvar[..., self.z_i_dim:]
        return {
            "mu": mu, "logvar": logvar,
            "mu_i": mu_i, "logvar_i": lv_i,
            "mu_s": mu_s, "logvar_s": lv_s,
        }


class PriorNet(nn.Module):
    """
    Conditional prior network p(z | covariates) = N(mu, exp(logvar)).
    
    Args:
        u_dim: Covariate dimension
        z_dim: Latent dimension
        hidden: Hidden layer dimensions
        dropout: Dropout probability
        batchnorm: Whether to use batch normalization
    """
    def __init__(
        self,
        u_dim: int,
        z_dim: int,
        hidden: Tuple[int, ...] = (128, 128),
        dropout: float = 0.1,
        batchnorm: bool = True,
    ):
        super().__init__()
        self.backbone = MLP(u_dim, hidden, out_dim=None, dropout=dropout, batchnorm=batchnorm)
        self.head = GaussianHead(hidden[-1], z_dim)

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(u)
        mu, logvar = self.head(h)
        return mu, logvar


class DecoderNB(nn.Module):
    """
    Negative Binomial decoder for count data.
    
    Architecture:
    1. MLP to hidden representation
    2. Softmax layer to get gene frequency distribution (sums to 1)
    3. Scale by library size to get mean counts
    
    Args:
        z_i_dim: Biological latent dimension
        z_s_dim: Spurious latent dimension
        g_dim: Number of genes
        hidden: Hidden layer dimensions
        dropout: Dropout probability
        batchnorm: Whether to use batch normalization
        use_library_size: Whether to scale output by library size
        
    Parameterization:
        x_g ~ NB(mean=mu_g, inv_dispersion=alpha_g)
    """
    def __init__(
        self,
        z_i_dim: int,
        z_s_dim: int,
        g_dim: int,
        hidden: Tuple[int, ...] = (128, 128),
        dropout: float = 0.1,
        batchnorm: bool = True,
        use_library_size: bool = True,
    ):
        super().__init__()
        self.g_dim = g_dim
        self.use_library_size = use_library_size

        in_dim = z_i_dim + z_s_dim
        # MLP to hidden
        self.decoder_raw_mean = MLP(in_dim, hidden, out_dim=None, dropout=dropout, batchnorm=batchnorm)
        # Softmax layer to get frequencies
        self.decoder_freq = nn.Sequential(
            nn.Linear(hidden[-1], g_dim, bias=False),
            nn.Softmax(dim=-1),
        )
        # gene-wise inverse-dispersion alpha_g (>0). 
        self.decoder_raw_theta = nn.Parameter(torch.randn(g_dim))

    def forward(
        self,
        z_i: torch.Tensor,
        z_s: torch.Tensor,
        size_factor: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        h = torch.cat([z_i, z_s], dim=-1)
        decoder_raw = self.decoder_raw_mean(h)
        # Get frequencies (sum to 1 per cell)
        decoder_freq = self.decoder_freq(decoder_raw)  # (B, G)
        
        if self.use_library_size:
            if size_factor is None:
                raise ValueError("DecoderNB expected size_factor; set use_library_size=False to disable.")
            if size_factor.dim() == 1:
                size_factor = size_factor.unsqueeze(-1)
            mu = decoder_freq * size_factor
        else:
            mu = decoder_freq
        
        theta_clamped = torch.clamp(self.decoder_raw_theta, min=THETA_CLAMP_MIN, max=THETA_CLAMP_MAX)
        alpha = torch.exp(theta_clamped) + ALPHA_EPSILON
        
        return {"mu": mu, "alpha": alpha}


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick: z = mu + sigma * epsilon, where epsilon ~ N(0, I).
    
    Args:
        mu: Mean of the Gaussian distribution
        logvar: Log variance of the Gaussian distribution
        
    Returns:
        Sampled latent variable
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class InVAE(nn.Module):
    """
    Complete inVAE model with encoder, priors, and decoder.
    
    Components:
    - Encoder: q(z | x, b, t)
    - Priors: p(z_i | b), p(z_s | t)
    - Decoder: p(x | z) with Negative Binomial likelihood
    
    Args:
        cfg: InVAEConfig with model architecture parameters
    """
    def __init__(self, cfg: InVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(
            x_dim=cfg.x_dim,
            b_dim=cfg.b_dim,
            t_dim=cfg.t_dim,
            z_i_dim=cfg.z_i_dim,
            z_s_dim=cfg.z_s_dim,
            hidden=cfg.enc_hidden,
            dropout=cfg.dropout,
            batchnorm=cfg.batchnorm,
        )
        self.prior_i = PriorNet(
            u_dim=cfg.b_dim,
            z_dim=cfg.z_i_dim,
            hidden=cfg.prior_hidden,
            dropout=cfg.dropout,
            batchnorm=cfg.batchnorm,
        )
        self.prior_s = PriorNet(
            u_dim=cfg.t_dim,
            z_dim=cfg.z_s_dim,
            hidden=cfg.prior_hidden,
            dropout=cfg.dropout,
            batchnorm=cfg.batchnorm,
        )
        self.decoder = DecoderNB(
            z_i_dim=cfg.z_i_dim,
            z_s_dim=cfg.z_s_dim,
            g_dim=cfg.x_dim,
            hidden=cfg.dec_hidden,
            dropout=cfg.dropout,
            batchnorm=cfg.batchnorm,
            use_library_size=cfg.use_library_size,
        )

    @torch.inference_mode(False)
    def forward(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        t: torch.Tensor,
        size_factor: Optional[torch.Tensor] = None,
        sample_z: bool = True,
    ) -> Dict[str, torch.Tensor]:
        
        # Encoder posterior
        enc = self.encoder(x, b, t)
        
        # Priors
        mu_pi, lv_pi = self.prior_i(b)
        mu_ps, lv_ps = self.prior_s(t)

        # Sample z
        if sample_z:
            z_i = reparameterize(enc["mu_i"], enc["logvar_i"])
            z_s = reparameterize(enc["mu_s"], enc["logvar_s"])
        else:
            z_i, z_s = enc["mu_i"], enc["mu_s"]
        # Decoder
        dec = self.decoder(z_i, z_s, size_factor=size_factor)

        return {
            "mu_x": dec["mu"], "alpha": dec["alpha"],
            "z_i": z_i, "z_s": z_s,
            "enc": enc,
            "prior": {"mu_i": mu_pi, "logvar_i": lv_pi, "mu_s": mu_ps, "logvar_s": lv_ps},
        }
