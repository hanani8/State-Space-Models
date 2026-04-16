"""
S4 / S4D model for speech classification.

Two SSM backends are supported, selectable via ssm_type:

  's4d'  — Diagonal SSM (S4D, NeurIPS 2022)
             Uses S4DKernel from s4d.py.  Purely diagonal A matrix with
             harmonic initialisation (A_imag = π·[0,…,N/2-1]).
             Numerically simple; no Cauchy kernel needed.

  's4'   — Structured SSM with HiPPO-LegS init (S4, ICLR 2022)
             Uses FFTConv(mode='s4') → SSMKernelDPLR from s4.py.
             Diagonal + Low-Rank (NPLR) A parameterisation; exact HiPPO
             initialisation via the Legendre (scaled) basis.

Both layers share the same external interface:
  input  (B, H, L)   [transposed layout]
  output (B, H, L), state
"""

import sys
import os
import importlib.util
import torch
import torch.nn as nn

# ── path setup ───────────────────────────────────────────────────────────────
_S4_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 's4'))
_S4D_DIR  = os.path.join(_S4_ROOT, 'models', 's4')

# s4d.py internally does:  from src.models.nn import DropoutNd
# That resolves through the s4 repo root.
if _S4_ROOT not in sys.path:
    sys.path.insert(0, _S4_ROOT)
# s4d.py lives here
if _S4D_DIR not in sys.path:
    sys.path.insert(0, _S4D_DIR)

# S4D — minimal diagonal SSM (NeurIPS 2022)
from s4d import S4D  # noqa: E402

# Full S4 — load via importlib to avoid a name collision with the s4/ repo
# directory that may also appear on sys.path.
_s4_spec = importlib.util.spec_from_file_location(
    "_s4_standalone",
    os.path.join(_S4D_DIR, "s4.py"),
)
_s4_mod = importlib.util.module_from_spec(_s4_spec)
sys.modules["_s4_standalone"] = _s4_mod
_s4_spec.loader.exec_module(_s4_mod)
FFTConv = _s4_mod.FFTConv   # wraps SSMKernelDPLR (mode='s4') or SSMKernelDiag (mode='s4d')
# ─────────────────────────────────────────────────────────────────────────────


def _build_ssm_layer(ssm_type, d_model, d_state, dropout, dt_min, dt_max):
    """Return one SSM layer of the requested type."""
    if ssm_type == 's4d':
        # S4D: standalone implementation in s4d.py.
        # d_output == d_model; returns (B, H, L) when transposed=True.
        return S4D(
            d_model=d_model,
            d_state=d_state,
            dropout=dropout,
            transposed=True,
            dt_min=dt_min,
            dt_max=dt_max,
        )
    elif ssm_type == 's4':
        # Full S4: FFTConv with the NPLR/DPLR kernel (HiPPO-LegS default init).
        # channels=1  → d_output == d_model; returns (B, H, L) when transposed=True.
        # lr=0.001    → special per-parameter lr for A/B/C/dt (picked up by
        #               utils.setup_optimizer via the _optim attribute).
        return FFTConv(
            d_model=d_model,
            channels=1,
            activation='gelu',
            transposed=True,
            dropout=dropout,
            mode='s4',
            d_state=d_state,
            dt_min=dt_min,
            dt_max=dt_max,
            lr=0.001,
        )
    else:
        raise ValueError(
            f"Unknown ssm_type '{ssm_type}'. Valid choices: 's4d', 's4'."
        )


class S4Model(nn.Module):
    """
    Speech classifier backed by a stack of S4 or S4D layers.

    Architecture per residual block
    ────────────────────────────────
    [prenorm]  LayerNorm
               ↓  transpose  (B,L,H) → (B,H,L)
               SSM layer   (S4D or S4)
               ↓  transpose  (B,H,L) → (B,L,H)
               Dropout
               + residual
    [postnorm] LayerNorm

    After all blocks: mean-pool over time → MLP classifier.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        d_state: int = 64,
        num_layers: int = 4,
        num_classes: int = 35,
        dropout: float = 0.1,
        pooling: str = 'mean',
        prenorm: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        ssm_type: str = 's4d',
    ):
        """
        Parameters
        ----------
        ssm_type    's4d' – diagonal SSM (NeurIPS 2022, simpler / faster)
                    's4'  – NPLR/DPLR SSM with HiPPO-LegS init (ICLR 2022)
        input_dim   Input feature dimension.
        d_model     Model width H.  Each layer runs H independent SSMs in parallel.
        d_state     SSM state size N.  Controls kernel expressivity.
                    Larger N → richer frequency content, more parameters per layer.
        num_layers  Number of SSM residual blocks.
        num_classes Number of output classes.
        dropout     Dropout rate applied after each SSM layer.
        prenorm     True  → LayerNorm before SSM (recommended for S4-family).
                    False → LayerNorm after residual add.
        dt_min/max  Bounds for the log-uniform dt initialisation.
                    For 16 kHz audio: [0.001, 0.1] covers 1 ms – 100 ms,
                    which spans phoneme-level to word-level timescales.
        """
        super().__init__()

        self.prenorm  = prenorm
        self.pooling  = pooling
        self.ssm_type = ssm_type

        self.encoder = nn.Linear(input_dim, d_model)

        self.ssm_layers = nn.ModuleList([
            _build_ssm_layer(ssm_type, d_model, d_state, dropout, dt_min, dt_max)
            for _ in range(num_layers)
        ])
        self.norms    = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout)   for _ in range(num_layers)])

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, input_dim)

        Returns
        -------
        logits : (B, num_classes)
        """
        x = self.encoder(x)  # (B, L, d_model)

        for layer, norm, drop in zip(self.ssm_layers, self.norms, self.dropouts):
            residual = x

            if self.prenorm:
                x = norm(x)

            # Both S4D and FFTConv (S4) expect (B, H, L) when transposed=True
            x = x.transpose(-1, -2)   # (B, d_model, L)
            x, _ = layer(x)            # (B, d_model, L)
            x = x.transpose(-1, -2)   # (B, L, d_model)

            x = drop(x) + residual

            if not self.prenorm:
                x = norm(x)

        if self.pooling == 'mean':
            x = x.mean(dim=1)  # (B, d_model)
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

        return self.classifier(x)
