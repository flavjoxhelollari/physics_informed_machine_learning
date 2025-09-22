"""
models.py
=========

Canonical neural building-blocks for the *Pendulum V-JEPA* study
----------------------------------------------------------------

This module is **self-contained** — no other project files are needed to
instantiate the classes — and can therefore be imported from any research
notebook or downstream library.

Public classes
--------------

PatchEmbed
    Small Conv-stem ⇒ patch tokens  (images → sequence).
TransformerEncoder
    Thin wrapper around `torch.nn.TransformerEncoderLayer`.
MaskingStrategy
    Generates the *context* and *target* masks used by **V-JEPA**.
VJEPA
    Vision-only Joint Embedding Predictive Architecture backbone.
LNN
    Minimal Lagrangian Neural Network (scalar *L(q,v)*).
HNN
    Minimal Hamiltonian Neural Network  in 2-D canonical form.

Every class exposes **only** pure-Python / PyTorch code and has no
side-effects such as file I/O; this makes unit-testing and reuse trivial.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pick one device here; sub-modules inherit it.
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1. Patch-tokeniser ————————————————————————————————————————————
# ---------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """
    Convolutional patch tokeniser (a *very* light ViT stem).

    Parameters
    ----------
    img_size : int, default 64
        Input images are assumed square **img_size × img_size**.
    patch_size : int, default 8
        Square patch edge length (⇒ stride & kernel_size of the conv).
    in_chans : int, default 3
        Number of input channels (RGB = 3).
    embed_dim : int, default 384
        Output channel dimension per patch token.

    Returns
    -------
    torch.Tensor
        Shape **(B, N_patches, embed_dim)**.
    """
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        # Total number of non-overlapping patches
        self.n_patches: int = (img_size // patch_size) ** 2

        # Single conv does the unfolding + linear projection
        self.proj: nn.Conv2d = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Shapes
        ------
        x : (B, C, H, W)  
        out : (B, N_patches, embed_dim)
        """
        x = self.proj(x)               # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2)               # (B, embed_dim, N_patches)
        x = x.transpose(1, 2)          # (B, N_patches, embed_dim)
        return x


# ---------------------------------------------------------------------
# 2. Tiny ViT encoder ————————————————————————————————————————————
# ---------------------------------------------------------------------
class TransformerEncoder(nn.Module):
    """
    Very thin wrapper around *depth* identical
    `nn.TransformerEncoderLayer` blocks (no CLS token, batch_first).

    All kwargs map 1-to-1 to `nn.TransformerEncoderLayer`.
    """
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )

    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.layers:
            x = blk(x)
        return x


# ---------------------------------------------------------------------
# 3. Mask generators ————————————————————————————————————————————
# ---------------------------------------------------------------------
class MaskingStrategy:
    """
    Re-implements the masking schedule from the original Meta-AI
    **V-JEPA** code-base.

    Two flavours
    ------------

    • `block_masking(B)`  – contiguous *context* mask (coarse blocks).  
    • `random_masking(B)` – iid Bernoulli  *target* mask.

    Returned masks are `torch.bool` with **True = masked**.
    """

    def __init__(
        self,
        num_patches: int,
        mask_ratio: float = 0.75,
        block_size: int = 4,
    ) -> None:
        self.N = num_patches  # total patch tokens
        self.r = mask_ratio   # global mask ratio
        self.bs = block_size  # edge length of a square mask block

    # -----------------------------------------------------------------
    def random_masking(self, B: int) -> torch.BoolTensor:
        """
        Randomly *keeps* ⌊N·(1 − mask_ratio)⌋ patches per sample.
        """
        n_keep = int(self.N * (1.0 - self.r))

        # Per-sample random scores → sort → keep n_keep smallest
        scores = torch.rand(B, self.N, device=device)
        ids_sorted = torch.argsort(scores, dim=1)           # low→high
        ids_restore = torch.argsort(ids_sorted, dim=1)      # undo sort

        mask = torch.ones(B, self.N, device=device)         # start all-masked
        mask[:, :n_keep] = 0                                # un-mask first keep
        mask = torch.gather(mask, 1, ids_restore)           # restore order
        return mask.bool()

    # -----------------------------------------------------------------
    def block_masking(self, B: int) -> torch.BoolTensor:
        """
        Block-wise masking: pick `n_blk` random squares of size
        `block_size × block_size` on the patch grid.
        """
        grid = int(math.sqrt(self.N))                       # patches per side
        masks = torch.zeros(B, self.N, device=device)
        n_blk = int((self.N * self.r) / (self.bs**2))       # how many blocks?

        for b in range(B):
            for _ in range(n_blk):
                # uniformly sample top-left corner
                h0 = np.random.randint(0, grid - self.bs + 1)
                w0 = np.random.randint(0, grid - self.bs + 1)
                # paint the square
                for dh in range(self.bs):
                    for dw in range(self.bs):
                        idx = (h0 + dh) * grid + (w0 + dw)
                        masks[b, idx] = 1
        return masks.bool()


# ---------------------------------------------------------------------
# 4. Vision-only JEPA backbone ————————————————————————————————
# ---------------------------------------------------------------------
class VJEPA(nn.Module):
    """
    Vision-only Joint-Embedding Predictive Architecture.

    The class exposes three call-paths

    * `forward(img)`             – classic JEPA training pass  
    * `forward_context(img, m)`  – visible-patch encoder (no grad stops)  
    * `forward_target(img, m)`   – masked-patch  encoder (no grad stops)

    Notes
    -----
    • All tensors are kept *patch-first* (no `[CLS]` token).  
    • The predictor head is a 2-layer MLP (`embed → pred_dim → embed`).
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        prediction_head_dim: int = 192,
    ) -> None:
        super().__init__()

        # --- patch tokenizer & learnable sin-cos pos-enc -----------
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.normal_(self.pos_embed, std=0.02)

        # --- Siamese encoders --------------------------------------
        self.context_encoder = TransformerEncoder(
            embed_dim, depth, num_heads, mlp_ratio
        )
        self.target_encoder = TransformerEncoder(
            embed_dim, depth, num_heads, mlp_ratio
        )

        # --- simple predictor MLP ----------------------------------
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, prediction_head_dim),
            nn.GELU(),
            nn.Linear(prediction_head_dim, embed_dim),
        )

        # --- masking engine ----------------------------------------
        self.masking = MaskingStrategy(self.num_patches)

    # -----------------------------------------------------------------
    # helpers (used by downstream latent analysis)
    # -----------------------------------------------------------------
    def forward_context(
        self, imgs: torch.Tensor, context_mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Encode *visible* patches (mask = False).

        Shapes
        ------
        imgs : (B, 3, H, W)  
        context_mask : (B, N) bool  (True = masked)  
        returns      : (B, N, D)
        """
        x = self.patch_embed(imgs) + self.pos_embed            # add positional
        B, N, D = x.shape
        keep = ~context_mask                                   # invert mask
        # pad to full length so Transformer sees a fixed sequence
        x_vis = torch.stack(
            [F.pad(x[i][keep[i]], (0, 0, 0, N - keep[i].sum())) for i in range(B)]
        )
        return self.context_encoder(x_vis)

    def forward_target(
        self, imgs: torch.Tensor, target_mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Encode *masked* patches (mask = True).
        """
        x = self.patch_embed(imgs) + self.pos_embed
        B, N, D = x.shape
        x_msk = torch.stack(
            [F.pad(x[i][target_mask[i]], (0, 0, 0, N - target_mask[i].sum())) for i in range(B)]
        )
        return self.target_encoder(x_msk)

    # -----------------------------------------------------------------
    # main JEPA training forward
    # -----------------------------------------------------------------
    def forward(
        self, imgs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        imgs : torch.Tensor
            Mini-batch of RGB images **(B, 3, H, W)**.

        Returns
        -------
        loss : torch.Tensor, scalar
            MSE between predictor output and target embeddings.
        pred : torch.Tensor
            Predictor output **(B, N, D)**.
        target : torch.Tensor
            Stop-gradient target embeddings **(B, N, D)**.
        """
        B = imgs.size(0)

        # build disjoint context / target masks
        ctx_mask = self.masking.block_masking(B)               # (B,N)
        tgt_mask = self.masking.random_masking(B) & ~ctx_mask  # ensure disjoint

        ctx_emb = self.forward_context(imgs, ctx_mask)
        with torch.no_grad():                                  # stop-grad
            tgt_emb = self.forward_target(imgs, tgt_mask)

        pred = self.predictor(ctx_emb)                         # MLP head
        loss = F.mse_loss(pred, tgt_emb)
        return loss, pred, tgt_emb


# ---------------------------------------------------------------------
# 5. Lagrangian Neural Network ————————————————————————————————
# ---------------------------------------------------------------------
class LNN(nn.Module):
    """
    **Minimal** Lagrangian NN (Cranmer *et al.* 2020).

    The network learns a scalar field **L(q, v)**.  For convenience a
    helper method `lagrangian_residual` computes the Euler-Lagrange
    residual MSE over a mini-batch trajectory.

    Notes
    -----
    *Input convention* for all research code in this repo: **d = 1**
    (simple pendulum ⇒ q = θ, v = ω).  Nonetheless the implementation
    supports *any d* w/out modification.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of **(q, v)** — for a single pendulum this is 2.
        hidden_dim : int,  default 256
            Size of the two hidden fully-connected layers (tanh activations).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    # -----------------------------------------------------------------
    def forward(self, qv: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        qv : torch.Tensor
            Concatenated (q, v) of shape **(…, 2d)**.

        Returns
        -------
        torch.Tensor
            Scalar **L** for each sample (same leading dims as input).
        """
        return self.net(qv)

    # -----------------------------------------------------------------
    def lagrangian_residual(
        self,
        q : torch.Tensor,      # (B,T,d)
        v : torch.Tensor,      # (B,T,d)
        dt: float = 0.01,
        *,                     # ← kw-only from here
        per_time_step: bool = False   # NEW
    ) -> torch.Tensor:
        """
        Euler–Lagrange residual.

        Parameters
        ----------
        per_time_step
            *False* (default) → return scalar mean over (B,T-1,d)  
            *True*            → return tensor (B,T-1)   (mean over d)
        """
        B, T, d = q.shape
        z  = torch.cat([q, v], -1).reshape(B*T, -1).requires_grad_(True)
        L  = self.forward(z).sum()

        dLd = torch.autograd.grad(L, z, create_graph=True)[0]
        dLdq, dLdv = dLd.split(d, -1)
        dLdq = dLdq.view(B, T, d)
        dLdv = dLdv.view(B, T, d)

        d_dt_dLdv = (dLdv[:, 1:] - dLdv[:, :-1]) / dt        # (B,T-1,d)
        res       = d_dt_dLdv - dLdq[:, :-1]                 # Euler–Lagrange

        if per_time_step:                                   # curve wanted
            return res.pow(2).mean(-1)                      # (B,T-1)
        return res.pow(2).mean()  


# ---------------------------------------------------------------------
# 6. Hamiltonian Neural Network ————————————————————————————————
# ---------------------------------------------------------------------
class HNN(nn.Module):
    """
    Minimal 2-D Hamiltonian Neural Network (canonical coordinates).

    The network outputs **(F₁, F₂)**.  The time-derivative is recovered
    via the fixed symplectic matrix ``J = [[0, 1], [−1, 0]]``:

    ``(θ̇, ω̇) = ∂F₂/∂(θ, ω) · Jᵀ``.
    """

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),   # outputs (F₁, F₂)
        )
        # Fixed symplectic matrix stored as buffer so it moves w/ `.to()`
        self.register_buffer("J", torch.tensor([[0.0, 1.0], [-1.0, 0.0]]))

    # -----------------------------------------------------------------
    def time_derivative(self, qp: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        qp : torch.Tensor,  shape **(B, 2)**  
            *Canonical* state `[θ, ω]`.

        Returns
        -------
        torch.Tensor
            Canonical vector field **(θ̇, ω̇)**, same shape as input.
        """
        F1, F2 = self.net(qp).split(1, 1)                  # (B, 1) each
        # `create_graph=True` keeps higher-order grads for HNN loss
        dF2 = torch.autograd.grad(F2.sum(), qp, create_graph=True)[0]
        return dF2 @ self.J.T                              # matrix product

class ThetaHead(nn.Module):
    """Single-frame θ head; outputs normalized θ (mean≈0, std≈1)."""
    def __init__(self, d, hidden=None):
        super().__init__()
        h = hidden or (d // 2)
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.GELU(),
            nn.Linear(h, 1)
        )
    def forward(self, z_t):  # z_t: (B, D) or (B, T, D)
        if z_t.dim() == 3:
            B, T, D = z_t.shape
            y = self.net(z_t.reshape(B*T, D)).reshape(B, T, 1)
        else:
            y = self.net(z_t).unsqueeze(-1)  # (B,1)
        return y  # normalized θ
        

class OmegaHead(nn.Module):
    def __init__(self, d, hidden=None):
        super().__init__()
        h = hidden or (2*d)
        in_dim = 3*d
        self.norm_z  = nn.LayerNorm(d)
        self.norm_zm = nn.LayerNorm(d)
        self.norm_dz = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, h), nn.GELU(),
            nn.Linear(h, h//2),   nn.GELU(),
            nn.Linear(h//2, 1)
        )

    def forward(self, z_t, z_tm1, dt):
        # remember original rank BEFORE any reshape
        seq_mode = (z_t.dim() == 3)
        if seq_mode:
            B, Tp, D = z_t.shape
            z_t   = z_t.reshape(B*Tp, D)
            z_tm1 = z_tm1.reshape(B*Tp, D)

        dz = (z_t - z_tm1) / dt
        x = torch.cat([self.norm_z(z_t),
                       self.norm_zm(z_tm1),
                       self.norm_dz(dz)], dim=-1)
        y = self.mlp(x)  # (B*Tp,1) or (B,1)

        if seq_mode:
            y = y.reshape(B, Tp, 1)  # (B, T-1, 1)
        return y