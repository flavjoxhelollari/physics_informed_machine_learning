import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# 2. V-JEPA backbone
# ============================
class PatchEmbed(nn.Module):
    """Image → patch embeddings"""
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        # convolution patch embedding
        self.n_patches = (img_size // patch_size) ** 2
        self.proj      = nn.Conv2d(in_chans, embed_dim,
                                   kernel_size=patch_size,
                                   stride=patch_size)
    def forward(self, x):
        x = self.proj(x)                     # (B,embed,H',W')
        x = x.flatten(2).transpose(1, 2)     # (B,N,embed)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, mlp_ratio):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                embed_dim,
                num_heads,
                int(embed_dim * mlp_ratio),
                batch_first=True,
                norm_first=True
            ) for _ in range(depth)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MaskingStrategy:
    """Block + random masking identical to original V-JEPA impl."""
    def __init__(self, num_patches, mask_ratio=0.75, block_size=4):
        self.N  = num_patches
        self.r  = mask_ratio
        self.bs = block_size

    def random_masking(self, B):
        n_keep = int(self.N * (1 - self.r))
        noise  = torch.rand(B, self.N, device=device)
        ids    = torch.argsort(noise, dim=1)
        restore= torch.argsort(ids , dim=1)
        mask   = torch.ones(B, self.N, device=device)
        mask[:, :n_keep] = 0
        mask   = torch.gather(mask, 1, restore)
        return mask.bool()

    def block_masking(self, B):
        grid  = int(np.sqrt(self.N))
        masks = torch.zeros(B, self.N, device=device)
        n_blk = int((self.N * self.r) / (self.bs ** 2))
        for b in range(B):
            for _ in range(n_blk):
                h0 = np.random.randint(0, grid - self.bs + 1)
                w0 = np.random.randint(0, grid - self.bs + 1)
                for h in range(h0, h0 + self.bs):
                    for w in range(w0, w0 + self.bs):
                        masks[b, h * grid + w] = 1
        return masks.bool()

class VJEPA(nn.Module):
    """
    Vision-based Joint Embedding Predictive Architecture
    (unchanged naming; context / target encoders etc.)
    """
    def __init__(
        self,
        img_size=64,
        patch_size=8,
        in_chans=3,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        prediction_head_dim=192,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.n_patches
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.context_encoder = TransformerEncoder(
            embed_dim, depth, num_heads, mlp_ratio
        )
        self.target_encoder  = TransformerEncoder(
            embed_dim, depth, num_heads, mlp_ratio
        )

        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, prediction_head_dim),
            nn.GELU(),
            nn.Linear(prediction_head_dim, embed_dim)
        )

        self.masking = MaskingStrategy(self.num_patches)
        nn.init.normal_(self.pos_embed, std=0.02)

    # ---------- helpers ----------
    def forward_context(self, imgs, context_mask):
        x = self.patch_embed(imgs) + self.pos_embed  # (B,N,D)
        B, N, D = x.shape
        keep    = ~context_mask
        out     = torch.stack([
            F.pad(x[i][keep[i]], (0, 0, 0, N - keep[i].sum()))
            for i in range(B)
        ])
        return self.context_encoder(out)

    def forward_target(self, imgs, target_mask):
        x = self.patch_embed(imgs) + self.pos_embed
        B, N, D = x.shape
        out = torch.stack([
            F.pad(x[i][target_mask[i]], (0, 0, 0, N - target_mask[i].sum()))
            for i in range(B)
        ])
        return self.target_encoder(out)

    # ---------- main forward ----------
    def forward(self, imgs):
        """
        Returns:
            loss  : MSE between predictor(ctx) and target_feats
            pred  : predicted embeddings
            target: target embeddings
        """
        B = imgs.size(0)
        cmask = self.masking.block_masking(B)
        tmask = self.masking.random_masking(B) & ~cmask

        ctx_feats = self.forward_context(imgs, cmask)
        with torch.no_grad():
            tgt_feats = self.forward_target(imgs, tmask)

        pred = self.predictor(ctx_feats)
        loss = F.mse_loss(pred, tgt_feats)

        return loss, pred, tgt_feats

# ============================
# 3. Lagrangian Neural Network
# ============================
class LNN(nn.Module):
    """
    Minimal PyTorch port of Miles Cranmer's LNN.
    Learns scalar L(q,v); physics loss is Euler–Lagrange residual.
    """
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, qv):                     # (B,2d) → scalar L
        return self.net(qv)

    def lagrangian_residual(self, q, v, dt=0.01):
        """
        q, v : (B, T, d) tensors with T ≥ 2.
        Returns mean-squared Euler–Lagrange residual over (B,T-1,d).
        """
        B, T, d = q.shape
        z   = torch.cat([q, v], dim=-1).reshape(B*T, -1).requires_grad_(True)
        L   = self.forward(z).sum()
        dLd = torch.autograd.grad(L, z, create_graph=True)[0]
        dLdq, dLdv = dLd.split(d, dim=-1)
        dLdq = dLdq.view(B, T, d)
        dLdv = dLdv.view(B, T, d)

        ddt_dLdv = (dLdv[:, 1:] - dLdv[:, :-1]) / dt  # (B,T-1,d)
        residual = ddt_dLdv - dLdq[:, :-1]           # Euler–Lagrange
        return (residual ** 2).mean()

# ============================
# 3b. Hamiltonian Neural Network
# ============================
class HNN(nn.Module):
    """Minimal 2-D Hamiltonian NN (q = θ, p = ω)."""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2)               # outputs F1, F2
        )
        self.register_buffer("J", torch.tensor([[0., 1.],
                                                [-1., 0.]]))  # symplectic matrix

    def time_derivative(self, qp):                 # qp: (B,2)
        F1, F2 = self.net(qp).split(1, 1)
        dF2 = torch.autograd.grad(F2.sum(), qp, create_graph=True)[0]
        return dF2 @ self.J.T                      # (dq/dt, dp/dt)