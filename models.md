# Walking through **`models.py`** line-by-line  
*(Pendulum V-JEPA study: tokeniser → tiny ViT → V-JEPA → LNN & HNN)*

Below you’ll find a **section-scoped commentary**.  
Each bullet cites the **exact code line** (relative to the snippet you pasted) and explains the why, not just the what.

---

## 0 · Global setup (lines 1-40)

* **8** – `device` is resolved once and reused by every sub-module → you never risk half-GPU/half-CPU tensors.
* The file imports only `numpy`, `torch`, `math` → truly *self-contained*; you can copy-paste this module anywhere.

---

---

## 1 · `PatchEmbed` – turning an image into a sequence of patch tokens
```python
44 class PatchEmbed(nn.Module):
45     """
46     Convolutional patch tokeniser (a *very* light ViT stem).
```
*Lines 44-46* A *tokeniser* converts a dense image into a length-`N` set of
vectors so the Transformer can treat it like an ordinary sequence.

```python
56         self.n_patches: int = (img_size // patch_size) ** 2
```
*Line 56* **Important assumption:** `img_size` is divisible by
`patch_size`.  
Result: the image is tiled into `H = W = img_size / patch_size`
non-overlapping squares ⇒ total patches = `H·W`.

> ▸ *Why not allow overlap?* Non-overlapping patches keep the sequence
>   short; overlapping would explode `N` & slow the model *quadratically*.

```python
60         self.proj: nn.Conv2d = nn.Conv2d(
61             in_chans,
62             embed_dim,
63             kernel_size=patch_size,
64             stride=patch_size,
65         )
```
*Lines 60-65* One **single** `Conv2d` does the heavy lifting:

| Parameter | Effect |
|-----------|--------|
| `kernel_size = patch_size` | Each convolutional “receptive field” covers exactly **one patch**. |
| `stride = patch_size` | The filters slide by *one patch at a time* → no overlap, no padding. |
| `in_chans → embed_dim` | Turns every **patch-sized RGB cut-out** into an `embed_dim`-long vector (by dotting with `embed_dim` kernels). |

> ▸ *Why convolution instead of unfolding + linear?*  
>   Convolution is **vectorised C++/CUDA** and fuses unfold + mat-mul in one op. It’s ~2× faster and uses less memory.

```python
72     def forward(self, x: torch.Tensor) -> torch.Tensor:
73         """
74         x : (B, C, H, W)  → out : (B, N_patches, embed_dim)
75         """
76         x = self.proj(x)               # (B, D, H/ps, W/ps)
77         x = x.flatten(2)               # (B, D, N)
78         x = x.transpose(1, 2)          # (B, N, D)
79         return x
```
*Line 76* After conv, channels = `embed_dim`, spatial dims =
`H/patch_size`, `W/patch_size`.

*Line 77* `flatten(2)` merges the two spatial axes into one long axis
(`N = H/ps · W/ps`).

*Line 78* `transpose(1,2)` swaps **channels ↔ patches** so the final shape
is `(B, N, D)` (`batch_first=True` convention).

> **Take-away:** 3 lines of tensor ops produce a ViT-ready sequence without an extra `nn.Linear` or `torch.nn.Unfold`.

---

### ★ Power-user tuning knobs (PatchEmbed)

| What you want | Change | Side effects |
|---------------|--------|--------------|
| **Higher spatial resolution** | decrease `patch_size` (e.g. 8→4) | `N_patches ↑ 4×` ⇒ attention cost ↑ O(N²). |
| **More expressive tokens** | increase `embed_dim` | Memory ↑ linearly, attention cost ↑ quadratically (because `D` enters Q,K,V mat-muls). |
| **Channel dropout / colour jitter** | prepend a `torchvision` transform before `PatchEmbed` | Keeps this module stateless. |

---

## 2 · `TransformerEncoder` – mini-ViT stack
```python
81 class TransformerEncoder(nn.Module):
82     """
83     Very thin wrapper around *depth* identical
84     `nn.TransformerEncoderLayer` blocks (no CLS token, batch_first).
85     """
```
*Lines 83-84* Key departures from vanilla ViT:

* **No `CLS` token** – outputs remain patch-aligned.
* **batch_first=True** – easier to index `(B, N, D)` throughout.

```python
88     def __init__(
89         self,
90         embed_dim: int,
91         depth: int,
92         num_heads: int,
93         mlp_ratio: float = 4.0,
94     ) -> None:
```
*Lines 88-94* Hyper-params directly mirror the original ViT paper.

```python
95         super().__init__()
96         self.layers = nn.ModuleList(
97             [
98                 nn.TransformerEncoderLayer(
99                     d_model=embed_dim,
100                    nhead=num_heads,
101                    dim_feedforward=int(embed_dim * mlp_ratio),
102                    batch_first=True,
103                    norm_first=True,
104                )
105                for _ in range(depth)
106            ]
107        )
```
*Lines 98-104* For each layer:

| Arg | Purpose |
|-----|---------|
| `d_model=embed_dim` | Token dimension. |
| `nhead=num_heads` | Multi-head attention splits `embed_dim` evenly. |
| `dim_feedforward=int(embed_dim*mlp_ratio)` | Hidden size of the FFN (ViT uses 4× by default). |
| `batch_first=True` | Accepts `(B, N, D)` without transpose gymnastics. |
| `norm_first=True` | Pre-Norm variant (stabler on deep nets). |

`ModuleList` is used—**not** `nn.TransformerEncoder`—so you can later
index or replace individual layers (handy for probing attention maps).

```python
103                    norm_first=True,
104                )
105                for _ in range(depth)
...
106        )
```
*Line 105* `for _ in range(depth)` – `depth` is the **stack depth**;
`depth=6` gives a 6-layer ViT-tiny–like encoder.

```python
108     def forward(self, x: torch.Tensor) -> torch.Tensor:
109         for blk in self.layers:
110             x = blk(x)
111         return x
```
*Lines 109-110* Serially applies each block.  
Because every `TransformerEncoderLayer` already contains
*residual + layer-norm*, no outer skip connection is needed.

> **Memory note:** With PyTorch ≥1.12 you can add
> ```
> with torch.cuda.amp.autocast():
>     x = blk(x)
> ```
> here for mixed-precision; no edits elsewhere.

---

### ★ Design rationale & trade-offs (Transformer)

| Choice | Why it matters |
|--------|---------------|
| **Pre-Norm (`norm_first=True`)** | Tolerates deeper stacks without exploding/vanishing grads. |
| **Small `depth` (≤ 6)** | Pendulum frames are simple; bigger ViTs would over-fit & slow JEPA training. |
| **No positional dropout** | Learned pos-embed is small (`N≤64`). With such small token counts, dropout would remove too much signal. |

---

### 🔧 Extending / hacking the encoder

1. **Spatial ⇄ temporal unrolling**  
   If you later combine multiple frames into a *spatio-temporal* ViT,
   you can simply reshape `(B, T, N_patch, D)` → `(B, T·N_patch, D)` and feed the same module.

2. **Patch-drop regularisation**  
   Insert
   ```python
   keep = torch.rand(B, N, device=x.device) > p_drop
   x = x[keep].reshape(B, -1, D)
   ```
   **before** the loop to inject stochastic token dropping.

3. **Per-layer supervision**  
   Replace the `for blk in self.layers` loop with
   ```python
   for i, blk in enumerate(self.layers):
       x = blk(x)
       if i in probe_layers:
           feats.append(x.detach())
   ```
   to record mid-level representations.

---

## 3 · `MaskingStrategy` – generating context/target masks for V-JEPA

### 3.0  Constructor (lines 110-119)
```python
110 class MaskingStrategy:
...
118     def __init__(
119         self,
120         num_patches: int,
121         mask_ratio: float = 0.75,
122         block_size: int = 4,
123     ) -> None:
124         self.N  = num_patches   # total patch tokens
125         self.r  = mask_ratio    # global masking ratio
126         self.bs = block_size    # edge length of a square block
```
*Lines 124-126*  
Store three **hyper-parameters**:

| Symbol | Meaning | Typical |
|--------|---------|---------|
| `N` | total tokens (≡ `patch_embed.n_patches`) | ≤ 64 for 64×64 / 8 |
| `r` | ratio of tokens that **will be masked** | 0.75 (paper default) |
| `bs`| block side length (in patch units) used by *block masking* | 4 |

> **Intent ►** keep the API **stateless**: every call to
>   `.random_masking` or `.block_masking` is purely functional and can be
>   re-used across dataloaders without race conditions.

---

### 3.1  `random_masking` (lines 127-141)
```python
127     def random_masking(self, B: int) -> torch.BoolTensor:
128         """
129         Randomly *keeps* ⌊N·(1 − mask_ratio)⌋ patches per sample.
130         """
131         n_keep = int(self.N * (1.0 - self.r))
...
134         scores = torch.rand(B, self.N, device=device)   # ∈ [0,1]
135         ids_sorted  = torch.argsort(scores, dim=1)      # low → high
136         ids_restore = torch.argsort(ids_sorted, dim=1)  # inverse perm
137
138         mask = torch.ones(B, self.N, device=device)     # 1 = masked
139         mask[:, :n_keep] = 0                            # 0 = visible
140         mask = torch.gather(mask, 1, ids_restore)       # undo sort
141         return mask.bool()
```

*Line-by-line*

| Ln | Action | Why this way? |
|----|--------|---------------|
| **131** | Number of *visible* tokens per sample. | Integer truncation gives deterministic length. |
| **134** | IID uniform scores per token ⇒ stable across devices and seeds. | Sorting preserves device order, no duplicates. |
| **135-136** | Compute **argsort** twice to get *restore permutation* `ids_restore`. | Needed to put kept/masked flags back in original patch order. |
| **138-140** | Start with all-ones (all masked), flip first `n_keep` to 0 (visible), *then* restore ordering. | Cheaper than building a Boolean index per row. |

> **Complexity:** `O(B·N·log N)` per call, but `N≤64` so negligible.

> **Numerical edge-case:** if `mask_ratio=1.0` then `n_keep=0`.
> All tokens will be masked → JEPA loss becomes ill-posed.
> The constructor doesn’t guard against this; caller’s responsibility.

---

### 3.2  `block_masking` (lines 143-155)
```python
143     def block_masking(self, B: int) -> torch.BoolTensor:
...
147         grid  = int(math.sqrt(self.N))               # patches per side
148         masks = torch.zeros(B, self.N, device=device)
149         n_blk = int((self.N * self.r) / (self.bs**2))# blocks per sample
...
151             for _ in range(n_blk):
152                 h0 = np.random.randint(0, grid - self.bs + 1)
153                 w0 = np.random.randint(0, grid - self.bs + 1)
154                 for dh in range(self.bs):
155                     for dw in range(self.bs):
156                         idx = (h0 + dh) * grid + (w0 + dw)
157                         masks[b, idx] = 1
158         return masks.bool()
```

*Key points*

* **147** – `grid = √N` assumes `N` is a perfect square
  (true for non-overlapping square patches).
* **149** – `n_blk` chosen so that total expected masked tokens ≈
  `N·r`. Rounding down keeps *mask ratio ≤ requested*.
* **152-157** – Double nested loops paint each square block.
  This is CPU-side Python; but `n_blk` is small (≤ 4) so GPU copy costs
  dominate anyway.

> **Behavioural difference vs `random_masking`:**
> `block_masking` creates **contiguous** occlusions, forcing the model to
> infer large missing regions → better long-range context learning.

---

### ★ Trade-offs & hacking tips (Masking)

| Want… | Do this | Effect |
|-------|---------|--------|
| **Keep semantic edges crisp** | decrease `block_size` | Smaller occlusions → easier task. |
| **Harder task / fewer visibles** | increase `mask_ratio` | Bigger MSE, slower convergence. |
| **Deterministic masks** | seeding NumPy & PyTorch PRNGs before each call | Required for ablation reproducibility. |
| **Video masking** | extend `MaskingStrategy` with `(T·N)` tokens awareness | Same pattern, just bigger `N`. |

---

## 4 · `VJEPA` – Vision-only Joint Embedding Predictive Architecture

### 4.0  Constructor overview (lines 159-204)
```python
178     def __init__(
...
186         embed_dim: int = 384,
187         depth: int = 6,
188         num_heads: int = 6,
189         mlp_ratio: float = 4.0,
190         prediction_head_dim: int = 192,
191     ) -> None:
```
*Lines 186-191* – Hyper-parameters:  
`embed_dim × depth × heads` ≈ ViT-Small scale (7 M parameters).

#### 4.0.1  Patch tokeniser & positional encoding (194-200)
```python
194         self.patch_embed = PatchEmbed(...)
198         self.num_patches = self.patch_embed.n_patches
199         self.pos_embed   = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
200         nn.init.normal_(self.pos_embed, std=0.02)
```
*199* – Learned **absolute** positional embeddings.
Cool-down: they are *not* re-scaled like in MAE; 0.02 std is standard ViT
initialisation.

#### 4.0.2  Siamese encoders (202-205)
```python
202         self.context_encoder = TransformerEncoder(...)
203         self.target_encoder  = TransformerEncoder(...)
```
*Design choice:* two **independent** weight sets.
⇒ Prevents trivial identity mapping (`ctx_emb == tgt_emb`) because encoders
see *different* masks and can specialise.

#### 4.0.3  Predictor MLP (207-211)
```python
207         self.predictor = nn.Sequential(
208             nn.Linear(embed_dim, prediction_head_dim),
209             nn.GELU(),
210             nn.Linear(prediction_head_dim, embed_dim),
211         )
```
Two-layer **bottleneck**:  
* reduce → non-linear → expand back.  
This follows the BYOL design; helps stabilise training.

#### 4.0.4  Mask engine instance (213-214)
```python
213         self.masking = MaskingStrategy(self.num_patches)
```
Single engine reused per forward pass.

---

### 4.1  `forward_context` (lines 218-231)
```python
219     x = self.patch_embed(imgs) + self.pos_embed        # (B,N,D)
220     B, N, D = x.shape
221     keep = ~context_mask                                # False = keep
```
*220-221* – `keep` is a **row-wise** boolean mask.

```python
222     x_vis = torch.stack(
223         [F.pad(x[i][keep[i]], (0,0,0,N - keep[i].sum())) for i in range(B)]
224     )
225     return self.context_encoder(x_vis)
```
*Rows 222-224* – **Per-sample padding** ensures that after masking every
sequence is back to length `N`, so batch dimensions align.  
`F.pad(..., (0,0,0, K))` pads missing tokens with **zeros** (learned
pos-embed is *not* added for padded tokens → prevents the model from
cheating by reading positional noise).

> **Computational note:** Because masking reduces the **effective token
> count**, you could skip padding and feed variable-length sequences
> using PyTorch 2.2’s `nn.MultiheadAttention` with `key_padding_mask`.
> The current implementation trades a bit of flops for code simplicity.

---

### 4.2  `forward_target` (lines 233-238)
Same as `forward_context` but **selects masked tokens** (`target_mask[i]`).

> **Stop-gradient** will be applied *outside* when this function is called.

---

### 4.3  Main `forward` training pass (lines 240-272)
```python
248     ctx_mask = self.masking.block_masking(B)           # contiguous
249     tgt_mask = self.masking.random_masking(B) & ~ctx_mask
```
*248-249* – Ensure **disjoint** masks so no patch is both context *and*
target.

```python
251     ctx_emb = self.forward_context(imgs, ctx_mask)
252     with torch.no_grad():
253         tgt_emb = self.forward_target(imgs, tgt_mask)
```
*252-253* – `torch.no_grad()` freezes gradients through the **target**
path → BYOL/JEPA style representation learning.

```python
255     pred = self.predictor(ctx_emb)       # (B,N,D)
256     loss = F.mse_loss(pred, tgt_emb)
257     return loss, pred, tgt_emb
```
*255-257* – **Mean-squared error** over all tokens and feature dims.

> **Contrast with MAE:** MAE reconstructs pixel-space; JEPA reconstructs
> *representation-space* → avoids blurry averages and speeds convergence.

---

### ★ Trade-offs & tuning levers (VJEPA)

| Desire | Change | Consequence |
|--------|--------|-------------|
| **More spatial detail** | decrease `patch_size` | `num_patches ↑` ⇒ more compute, masks cover smaller real-world area. |
| **Harder context task** | enlarge `block_size` *or* raise `mask_ratio` | Predictor must fill bigger holes; may need bigger `depth`. |
| **Prevent collapse** | add predictor-target *Cosine* loss term | Encourages scale-invariant alignment, like BYOL’s “τ-update”. |
| **Faster training** | share weights between `context_encoder` & `target_encoder` | Halves params but risks representational collapse. |

---

### 🔧 Hacking ideas

1. **Cross-modal JEPA**  
   Replace `PatchEmbed` with an *audio spectrogram* conv stem and share the
   same transformer: masking logic stays, enabling image↔audio alignment.

2. **Temporal JEPA**  
   Treat each video frame as an additional *patch dimension*:  
   reshape `(B, T, N, D) → (B, T·N, D)` and adjust `MaskingStrategy`
   to operate on `(T·N)` tokens.

3. **Curriculum masking**  
   Linearly increase `mask_ratio` from 0.25 → 0.75 over training epochs.
   Implement by reading `epoch` in training loop and re-instantiating
   `MaskingStrategy` with new `mask_ratio`.

---

With these explanations you now know **exactly** how masks are generated,
how they flow through the twin encoders and predictor, and where to hook
in modifications without breaking tensor shapes or broadcast semantics.

## 5 · `LNN` – minimal Lagrangian NN (lines 250-301)

* Network body: **MLP(2→hidden→hidden→1)** with tanh activations.  
  Input is `(q, v)` where `q=θ`, `v=ω` → 2 numbers → scalar **L**.

* **`lagrangian_residual`** (273-300)  
  *Purpose:* compute the Euler–Lagrange residual ‖d/dt (∂L/∂v) − ∂L/∂q‖².  
  *Key tricks:*  
  * **Automatic differentiation** over **all time steps at once** by concatenating them (`z.reshape(B*T, -1)`).  
  * `create_graph=True` keeps second-order grads so you can back-prop through the residual into network weights.  
  * Optional `per_time_step=True` returns a curve (useful for plotting how physics violation evolves).

---

## 6 · `HNN` – minimal Hamiltonian NN (lines 306-356)

* **MLP** outputs `(F₁, F₂)` (a 2-vector).  
* **Fixed symplectic matrix J** registered as a buffer so `.to(device)` moves it automatically.
* **`time_derivative`** (327-341)  
  1. Split MLP output.  
  2. Compute gradient of `F₂` with respect to input `(θ, ω)`.  
  3. Multiply by `Jᵀ` → canonical equations:  
     ```
     θ̇ =  ∂F₂/∂ω
     ω̇ = −∂F₂/∂θ
     ```
Thus the net learns a **single scalar potential** that implicitly defines the vector field.

---

### Cheat-sheet of how everything connects

```
image (B,3,H,W)
   │
PatchEmbed ──► (B,N,D) ────► +pos ──► context_encoder
                                    │         ▲
                                    │         │ no-grad
                                    ▼         │
mask.ctx  random.ctx        predictor MLP     │
mask.tgt  block.tgt            │              │
      ▼                        │              │
target_encoder ────────────────┴──────────────┘
              MSE loss (JEPA)
```

* The **visual backbone** (PatchEmbed + Transformer) supplies an embedding space.  
* **JEPA loss** aligns *visible* and *masked* sub-views.  
* **LNN/HNN** attach downstream to `(θ, ω)` sequences (not to pixels) to enforce physics consistency.

You now have a complete mental map to modify, extend, or debug any part of the architecture without black-box guessing.