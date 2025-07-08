# Walking through **`utils/pendulum_dataset.py`**

Below is a section-by-section tour of the file, with pointers to the exact code lines you pasted.  
The aim is to show **what each block does, why it exists, and how the moving parts fit together** so you can tweak things confidently.

---

## 1. High–level intent (lines 1–44)

| Lines | Key idea | What it buys you |
|-------|----------|------------------|
| 1-26  | Doc-string banner + ASCII table of “research knobs”. | Self-documenting, so when you `help(PendulumDataset)` the three tunables jump out (`sub_steps`, `init_grid`, `random_action`). |
| 29-44 | Imports & *device* resolution. | `device` is **resolved once** (CUDA if available, else CPU) so every tensor created later is already on the right accelerator. |

---

## 2. `PendulumDataset` class header (lines 48-118)

### Constructor signature (67-87)

```python
def __init__(..., num_episodes=100, episode_length=200,
             img_size=64, seq_len=3, *,
             sub_steps=1, init_grid=None,
             random_action=True, transform=None):
```

* **`seq_len` check** (90-93). Physics losses (e.g., Euler–Lagrange) need at least two frames.
* **`sub_steps` check** (94-95). You cannot down-sample time (would break dynamics).
* Everything the caller might vary is **stored as an attribute** (98-103) so you can read it back later (handy for reproducibility logs).

### Internal buffers (105-108)

```python
self.frames  : List[np.ndarray]         # each H×W×3 uint8 image
self.states  : List[Tuple[float, float]]# (θ, ω) ground-truth for each frame
self.indices : List[int]                # start-indices of every T-frame window
```

These are **flat lists** so that random indexing stays O(1).

### Kick-off generation (111-112)

```python
self._generate(num_episodes, episode_length)
```

All synthetic data are generated once at instantiation; the dataset is then *frozen*.

---

### 3 · Private helper `_generate`
```python
142  def _generate(self, n_episodes: int, epi_len: int) -> None:
143      """Populate `self.frames / self.states / self.indices` in place."""
144      print("Generating pendulum trajectories …")
145      env = gym.make("Pendulum-v1")                     # mujoco-free env
146
147      # Choose seeds -------------------------------------------------
148      episode_seeds = (
149          self.init_grid
150          if self.init_grid is not None
151          else [None] * n_episodes
152      )
153
154      for seed in tqdm(episode_seeds):                 # episode loop ──▶
155
156          # ─── initialise episode ────────────────────────────────
157          if seed is None:
158              obs, _ = env.reset()                    # random θ₀,ω₀
159          else:                                       # deterministic
160              theta0, omega0 = seed
161              env.reset()                             # reset env state
162              env.unwrapped.state = np.array([theta0, omega0],
163                                             dtype=np.float32)
164              obs = np.array([np.cos(theta0), np.sin(theta0), omega0],
165                             dtype=np.float32)
166
167          ep_imgs, ep_states = [], []                 # episode buffers
168
169          # ─── main roll-out loop ────────────────────────────────
170          for _ in range(epi_len):
171
172              # densify time: inner loop ────────────────
173              for _ in range(self.sub_steps):
174                  action = (env.action_space.sample()
175                            if self.random_action
176                            else np.array([0.0], dtype=np.float32))
177                  obs, _, terminated, truncated, _ = env.step(action)
178                  if terminated or truncated:          # rare in v1
179                      break
180
181              # convert observation to (θ, ω) ──────────
182              theta = float(np.arctan2(obs[1], obs[0]))
183              omega = float(obs[2])
184
185              ep_imgs  .append(self._render_pendulum(theta))
186              ep_states.append((theta, omega))
187
188          # ─── window indexing (slide over episode) ─────────────
189          for t0 in range(0, len(ep_imgs) - self.seq_len + 1):
190              self.indices.append(len(self.frames) + t0)
191
192          # flat-append episode into master buffers ───────────────
193          self.frames.extend(ep_imgs)
194          self.states.extend(ep_states)
195
196      env.close()
197      print(f"Created {len(self.indices)} windows (seq_len={self.seq_len})")
```

| Line(s) | What happens & why |
|---------|--------------------|
| **142-145** | Define method; print progress banner; build a *fresh* `gym` env one time (re-using it across episodes avoids slow `close()`/`make()` cycles). |
| **148-152** | Decide the list of episode seeds:<br>• If `init_grid`✅ → every `(θ₀,ω₀)` becomes **one deterministic episode**.<br>• Else → list of `None` placeholders, length `n_episodes`, meaning “pick a random start for each episode.” |
| **154** | Loop over that list with `tqdm` so you see a progress bar. |
| **157-165** | *Episode initialisation*.<br>• Random case → `env.reset()` returns observation `obs`.<br>• Deterministic case → manually overwrite `env.unwrapped.state` **after** a reset, then rebuild the observation vector expected by Pendulum (`[cosθ, sinθ, ω]`). |
| **167** | Create **episode-local** lists to accumulate images & states. |
| **170-186** | *Frame loop* (length `epi_len`).<br>Inside each frame… |
| **173-179** |  • Run physics **`sub_steps` times** (densification).<br> • Action policy chosen by `random_action`:<br>  ▫ `True` → sampled torque (exploratory).<br>  ▫ `False` → `[0.0]` (passive swing-up).<br> • If the env somehow terminates (rare for Pendulum-v1) → break. |
| **181-186** |  • Convert `obs` → (θ,ω): `atan2` gives signed angle.<br> • Render deterministic RGB frame with `_render_pendulum`.<br> • Append frame & state to episode buffers. |
| **188-191** | Slide a length-`seq_len` window across the episode and record **start indices** **relative to the *global* image list** (`len(self.frames)` = number of frames already stored from previous episodes). |
| **193-194** | Dump the episode lists into the **global** buffers (`self.frames`, `self.states`). This maintains one big contiguous memory block for fast slicing. |
| **196-197** | Clean up the env; final printout tells you how many windows (i.e. training samples) exist. |

---

### 4 · Dataset protocol `__getitem__`
```python
206  def __getitem__(self, idx: int):
207      """
208      Returns
209      -------
210      imgs   : FloatTensor (T, 3, H, W)  # 0-1 range
211      states : FloatTensor (T, 2)        # (θ, ω)
212      """
213      start = self.indices[idx]             # global frame index
214      end   = start + self.seq_len
215
216      # -------- stack & normalise images ---------------------------
217      imgs = [
218          torch.from_numpy(self.frames[i]).float()
219               .permute(2, 0, 1) / 255.0      # (3,H,W) in [0,1]
220          for i in range(start, end)
221      ]
222      imgs = torch.stack(imgs)                # (T,3,H,W)
223
224      # -------- stack states ---------------------------------------
225      states = torch.tensor(self.states[start:end],
226                            dtype=torch.float32)  # (T,2)
227
228      # -------- optional transform ---------------------------------
229      if self.transform is not None:
230          imgs = self.transform(imgs)
231
232      return imgs.to(device), states.to(device)
```

| Line(s) | What happens & why |
|---------|--------------------|
| **213-214** | Translate the **window index** → concrete `[start:end]` slice inside `self.frames` / `self.states`. This indirection is what makes windows overlap **without duplicating memory**. |
| **216-221** | Comprehension:<br>• `self.frames[i]` is `(H,W,3)` **uint8**.<br>• `torch.from_numpy(...).float()` casts to float32.<br>• `.permute(2,0,1)` changes layout to **CHW** for PyTorch.<br>• `/255.0` scales to `[0,1]` for stable training.<br>The list-comp yields `seq_len` tensors which are then `torch.stack`ed → shape `(T,3,H,W)`. |
| **225-226** | Build the ground-truth `(θ,ω)` tensor **in the same temporal order**; dtype float32 is enough (high-precision not needed). |
| **229-230** | Apply an optional `torchvision`-style transform (**after** normalisation!). Because the whole sequence is already one tensor you can write transforms that work on time as well (e.g. random temporal cropping, channel-wise normalisation). |
| **232** | Move both tensors to the pre-chosen `device` and return. (If you prefer to pin memory for DataLoader → change here.) |

---

## 5. How the three **research knobs** map to code

| Knob | Where it lives | Concrete effect |
|------|----------------|-----------------|
| `sub_steps` | inner `for _ in range(self.sub_steps)` (172-183) | Higher temporal resolution by integrating physics *k×* per visible frame. |
| `init_grid` | seed selection & manual state set (151-167) | Provides deterministic coverage over the `(θ, ω)` plane — essential for evaluation grids. |
| `random_action` | ternary on `action` (173-179) | If **False**, torque = 0 ⇒ conservative dynamics (no stochastic flailing). |

---

## 6. Where to tweak things next

| Goal | Change | Line hint |
|------|--------|-----------|
| **Longer windows** | `seq_len` in constructor | 71 |
| **Higher-res images** | `img_size` in constructor | 70 |
| **No progress bar (silent)** | remove `tqdm` wrapper | 157 |
| **Add Gaussian image noise** | write a `transform` that adds noise | pass via `transform=` arg |
| **CPU-only for debugging** | `PND_DEVICE=cpu python train.py` | env-var read at line 38 |

---

**Bottom line:** everything boils down to two well-isolated routines:

* `_generate` — **data creation** (episodes → flat buffers)  
* `__getitem__` — **data delivery** (window slicing)

Once you understand those, the rest is just convenience plumbing.

### Clarifying the three follow-up questions

---

#### 1. *Training order* — do we always exhaust **all** overlapping windows of episode #1 before touching episode #2?

**Inside the `Dataset`, yes; inside your `DataLoader`, maybe.**

* **Dataset construction**  
  In `_generate` lines 188-191 the index table is filled **immediately after** an episode finishes:

  ```python
  for t0 in range(0, len(ep_imgs) - self.seq_len + 1):
      self.indices.append(len(self.frames) + t0)
  ```

  Because this append happens **episode by episode**, the raw list `self.indices` has the layout:

  ```
  [ ep1-win0, ep1-win1, …, ep1-winN,
    ep2-win0, ep2-win1, …, ep2-winN,
    … ]
  ```

* **DataLoader iteration**  
  * If you build the loader with `shuffle=False` (the default)  
    ```python
    train_loader = DataLoader(dataset, batch_size=…, shuffle=False)
    ```  
    then windows are yielded **in the exact order above**: all of episode #1, then all of episode #2, etc.

  * If you set `shuffle=True`, PyTorch shuffles the *index list*, so batches are pulled randomly from the entire corpus; sequences from different episodes will be interleaved.

Bottom line: **episode-first ordering is guaranteed only when `shuffle=False`.**

---

#### 2. *Ground-truth* — are the *true* θ and ω stored for every rendered frame?

Yes.  
* Each time the environment is stepped (line 177) the *raw* Gym observation `obs = [cos θ, sin θ, ω]` is converted back to continuous (`θ, ω`) on lines 182-183:

  ```python
  theta = float(np.arctan2(obs[1], obs[0]))
  omega = float(obs[2])
  ```

* The pair `(theta, omega)` is appended to `ep_states` (line 186) and later flattened into `self.states` (line 194).

Therefore every single image has a **one-to-one** numeric label, and `__getitem__` returns those labels untouched (lines 225-226).  
No subsampling, no lossy quantisation.

---

#### 3. *`episode_length` vs `sub_steps`* — what do they control and how do they interact?

| Concept | Where set | What it means | Effect on dataset |
|---------|-----------|---------------|-------------------|
| **`episode_length`** | Constructor arg → passed to `_generate` as `epi_len` (line 142) | Number of **frames you actually keep** per episode. The outer `for` loop (line 170) runs exactly `episode_length` iterations. | • Directly proportional to dataset size.<br>• More frames ⇒ more overlapping windows. |
| **`sub_steps`** | Constructor arg stored as `self.sub_steps` | **How many *internal* physics steps** the environment advances **before you store one frame**. Implemented by the inner loop (lines 173-179). | • Increases the *simulation* resolution while keeping the same *visual* frame rate.<br>• Each stored frame represents a shorter physical time-interval (dt / `sub_steps`).<br>• Trajectories look **smoother** and satisfy physics constraints more tightly (good for HNN/LNN losses). |

> **Mnemonic:**  
> `episode_length` ⇒ **how long** your movie is.  
> `sub_steps` ⇒ **how finely** each movie frame is simulated before you press “record”.

Putting numbers on it:

| Setting | What happens physically | Stored frames |
|---------|-------------------------|---------------|
| `episode_length=200, sub_steps=1` | 200 env steps, 200 renders | 200 |
| `episode_length=200, sub_steps=4` | 800 env steps, but only every 4th state rendered | 200 (same!), but dynamics are now 4 × finer between frames |

So `episode_length` affects **dataset size**, `sub_steps` affects **temporal granularity** of the physics underlying each frame.

---

#### Cheat-sheet

* **Want more data?** → raise `episode_length` (or `num_episodes`).  
* **Want smoother, more accurate physics between frames?** → raise `sub_steps`.  
* **Need strict chronological batching (episode by episode)?** → keep `shuffle=False`.

With these three dials you can tailor the pendulum dataset to almost any dynamics‐learning experiment.

## Working with *whole-episode* trajectories & visual comparisons

Below is a practical recipe that answers the three sub-questions you asked.

---

### 1  ❱  Extract the **full (θ, ω) time-series** of an episode

Because `_generate` writes frames **episode by episode** and you supplied a
fixed `episode_length`, you can recover the chronological ordering with
simple slicing:

```python
epi_len = dataset.episode_length               # e.g. 200
ep_idx   = 0                                   # ← choose episode number

# Frame/time indices belonging to that episode
start = ep_idx * epi_len
end   = start + epi_len                        # *exclusive*

theta, omega = zip(*dataset.states[start:end]) # tuples → lists
theta = np.asarray(theta)                      # shape (epi_len,)
omega = np.asarray(omega)
```

> *Corner-case:* if you ever set `sub_steps > 1` **and** the env terminates
> early, the episode might be shorter than `episode_length`.
> In that case you can detect episode boundaries with:
> ```python
> # indices added by _generate just **after** each episode finishes
> episode_starts = np.flatnonzero(np.diff(dataset.indices, prepend=-1) < 0)
> ```

---

### 2  ❱  Rolling the **neural network** over a *full* trajectory

A model trained on overlapping windows can still *predict* an entire
sequence—just feed windows **sequentially** (sliding forward one step at a
time) and chain its outputs:

```python
def rollout(model, init_window, horizon):
    """
    Auto-regressively predict `horizon` future steps.
    `init_window`  : (T,C,H,W) tensor  – the same length the model saw during training
    returns preds  : (horizon, 2)      – θ̂ and ω̂
    """
    window = init_window.clone()        # avoid in-place edits
    preds  = []
    model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            θω_hat = model(window)      # shape (T,2) or maybe just (1,2)
            preds.append(θω_hat[-1])    # newest prediction
            # slide window: drop oldest frame, append newest prediction/render
            window = torch.roll(window, shifts=-1, dims=0)
            window[-1,0:2] = render_from_angles(θω_hat[-1,0])  # pseudo-code
    return torch.stack(preds)           # (horizon,2)
```

*That* is exactly what the “`rollout`” helper in your notebook was doing.

---

### 3  ❱  Plotting **true vs predicted** trajectories

Matplotlib code—drop it into a cell after you have `theta_true`, `theta_pred`,
`omega_true`, `omega_pred`.  
The `np.unwrap` call removes the ±π jumps so the sine wave looks continuous.

```python
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(len(theta_true))          # time axis (frame index)

plt.figure(figsize=(10,4))
plt.plot(t, np.unwrap(theta_true), label='θ true')
plt.plot(t, np.unwrap(theta_pred), '--', label='θ pred')
plt.xlabel('frame')
plt.ylabel('angle (rad)')
plt.legend()
plt.title('Pendulum angle vs time')

plt.figure(figsize=(10,4))
plt.plot(t, omega_true, label='ω true')
plt.plot(t, omega_pred, '--', label='ω pred')
plt.xlabel('frame')
plt.ylabel('angular velocity (rad/s)')
plt.legend()
plt.title('Angular velocity vs time')
plt.show()
```

The resulting plot **should** resemble a noisy sinusoid for θ and its
derivative-like waveform for ω; visually you want the dashed
“pred” curves hugging the solid “true” curves.

---

### 4  ❱  Concept recap

| Term | Controls | Impact |
|------|----------|--------|
| `episode_length` | **How many frames you save** per episode | Affects dataset size & how long each curve is on the plot. |
| `sub_steps` | **How many env steps between saved frames** | Higher value → same *visual* series length, but each step is a smaller Δt → smoother physics. |
| `seq_len` | Window size used for **training & inference input** | Model context; does **not** constrain how long a rollout can be. |

*You train on overlapping windows because it gives the net more training
samples and makes gradients local; you evaluate on entire rollouts to see
global fidelity.*

With those snippets you can:

1. Retrieve complete ground-truth trajectories.  
2. Run your model in rollout mode to get predicted trajectories.  
3. Plot both pairs (θ, ω) and eyeball the fit.

Happy debugging!