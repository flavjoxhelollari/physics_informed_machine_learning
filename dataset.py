import gym, torch, numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image, ImageDraw

# --------------------------------------------
# PendulumDataset   (back-compatible + 3 knobs)
# --------------------------------------------
#  Knobs (all optional)
#  --------------------
#   sub_steps      : int ≥1
#       finer temporal resolution.  If =3 the env is stepped 3×
#       internally for every *stored* frame  ⇒  ≈3× denser trajectory.
#
#   init_grid      : list[(theta0, omega0)]
#       explicit initial states.  Each tuple becomes ONE episode and
#       overrides num_episodes.  Lets you build a deterministic phase-
#       space mesh for evaluation.
#
#   random_action  : bool
#       True  – default/old behaviour (random torque)
#       False – action = 0  ⇒ purely conservative swing-up dynamics.
#
#  Leave every knob at its default to reproduce the original dataset.
# --------------------------------------------

class PendulumDataset(Dataset):
    def __init__(self,
                 num_episodes: int = 100,
                 episode_length: int = 200,
                 img_size: int = 64,
                 seq_len: int = 3,
                 *,
                 sub_steps: int = 1,                       # knob ①
                 init_grid: list[tuple[float, float]] | None = None,  # knob ②
                 random_action: bool = True,               # knob ③
                 transform=None):
        assert seq_len >= 2, "seq_len must be ≥ 2 for physics loss"
        self.img_size   = img_size
        self.seq_len    = seq_len

        # new-knob attributes
        self.sub_steps     = max(1, sub_steps)
        self.init_grid     = init_grid
        self.random_action = random_action
        self.transform     = transform

        self.frames, self.states, self.indices = [], [], []
        self._generate(num_episodes, episode_length)

    # ------------------------------------------------------------------
    # 1 · helper: draw pendulum image for angle theta
    # ------------------------------------------------------------------
    def _render_pendulum(self, theta: float) -> np.ndarray:
        L, cx, cy = self.img_size*0.4, self.img_size//2, self.img_size//2
        ex, ey = int(cx + L*np.sin(theta)), int(cy + L*np.cos(theta))

        img  = Image.new("RGB", (self.img_size, self.img_size), "white")
        draw = ImageDraw.Draw(img)
        draw.line([(cx, cy), (ex, ey)], fill="black", width=3)
        draw.ellipse([(cx-5, cy-5), (cx+5, cy+5)], fill="red")
        draw.ellipse([(ex-8, ey-8), (ex+8, ey+8)], fill="blue")
        return np.asarray(img)

    # ------------------------------------------------------------------
    # 2 · generate roll-outs
    # ------------------------------------------------------------------
    def _generate(self, n_episodes: int, epi_len: int):
        print("Generating pendulum trajectories …")
        env = gym.make("Pendulum-v1")

        # pick episode seeds
        seeds = self.init_grid if self.init_grid is not None else [None]*n_episodes

        for seed in tqdm(seeds):
            # --- set initial state -------------------------------------
            if seed is None:
                obs, _ = env.reset()
            else:
                theta0, omega0 = seed
                env.reset()
                env.unwrapped.state = np.array([theta0, omega0], dtype=np.float32)
                # newer gymnasium lacks state_to_obs(); build obs manually
                obs = np.array([np.cos(theta0), np.sin(theta0), omega0],
                               dtype=np.float32)
            # ------------------------------------------------------------

            ep_imgs, ep_states = [], []

            for _ in range(epi_len):
                # ----- optional temporal densification -----------------
                for _ in range(self.sub_steps):
                    action = (env.action_space.sample()
                              if self.random_action
                              else np.array([0.0], dtype=np.float32))
                    obs, _, done, trunc, _ = env.step(action)
                    if done or trunc:
                        break
                # --------------------------------------------------------

                theta = np.arctan2(obs[1], obs[0])
                omega = float(obs[2])

                ep_imgs.append(self._render_pendulum(theta))
                ep_states.append((theta, omega))

            # slide a window (unchanged logic)
            for t0 in range(0, len(ep_imgs) - self.seq_len + 1):
                self.indices.append(len(self.frames) + t0)

            self.frames.extend(ep_imgs)
            self.states.extend(ep_states)

        env.close()
        print(f"Created {len(self.indices)} windows (seq_len={self.seq_len})")

    # ------------------------------------------------------------------
    # 3 · PyTorch Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start, end = self.indices[idx], self.indices[idx] + self.seq_len

        imgs = [torch.from_numpy(self.frames[i]).float()
                           .permute(2,0,1)/255. for i in range(start, end)]
        imgs = torch.stack(imgs)                      # (T,C,H,W)

        states = torch.tensor(self.states[start:end], dtype=torch.float32)

        if self.transform:
            imgs = self.transform(imgs)
        return imgs, states