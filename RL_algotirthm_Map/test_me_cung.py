"""
RL Visual Benchmark — Local Observation Agent  (v2 – Anti-Overfitting)
=======================================================================
THAY ĐỔI SO VỚI v1 (tất cả nhằm giảm overfitting vào Map 1):

[ENV]
1. Random start position mỗi episode (trong vùng free cells).
2. Random obstacle phase offset mỗi reset → agent không học thuộc pha cố định.
3. Domain randomization nhẹ: thêm 1–2 tường ngẫu nhiên mỗi episode (tùy chọn,
   tắt mặc định, bật bằng DOMAIN_RAND=True).

[REWARD]
4. Giảm PROGRESS_SCALE (1.20 → 0.60) để bớt phụ thuộc hướng Euclidean.
5. Bỏ RIGHT_DIR_BONUS / WRONG_DIR_PENALTY (quá bias hướng Map 1).
6. Thêm count-based exploration bonus: ô chưa thăm được +EXPLORE_BONUS.
7. LOOP_PENALTY chỉ dựa trên cửa sổ 10 bước gần nhất (không toàn cục).

[STATE]
8. goal_dir được thêm nhiễu Gaussian nhỏ khi train (augmentation) để
   agent không overfit hướng tuyệt đối.

[MODELS]
9. Thêm Dropout(0.1) vào MLP ẩn → regularization cho DQN / PolicyAgent.
10. Tăng entropy coefficient PPO/A2C (0.01 → 0.05) → khuyến khích khám phá.
11. SAC alpha tăng nhẹ (0.2 → 0.3).

[TRAINING]
12. Epsilon tối thiểu tăng (0.05 → 0.08) để DQN-family tiếp tục khám phá.
13. REINFORCE: batch accumulate 4 episode trước khi update (giảm variance).

Điều khiển:
  SPACE  — toggle nhanh / chậm
  ENTER  — bỏ qua thuật toán hiện tại
  ESC    — thoát + in bảng so sánh
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
import random
import pygame
import time
import math
from collections import deque


# ══════════════════════════════════════════════════════
#  CẤU HÌNH
# ══════════════════════════════════════════════════════
GRID       = 15
CELL       = 38
MAZE_PX    = GRID * CELL
CHART_W    = 500
TRAIN_X    = 0
TEST_X     = MAZE_PX
PANEL_X    = MAZE_PX * 2
WIN_W      = MAZE_PX * 2 + CHART_W
WIN_H      = MAZE_PX

OBS_R      = 2
OBS_SIDE   = OBS_R * 2 + 1
OBS_DIM    = OBS_SIDE * OBS_SIDE
STATE_DIM  = OBS_DIM + OBS_DIM + OBS_DIM * 2 + 2  # 102

EPISODES   = 1000
STEP_DELAY = 55
FPS        = 60

MAX_STEPS_PER_EP = GRID * GRID

# ── Reward (đã điều chỉnh) ──────────────────────────
STEP_PENALTY      = -0.02
PROGRESS_SCALE    = 0.60          # ← giảm từ 1.20 (bớt bias hướng Euclidean)
# RIGHT_DIR_BONUS / WRONG_DIR_PENALTY đã bỏ
EXPLORE_BONUS     = 0.15          # ← mới: thưởng ô chưa thăm
LOOP_PENALTY      = -0.10         # ← tăng nhẹ, chỉ dựa vào 10 bước gần nhất
NEAR_WALL_PENALTY = -0.02
NEAR_OBS_PENALTY  = -0.10
WALL_PENALTY      = -15.0
OBSTACLE_PENALTY  = -25.0
GOAL_REWARD       = 100.0
TIMEOUT_PENALTY   = -8.0
WIN_REWARD_TH     = GOAL_REWARD * 0.5

EVAL_INTERVAL_EP = 50
EVAL_EPISODES    = 5

DOMAIN_RAND      = False          # bật để thêm tường random mỗi episode

# ── Goal-dir noise khi train ─────────────────────────
GOAL_DIR_NOISE_STD = 0.05         # nhiễu Gaussian thêm vào goal_dir lúc train

# ── REINFORCE batch size ──────────────────────────────
REINFORCE_BATCH  = 4              # tích lũy N episode rồi mới update

ALL_ALGOS = [
    "REINFORCE",
    "Q-Learning", "SARSA",
    "DQN", "Double DQN", "Dueling DQN", "PER",
    "A2C", "PPO", "SAC",
]


# ══════════════════════════════════════════════════════
#  MÀU
# ══════════════════════════════════════════════════════
BG    = (12, 16, 26)
PANEL = (18, 24, 38)
CARD  = (26, 34, 52)
BDR   = (45, 60, 88)

C_WALL  = (40, 52, 72)
C_WALLE = (60, 78, 108)
C_GOAL  = (29, 158, 117)
C_AGENT = (55, 139, 215)
C_OBS   = (239, 143, 39)
C_TRAIL = (99, 179, 237)
C_FOG   = (12, 16, 26)

C_TEXT  = (215, 228, 248)
C_DIM   = (100, 120, 155)
C_GREEN = (52, 211, 153)
C_RED   = (248, 93, 93)
C_YELL  = (250, 204, 21)
C_BLUE  = (99, 179, 237)
C_ORG   = (251, 146, 60)

ALGO_COL = {
    "Q-Learning":  (129, 140, 248),
    "SARSA":       (167, 139, 250),
    "DQN":         (99, 179, 237),
    "Double DQN":  (52, 211, 153),
    "Dueling DQN": (250, 204, 21),
    "PER":         (251, 146, 60),
    "REINFORCE":   (248, 93, 93),
    "A2C":         (244, 114, 182),
    "PPO":         (34, 211, 238),
    "SAC":         (163, 230, 53),
}


# ══════════════════════════════════════════════════════
#  MÔI TRƯỜNG
# ══════════════════════════════════════════════════════
class Maze:
    def __init__(self):
        self.g = GRID
        self._base_maze = np.zeros((GRID, GRID), dtype=np.float32)

        self._base_maze[0, :] = 1
        self._base_maze[-1, :] = 1
        self._base_maze[:, 0] = 1
        self._base_maze[:, -1] = 1

        for i in range(2, 13, 3):
            self._base_maze[i, 2:10] = 1
            self._base_maze[14 - i, 5:13] = 1

        self.maze = self._base_maze.copy()
        self.start = [1, 1]
        self.goal  = [GRID - 2, GRID - 2]
        self.maze[self.goal[0], self.goal[1]] = 0

        self._obs_tmpl = [
            {"pos": [4, 2],   "dir": [0, 1],  "range": [2, 12]},
            {"pos": [10, 12], "dir": [0, -1], "range": [2, 12]},
        ]

        self.pos = list(self.start)
        self.obs = []
        self.trail = []
        self.flash = None
        self._prev_obs_pos = []
        self.steps = 0
        self.visited_local = deque(maxlen=10)   # ← chỉ 10 bước (fix #7)
        self.visited_count = {}                  # ← count-based exploration
        self.last_event = None

        # free cells cache cho random start
        self._free_cells = None

    def _compute_free_cells(self):
        cells = []
        for r in range(1, GRID - 1):
            for c in range(1, GRID - 1):
                if (self.maze[r, c] == 0
                        and [r, c] != self.goal
                        and [r, c] != self.start):
                    cells.append([r, c])
        return cells

    def reset(self, random_start=True, add_noise_walls=False):
        """
        random_start: nếu True thì chọn vị trí start ngẫu nhiên (fix #1)
        add_noise_walls: domain randomization (fix #3)
        """
        # --- Domain randomization (tùy chọn) ---
        if add_noise_walls and DOMAIN_RAND:
            self.maze = self._base_maze.copy()
            free = self._compute_free_cells()
            n_extra = random.randint(0, 3)
            for _ in range(n_extra):
                if free:
                    cell = random.choice(free)
                    self.maze[cell[0], cell[1]] = 1
                    free.remove(cell)
        else:
            self.maze = self._base_maze.copy()

        self.maze[self.goal[0], self.goal[1]] = 0

        # --- Random start position (fix #1) ---
        if random_start:
            if self._free_cells is None:
                self._free_cells = self._compute_free_cells()
            if self._free_cells:
                start_pos = random.choice(self._free_cells)
                self.pos = list(start_pos)
            else:
                self.pos = list(self.start)
        else:
            self.pos = list(self.start)

        # --- Random obstacle phase (fix #2) ---
        self.obs = []
        for o in self._obs_tmpl:
            phase_offset = random.randint(0, 8)
            new_pos = list(o["pos"])
            new_pos[1] = min(
                o["range"][1] - 1,
                max(o["range"][0] + 1, new_pos[1] + phase_offset)
            )
            self.obs.append({
                "pos": new_pos,
                "dir": list(o["dir"]),
                "range": o["range"]
            })

        self._prev_obs_pos = [list(o["pos"]) for o in self.obs]
        self.trail = [tuple(self.pos)]
        self.visited_local = deque([tuple(self.pos)], maxlen=10)
        self.visited_count = {tuple(self.pos): 1}
        self.steps = 0
        self.last_event = None
        self.flash = None
        self._free_cells = None  # reset cache khi maze thay đổi

        return self._get_state()

    def _get_state(self, add_noise=False):
        """
        add_noise: thêm nhiễu vào goal_dir khi train (fix #8)
        """
        r, c = self.pos
        gr, gc = self.goal

        static = np.zeros(OBS_DIM, dtype=np.float32)
        dyn    = np.zeros(OBS_DIM, dtype=np.float32)
        vel    = np.zeros(OBS_DIM * 2, dtype=np.float32)

        obs_positions = {tuple(o["pos"]): i for i, o in enumerate(self.obs)}
        prev_positions = self._prev_obs_pos

        for dr in range(-OBS_R, OBS_R + 1):
            for dc in range(-OBS_R, OBS_R + 1):
                nr, nc = r + dr, c + dc
                idx = (dr + OBS_R) * OBS_SIDE + (dc + OBS_R)

                if nr < 0 or nr >= GRID or nc < 0 or nc >= GRID:
                    static[idx] = 1.0
                else:
                    static[idx] = float(self.maze[nr, nc] == 1)

                if (nr, nc) in obs_positions:
                    oi = obs_positions[(nr, nc)]
                    dyn[idx] = 1.0
                    if oi < len(prev_positions):
                        pr2, pc2 = prev_positions[oi]
                        vel[idx * 2]     = float(nr - pr2)
                        vel[idx * 2 + 1] = float(nc - pc2)

        drow = gr - r
        dcol = gc - c
        dist = math.sqrt(drow ** 2 + dcol ** 2) + 1e-6
        goal_dir = np.array([drow / dist, dcol / dist], dtype=np.float32)

        # fix #8: nhiễu nhỏ vào goal_dir lúc train
        if add_noise:
            noise = np.random.normal(0, GOAL_DIR_NOISE_STD, size=2).astype(np.float32)
            goal_dir = goal_dir + noise
            norm = np.linalg.norm(goal_dir)
            if norm > 1e-6:
                goal_dir = goal_dir / norm

        return np.concatenate([static, dyn, vel, goal_dir]).astype(np.float32)

    def _dist_to_goal(self, pos=None):
        if pos is None:
            pos = self.pos
        return math.sqrt(
            (self.goal[0] - pos[0]) ** 2 +
            (self.goal[1] - pos[1]) ** 2
        )

    def _near_wall_count(self, pos):
        r, c = pos
        cnt = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= GRID or nc < 0 or nc >= GRID:
                cnt += 1
            elif self.maze[nr, nc] == 1:
                cnt += 1
        return cnt

    def _near_dynamic_obstacle(self, pos):
        r, c = pos
        for o in self.obs:
            or_, oc_ = o["pos"]
            if abs(or_ - r) + abs(oc_ - c) <= 1:
                return True
        return False

    def step(self, action, training=True):
        """
        training: nếu True, thêm noise vào state và dùng reward shaping đầy đủ
        """
        self.steps += 1

        old_pos  = list(self.pos)
        old_dist = self._dist_to_goal(old_pos)

        r, c = self.pos

        if action == 0:   r -= 1
        elif action == 1: r += 1
        elif action == 2: c -= 1
        elif action == 3: c += 1

        candidate = [r, c]
        reward = STEP_PENALTY
        done   = False
        event  = "move"

        self._prev_obs_pos = [list(o["pos"]) for o in self.obs]

        for o in self.obs:
            o["pos"][1] += o["dir"][1]
            if o["pos"][1] <= o["range"][0] or o["pos"][1] >= o["range"][1]:
                o["dir"][1] *= -1

        if self.maze[r, c] == 1:
            reward += WALL_PENALTY
            done    = True
            event   = "wall"
            self.flash = ("wall", pygame.time.get_ticks() + 350)
            candidate = old_pos

        elif any(candidate == o["pos"] for o in self.obs):
            reward += OBSTACLE_PENALTY
            done    = True
            event   = "obs"
            self.pos = candidate
            self.flash = ("obs", pygame.time.get_ticks() + 350)

        elif candidate == self.goal:
            reward += GOAL_REWARD
            done    = True
            event   = "goal"
            self.pos = candidate
            self.flash = ("goal", pygame.time.get_ticks() + 400)

        else:
            self.pos = candidate
            new_dist = self._dist_to_goal(candidate)
            progress = old_dist - new_dist

            # fix #4: PROGRESS_SCALE nhỏ hơn, bỏ RIGHT/WRONG_DIR bonus
            reward += PROGRESS_SCALE * progress

            # fix #6: exploration bonus cho ô chưa thăm
            key = tuple(candidate)
            count = self.visited_count.get(key, 0)
            if count == 0:
                reward += EXPLORE_BONUS
            self.visited_count[key] = count + 1

            # fix #7: loop penalty chỉ dựa trên 10 bước gần nhất
            if tuple(candidate) in self.visited_local:
                reward += LOOP_PENALTY

            reward += NEAR_WALL_PENALTY * self._near_wall_count(candidate)

            if self._near_dynamic_obstacle(candidate):
                reward += NEAR_OBS_PENALTY

        self.visited_local.append(tuple(self.pos))

        if not done and self.steps >= MAX_STEPS_PER_EP:
            reward += TIMEOUT_PENALTY
            done   = True
            event  = "timeout"

        self.last_event = event
        self.trail.append(tuple(self.pos))
        if len(self.trail) > 90:
            self.trail.pop(0)

        info = {
            "event":            event,
            "is_success":       event == "goal",
            "steps":            self.steps,
            "distance_to_goal": self._dist_to_goal(self.pos),
        }

        next_state = self._get_state(add_noise=(training and GOAL_DIR_NOISE_STD > 0))
        return next_state, float(reward), done, info


class GeneralizationMaze(Maze):
    def __init__(self):
        super().__init__()

        self._base_maze = np.zeros((GRID, GRID), dtype=np.float32)
        self._base_maze[0, :] = 1
        self._base_maze[-1, :] = 1
        self._base_maze[:, 0] = 1
        self._base_maze[:, -1] = 1

        self._base_maze[3, 1:11] = 1;  self._base_maze[3, 6] = 0;  self._base_maze[3, 11:14] = 0
        self._base_maze[6, 4:14] = 1;  self._base_maze[6, 10] = 0
        self._base_maze[9, 1:11] = 1;  self._base_maze[9, 3] = 0
        self._base_maze[1:8, 5] = 1;   self._base_maze[5, 5] = 0
        self._base_maze[7:14, 10] = 1; self._base_maze[11, 10] = 0

        self.maze = self._base_maze.copy()

        self.start = [1, GRID - 2]
        self.goal  = [GRID - 2, 1]
        self.maze[self.start[0], self.start[1]] = 0
        self.maze[self.goal[0],  self.goal[1]]  = 0

        self._obs_tmpl = [
            {"pos": [2, 7],  "dir": [0, 1],  "range": [6, 13]},
            {"pos": [11, 3], "dir": [0, 1],  "range": [1, 9]},
        ]

        self.pos = list(self.start)
        self.obs = []
        self.trail = []
        self.flash = None
        self._prev_obs_pos = []
        self.steps = 0
        self.visited_local = deque(maxlen=10)
        self.visited_count = {}
        self.last_event = None
        self._free_cells = None

    def reset(self, random_start=False, add_noise_walls=False):
        # Evaluation: start cố định để so sánh công bằng
        return super().reset(random_start=False, add_noise_walls=False)


# ══════════════════════════════════════════════════════
#  REPLAY BUFFER
# ══════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, sz=15000):
        self.b = deque(maxlen=sz)

    def append(self, s, a, r, ns, d):
        self.b.append((s, a, r, ns, d))

    def sample(self, bs):
        batch = random.sample(self.b, bs)
        return (
            torch.FloatTensor(np.array([x[0] for x in batch])),
            torch.LongTensor([x[1] for x in batch]),
            torch.FloatTensor([x[2] for x in batch]),
            torch.FloatTensor(np.array([x[3] for x in batch])),
            torch.FloatTensor([x[4] for x in batch]),
        )

    def __len__(self):
        return len(self.b)


class PrioritizedReplayBuffer:
    def __init__(self, sz=15000, alpha=0.6):
        self.b = []
        self.p = []
        self.sz = sz
        self.alpha = alpha
        self.pos = 0

    def append(self, s, a, r, ns, d):
        mp = max(self.p) if self.b else 1.0
        if len(self.b) < self.sz:
            self.b.append((s, a, r, ns, d))
            self.p.append(mp)
        else:
            self.b[self.pos] = (s, a, r, ns, d)
            self.p[self.pos] = mp
        self.pos = (self.pos + 1) % self.sz

    def sample(self, bs, beta=0.4):
        p = np.array(self.p)
        prob = p ** self.alpha
        prob /= prob.sum()
        idx = np.random.choice(len(self.b), bs, p=prob)
        batch = [self.b[i] for i in idx]
        w = (len(self.b) * prob[idx]) ** (-beta)
        w /= w.max()
        return (
            torch.FloatTensor(np.array([x[0] for x in batch])),
            torch.LongTensor([x[1] for x in batch]),
            torch.FloatTensor([x[2] for x in batch]),
            torch.FloatTensor(np.array([x[3] for x in batch])),
            torch.FloatTensor([x[4] for x in batch]),
            idx,
            torch.FloatTensor(w),
        )

    def update_priorities(self, idx, prios):
        for i, p in zip(idx, prios):
            self.p[i] = float(p) + 1e-6

    def __len__(self):
        return len(self.b)


# ══════════════════════════════════════════════════════
#  MODELS  (fix #9: thêm Dropout)
# ══════════════════════════════════════════════════════
def _mlp(in_dim, hidden, out_dim, dropout=0.1):
    """MLP với Dropout để tránh overfitting weights."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class VanillaQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _mlp(STATE_DIM, 128, 4)

    def forward(self, x):
        return self.net(x)


class DuelingQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(STATE_DIM, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 128),       nn.ReLU(), nn.Dropout(0.1),
        )
        self.v = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.a = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        f = self.f(x)
        v = self.v(f)
        a = self.a(f)
        return v + (a - a.mean(-1, keepdim=True))


class AC(nn.Module):
    """Actor-Critic với Dropout."""
    def __init__(self):
        super().__init__()
        self.sh = nn.Sequential(
            nn.Linear(STATE_DIM, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 128),       nn.ReLU(), nn.Dropout(0.1),
        )
        self.actor  = nn.Linear(128, 4)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        f = self.sh(x)
        return self.actor(f), self.critic(f)


# ══════════════════════════════════════════════════════
#  TABULAR AGENT
# ══════════════════════════════════════════════════════
class TabularAgent:
    def __init__(self, mode):
        self.mode      = mode
        self.q         = {}
        self.lr        = 0.1
        self.gamma     = 0.95
        self.epsilon   = 1.0
        self.eps_decay = 0.995
        self.eps_min   = 0.05

    def _key(self, state):
        return tuple((state * 4).astype(int).clip(-10, 10))

    def _q(self, key):
        if key not in self.q:
            self.q[key] = np.zeros(4)
        return self.q[key]

    def act(self, state, **_):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        k = self._key(state)
        return int(np.argmax(self._q(k)))

    def update(self, s, a, r, ns, done, na=None):
        k  = self._key(s)
        kn = self._key(ns)
        if self.mode == "Q-Learning":
            tgt = r + self.gamma * np.max(self._q(kn)) * (1 - done)
        else:
            tgt = r + self.gamma * self._q(kn)[na] * (1 - done)
        self._q(k)[a] += self.lr * (tgt - self._q(k)[a])
        if done and self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay


# ══════════════════════════════════════════════════════
#  DQN AGENTS  (fix #12: eps_min = 0.08)
# ══════════════════════════════════════════════════════
class DQNAgent:
    def __init__(self, mode):
        self.mode = mode

        Cls = DuelingQ if mode == "Dueling DQN" else VanillaQ
        self.net = Cls()
        self.tgt = Cls()
        self.tgt.load_state_dict(self.net.state_dict())

        self.opt = optim.Adam(self.net.parameters(), lr=3e-4)

        self.buf     = PrioritizedReplayBuffer() if mode == "PER" else ReplayBuffer()
        self.gamma   = 0.95
        self.epsilon = 1.0
        self.eps_decay = 0.997
        self.eps_min   = 0.08     # ← tăng từ 0.05 (tiếp tục khám phá)
        self._step   = 0

    def act(self, state, **_):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        # eval mode để tắt dropout khi chọn action
        self.net.eval()
        with torch.no_grad():
            q = self.net(torch.FloatTensor(state))
            a = int(torch.argmax(q).item())
        self.net.train()
        return a

    def update(self, bs=64):
        if len(self.buf) < bs:
            return

        if self.mode == "PER":
            st, ac, rw, ns, dn, idx, w = self.buf.sample(bs)
        else:
            st, ac, rw, ns, dn = self.buf.sample(bs)
            w = torch.ones(bs)

        self.net.train()
        cq = self.net(st).gather(1, ac.unsqueeze(1)).squeeze(1)

        self.net.eval()
        with torch.no_grad():
            if self.mode == "Double DQN":
                ba = self.net(ns).max(1)[1].unsqueeze(1)
                mq = self.tgt(ns).gather(1, ba).squeeze(1)
            else:
                mq = self.tgt(ns).max(1)[0]
            tq = rw + self.gamma * mq * (1 - dn)
        self.net.train()

        td   = torch.abs(cq - tq)
        loss = (w * td ** 2).mean()

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()

        if self.mode == "PER":
            self.buf.update_priorities(idx, td.detach().numpy())

        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay

        self._step += 1
        if self._step % 200 == 0:
            self.tgt.load_state_dict(self.net.state_dict())


# ══════════════════════════════════════════════════════
#  POLICY AGENT  (fix #10,#11,#13)
# ══════════════════════════════════════════════════════
class PolicyAgent:
    def __init__(self, mode):
        self.mode = mode
        self.net  = AC()

        lr = 1e-4 if mode == "REINFORCE" else 3e-4
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        self.gamma    = 0.95
        self.ppo_clip = 0.2
        self.alpha    = 0.3      # ← SAC alpha tăng (fix #11)

        # fix #13: REINFORCE batch
        self.traj      = []
        self._rein_buf = []      # tích lũy nhiều episode
        self.epsilon   = 0.0
        self.grad_clip = 1.0
        self.last_loss = 0.0

    def _state_tensor(self, state):
        x = torch.FloatTensor(state).view(1, -1)
        return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

    def _safe_logits(self, logits):
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        return torch.clamp(logits, -20.0, 20.0)

    def _optimizer_step(self, loss):
        if not torch.isfinite(loss):
            return False
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.grad_clip)
        self.opt.step()
        self.last_loss = float(loss.item())
        return True

    def act(self, state, **_):
        x = self._state_tensor(state)
        # QUAN TRỌNG: KHÔNG dùng no_grad() ở đây.
        # log_prob phải giữ grad_fn để loss.backward() hoạt động
        # với REINFORCE / A2C / PPO. Dropout vẫn chạy ở train mode.
        self.net.train()
        logits, _ = self.net(x)
        logits = self._safe_logits(logits.squeeze(0))
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    # ── REINFORCE (batch version, fix #13) ──────────
    def push_reinforce_episode(self, traj):
        """Tích lũy 1 episode vào buffer."""
        self._rein_buf.append(traj)

    def upd_reinforce(self, force=False):
        """Update chỉ khi đủ REINFORCE_BATCH episodes (hoặc force=True)."""
        if len(self._rein_buf) < REINFORCE_BATCH and not force:
            return 0.0
        if not self._rein_buf:
            return 0.0

        all_log_probs = []
        all_returns   = []

        for traj in self._rein_buf:
            R = 0.0
            returns = []
            for _, _, reward, _ in reversed(traj):
                reward = float(reward) if np.isfinite(float(reward)) else 0.0
                R = reward + self.gamma * R
                returns.insert(0, R)

            returns = torch.tensor(returns, dtype=torch.float32)
            returns = torch.nan_to_num(returns, nan=0.0, posinf=1.0, neginf=-1.0)

            if returns.numel() > 1:
                mean = returns.mean()
                std  = returns.std(unbiased=False)
                if torch.isfinite(std) and std.item() > 1e-8:
                    returns = (returns - mean) / (std + 1e-8)

            log_probs = torch.stack([lp for _, lp, _, _ in traj])
            min_len   = min(len(log_probs), len(returns))
            all_log_probs.append(log_probs[:min_len])
            all_returns.append(returns[:min_len])

        all_log_probs = torch.cat(all_log_probs)
        all_returns   = torch.cat(all_returns)

        loss = -(all_log_probs * all_returns).sum()
        updated = self._optimizer_step(loss)
        self._rein_buf.clear()
        return self.last_loss if updated else 0.0

    # ── A2C (fix #10: entropy coef 0.05) ────────────
    def upd_a2c(self, s, lp, r, ns, done):
        s_t  = self._state_tensor(s)
        ns_t = self._state_tensor(ns)

        # 1 forward pass duy nhất cho s → lấy cả v lẫn logits để tính entropy
        logits, v = self.net(s_t)
        logits = self._safe_logits(logits)
        dist   = Categorical(logits=logits.squeeze(0))

        with torch.no_grad():
            _, nv = self.net(ns_t)
            target = torch.tensor([[float(r)]], dtype=torch.float32) \
                     + self.gamma * nv * (1.0 - float(done))

        adv          = target - v
        actor_loss   = -lp * adv.detach().squeeze()
        critic_loss  = F.mse_loss(v, target)
        entropy_loss = -0.05 * dist.entropy()

        loss = actor_loss + 0.5 * critic_loss + entropy_loss
        self._optimizer_step(loss)

    # ── PPO (fix #10: entropy 0.05) ─────────────────
    def upd_ppo(self, st, at, lpt, ret, adv):
        for _ in range(4):
            logits, v  = self.net(st)
            logits     = self._safe_logits(logits)
            dist       = Categorical(logits=logits)
            new_log_prob = dist.log_prob(at)

            ratio  = torch.exp(new_log_prob - lpt)
            surr1  = ratio * adv
            surr2  = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * adv

            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(v.squeeze(-1), ret)
            entropy_loss = -0.05 * dist.entropy().mean()   # ← 0.01 → 0.05

            loss = actor_loss + 0.5 * critic_loss + entropy_loss
            self._optimizer_step(loss)

    # ── SAC (fix #11: alpha=0.3) ─────────────────────
    def upd_sac(self, s, a, r, ns, done):
        s_t  = self._state_tensor(s)
        ns_t = self._state_tensor(ns)

        logits,      v  = self.net(s_t)
        next_logits, nv = self.net(ns_t)
        logits      = self._safe_logits(logits)
        next_logits = self._safe_logits(next_logits)

        prob      = F.softmax(logits, dim=-1)
        log_prob  = F.log_softmax(logits, dim=-1)
        next_prob = F.softmax(next_logits, dim=-1)
        next_log  = F.log_softmax(next_logits, dim=-1)

        with torch.no_grad():
            next_value = (next_prob * (nv - self.alpha * next_log)).sum(dim=-1)
            target     = torch.tensor([float(r)], dtype=torch.float32) \
                         + self.gamma * next_value * (1.0 - float(done))

        value_loss  = F.mse_loss(v.squeeze(-1), target)
        policy_loss = (prob * (self.alpha * log_prob - v.detach())).sum(dim=-1).mean()

        self._optimizer_step(value_loss + policy_loss)


# ══════════════════════════════════════════════════════
#  RENDERER  (không đổi)
# ══════════════════════════════════════════════════════
class Renderer:
    def __init__(self, screen):
        self.sc = screen
        self.fn = pygame.font.SysFont("Segoe UI", 12)
        self.fb = pygame.font.SysFont("Segoe UI", 14, bold=True)
        self.fl = pygame.font.SysFont("Segoe UI", 18, bold=True)

    def draw_maze(self, env, acol, show_obs_window=True, x0=0,
                  title=None, subtitle=None, eval_active=False):
        s = self.sc
        t = pygame.time.get_ticks()

        pygame.draw.rect(s, BG, (x0, 0, MAZE_PX, WIN_H))

        fog = pygame.Surface((MAZE_PX, WIN_H), pygame.SRCALPHA)
        fog.fill((*BG, 180))

        pr, pc = env.pos
        for dr in range(-OBS_R, OBS_R + 1):
            for dc in range(-OBS_R, OBS_R + 1):
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < GRID and 0 <= nc < GRID:
                    pygame.draw.rect(fog, (0, 0, 0, 0),
                                     pygame.Rect(nc * CELL, nr * CELL, CELL, CELL))

        for r in range(GRID):
            for c in range(GRID):
                rect = pygame.Rect(x0 + c * CELL, r * CELL, CELL, CELL)
                if env.maze[r, c] == 1:
                    pygame.draw.rect(s, C_WALL,  rect)
                    pygame.draw.rect(s, C_WALLE, rect, 1)
                elif [r, c] == env.goal:
                    self._draw_goal(s, rect, t)
                else:
                    pygame.draw.rect(s, (16, 22, 34), rect)
                    pygame.draw.rect(s, (22, 30, 46), rect, 1)

        ts = pygame.Surface((MAZE_PX, WIN_H), pygame.SRCALPHA)
        n  = len(env.trail)
        for i, (tr, tc) in enumerate(env.trail):
            a_ = int(90 * i / max(1, n - 1))
            pygame.draw.circle(ts, (*C_TRAIL, a_),
                               (tc * CELL + CELL // 2, tr * CELL + CELL // 2),
                               max(2, CELL // 8))
        s.blit(ts, (x0, 0))

        for o in env.obs:
            self._draw_obstacle(s, o, t, x0=x0)
        self._draw_agent(s, env.pos, acol, t, x0=x0)
        s.blit(fog, (x0, 0))

        if show_obs_window:
            ox = x0 + (pc - OBS_R) * CELL
            oy = (pr - OBS_R) * CELL
            ow = OBS_SIDE * CELL
            obs_s = pygame.Surface((ow, ow), pygame.SRCALPHA)
            pygame.draw.rect(obs_s, (*acol, 25), (0, 0, ow, ow))
            pygame.draw.rect(obs_s, (*acol, 180), (0, 0, ow, ow), 2)
            s.blit(obs_s, (ox, oy))

        if env.flash:
            kind, end = env.flash
            rem = end - t
            if rem > 0:
                alpha = int(130 * rem / 350)
                col   = C_RED if kind == "wall" else C_ORG if kind == "obs" else C_GREEN
                fs    = pygame.Surface((MAZE_PX, WIN_H), pygame.SRCALPHA)
                fs.fill((*col, alpha))
                s.blit(fs, (x0, 0))
            else:
                env.flash = None

        if title:
            badge = pygame.Surface((MAZE_PX, 34), pygame.SRCALPHA)
            badge.fill((0, 0, 0, 120))
            s.blit(badge, (x0, 0))
            col = C_YELL if eval_active else acol
            s.blit(self.fb.render(title,    True, col),    (x0 + 10, 4))
            if subtitle:
                s.blit(self.fn.render(subtitle, True, C_TEXT), (x0 + 10, 19))

        pygame.draw.line(s, BDR,
                         (x0 + MAZE_PX - 1, 0),
                         (x0 + MAZE_PX - 1, WIN_H), 2)

    def _draw_goal(self, s, rect, t):
        pygame.draw.rect(s, (10, 35, 25), rect)
        pulse = abs(math.sin(t / 700))
        gs = pygame.Surface((CELL + 8, CELL + 8), pygame.SRCALPHA)
        pygame.draw.rect(gs, (*C_GOAL, int(35 + 25 * pulse)),
                         (0, 0, CELL + 8, CELL + 8), border_radius=5)
        s.blit(gs, (rect.x - 4, rect.y - 4))
        pygame.draw.rect(s, C_GOAL, rect, border_radius=4)
        mx, my = rect.centerx, rect.centery
        pygame.draw.line(s, (10, 40, 25), (mx - 6, my),    (mx - 1, my + 5), 3)
        pygame.draw.line(s, (10, 40, 25), (mx - 1, my + 5), (mx + 7, my - 5), 3)

    def _draw_obstacle(self, s, o, t, x0=0):
        r_, c_ = o["pos"]
        rect   = pygame.Rect(x0 + c_ * CELL + 2, r_ * CELL + 2, CELL - 4, CELL - 4)
        pulse  = abs(math.sin(t / 380))
        gs     = pygame.Surface((CELL + 6, CELL + 6), pygame.SRCALPHA)
        pygame.draw.rect(gs, (*C_OBS, int(25 + 35 * pulse)),
                         (0, 0, CELL + 6, CELL + 6), border_radius=5)
        s.blit(gs, (x0 + c_ * CELL - 3, r_ * CELL - 3))
        pygame.draw.rect(s, C_OBS, rect, border_radius=5)
        dx, dy = o["dir"][1], o["dir"][0]
        mx, my = rect.centerx, rect.centery
        pygame.draw.line(s, (30, 20, 10),
                         (mx - dx * 6, my - dy * 6),
                         (mx + dx * 6, my + dy * 6), 2)

    def _draw_agent(self, s, pos, col, t, x0=0):
        ar, ac = pos
        cx = x0 + ac * CELL + CELL // 2
        cy = ar * CELL + CELL // 2
        pulse = abs(math.sin(t / 280))
        rad   = int(CELL // 2 - 4 + 2 * pulse)
        gs    = pygame.Surface((CELL * 2, CELL * 2), pygame.SRCALPHA)
        for d in [10, 6, 3]:
            pygame.draw.circle(gs, (*col, 14), (CELL, CELL), rad + d)
        s.blit(gs, (cx - CELL, cy - CELL))
        pygame.draw.circle(s, col, (cx, cy), rad)
        pygame.draw.circle(s, (200, 230, 255), (cx - 3, cy - 3), rad // 3)

    def draw_obs_panel(self, s, state, env, acol, x, y, w, h):
        pygame.draw.rect(s, CARD, (x, y, w, h), border_radius=6)
        pygame.draw.rect(s, BDR,  (x, y, w, h), 1, border_radius=6)
        s.blit(self.fn.render("Cửa sổ quan sát (5×5)", True, C_DIM), (x + 6, y + 5))

        static_layer = state[:OBS_DIM].reshape(OBS_SIDE, OBS_SIDE)
        dyn_layer    = state[OBS_DIM:OBS_DIM * 2].reshape(OBS_SIDE, OBS_SIDE)
        vel_layer    = state[OBS_DIM * 2:OBS_DIM * 4].reshape(OBS_SIDE, OBS_SIDE, 2)

        cell_px = (min(w, h) - 28) // OBS_SIDE
        ox = x + (w - cell_px * OBS_SIDE) // 2
        oy = y + 20

        for r in range(OBS_SIDE):
            for c in range(OBS_SIDE):
                rx = ox + c * cell_px
                ry = oy + r * cell_px
                cr = pygame.Rect(rx, ry, cell_px - 1, cell_px - 1)

                if r == OBS_R and c == OBS_R:
                    pygame.draw.rect(s, (*acol, 180), cr, border_radius=2)
                elif static_layer[r, c] > 0.5:
                    pygame.draw.rect(s, C_WALL, cr, border_radius=2)
                elif dyn_layer[r, c] > 0.5:
                    pygame.draw.rect(s, C_OBS, cr, border_radius=2)
                    vr = vel_layer[r, c, 0]
                    vc = vel_layer[r, c, 1]
                    mx_ = rx + cell_px // 2
                    my_ = ry + cell_px // 2
                    if abs(vr) + abs(vc) > 0:
                        ex = int(mx_ + vc * cell_px * 0.35)
                        ey = int(my_ + vr * cell_px * 0.35)
                        pygame.draw.line(s, (30, 20, 5), (mx_, my_), (ex, ey), 2)
                else:
                    pygame.draw.rect(s, (22, 32, 48), cr, border_radius=2)

    def draw_charts(self, algo, acol, ep, total_ep, ep_rewards, wins, losses,
                    epsilon, fast_mode, algo_idx, n_algos, state, env,
                    eval_history=None, eval_stats=None, eval_active=False):
        s  = self.sc
        px = PANEL_X
        iw = CHART_W - 20
        eval_history = eval_history or []

        pygame.draw.rect(s, PANEL, (px, 0, CHART_W, WIN_H))
        pygame.draw.line(s, BDR, (px, 0), (px, WIN_H), 2)

        y = 10
        pygame.draw.rect(s, CARD, (px + 6, y, iw, 52), border_radius=7)
        pygame.draw.rect(s, acol, (px + 6, y, iw, 52), 2, border_radius=7)
        s.blit(self.fn.render(f"[{algo_idx+1}/{n_algos}]  Train + Generalization Test",
                               True, C_DIM), (px + 12, y + 5))
        s.blit(self.fl.render(algo, True, acol), (px + 12, y + 22))
        if fast_mode:
            s.blit(self.fn.render("FAST",         True, C_YELL), (px + iw - 55,  y + 5))
        if eval_active:
            s.blit(self.fn.render("TESTING MAP-2", True, C_YELL), (px + iw - 115, y + 22))
        y += 60

        pct = ep / max(1, total_ep)
        pygame.draw.rect(s, CARD, (px + 6, y, iw, 18), border_radius=4)
        if pct > 0:
            pygame.draw.rect(s, (*acol, 150),
                              (px + 6, y, int(iw * pct), 18), border_radius=4)
        s.blit(self.fn.render(f"Episode {ep}/{total_ep}", True, C_TEXT), (px + 10, y + 2))
        s.blit(self.fn.render(f"{pct*100:.0f}%", True, C_DIM), (px + iw - 32, y + 2))
        y += 26

        total_done = wins + losses
        train_wr   = wins / max(1, total_done) * 100
        ep_r       = ep_rewards[-1] if ep_rewards else 0
        avg_r      = np.mean(ep_rewards[-50:]) if len(ep_rewards) >= 5 else 0

        if eval_history:
            last      = eval_history[-1]
            test_wr   = last["success_rate"] * 100
            test_dist = last["avg_final_dist"]
            gap       = train_wr - test_wr
            test_wr_txt = f"{test_wr:.1f}%"
            gap_txt     = f"{gap:+.1f}%"
            dist_txt    = f"{test_dist:.1f}"
        else:
            test_wr = test_dist = None
            test_wr_txt = gap_txt = dist_txt = "N/A"

        stats = [
            ("Train EP",   f"{ep_r:+.1f}",    C_BLUE if ep_r >= 0 else C_RED),
            ("Train Avg50",f"{avg_r:+.1f}",   C_GREEN if avg_r > 0 else C_ORG),
            ("Train WR",   f"{train_wr:.1f}%", C_GREEN if train_wr > 50 else C_YELL if train_wr > 20 else C_RED),
            ("Test WR",    test_wr_txt,         C_GREEN if test_wr is not None and test_wr > 50 else C_YELL if test_wr is not None else C_DIM),
            ("Gen Gap",    gap_txt,             C_RED if gap_txt != "N/A" and (train_wr - (test_wr or 0)) > 30 else C_YELL if gap_txt != "N/A" else C_DIM),
            ("Test Dist↓", dist_txt,            C_GREEN if test_dist is not None and test_dist < 3 else C_ORG if test_dist is not None else C_DIM),
            ("Thắng",      str(wins),           C_GREEN),
            ("Thua",       str(losses),         C_RED),
            ("ε",          f"{epsilon:.3f}",   C_ORG),
        ]

        sw = iw // 3
        for i, (lbl, val, col) in enumerate(stats):
            cx_ = px + 6 + (i % 3) * sw
            cy_ = y + (i // 3) * 42
            sc  = pygame.Rect(cx_, cy_, sw - 3, 38)
            pygame.draw.rect(s, CARD, sc, border_radius=4)
            pygame.draw.rect(s, BDR,  sc, 1, border_radius=4)
            s.blit(self.fn.render(lbl, True, C_DIM), (cx_ + 5, cy_ + 4))
            s.blit(self.fb.render(val, True, col),   (cx_ + 5, cy_ + 18))

        y += math.ceil(len(stats) / 3) * 42 + 4

        self._linechart(s, px+6, y, iw, 105, "Train reward / episode",
                        ep_rewards, acol, show_zero=True, fill=True)
        y += 112

        if ep_rewards:
            wins_arr = [1 if r >= WIN_REWARD_TH else 0 for r in ep_rewards]
            rolled   = []
            for i in range(len(wins_arr)):
                st = max(0, i - 19)
                rolled.append(sum(wins_arr[st:i+1]) / (i - st + 1) * 100)
        else:
            rolled = []

        self._linechart(s, px+6, y, iw, 86, "Train win rate % rolling 20ep",
                        rolled, C_GREEN, y_min=0, y_max=100)
        y += 94

        eval_wr_series = [x["success_rate"] * 100 for x in eval_history]
        self._linechart(s, px+6, y, iw, 76,
                        f"Map-2 test WR% mỗi {EVAL_INTERVAL_EP} ep",
                        eval_wr_series, C_YELL, y_min=0, y_max=100)
        y += 84

        obs_h = WIN_H - y - 40
        if obs_h >= 78:
            self.draw_obs_panel(s, state, env, acol, px+6, y, iw, obs_h)
            y += obs_h + 4

        for key, desc in [("SPACE","Nhanh/Chậm"),("ENTER","Bỏ qua"),("ESC","Thoát")]:
            ks = self.fn.render(f"[{key}]", True, C_YELL)
            ds = self.fn.render(desc,       True, C_DIM)
            s.blit(ks, (px + 10, y))
            s.blit(ds, (px + 10 + ks.get_width() + 4, y))
            y += 16

    def _linechart(self, surf, x, y, w, h, title, data, col,
                   y_min=None, y_max=None, show_zero=False, fill=False):
        pygame.draw.rect(surf, CARD, (x, y, w, h), border_radius=5)
        pygame.draw.rect(surf, BDR,  (x, y, w, h), 1, border_radius=5)
        surf.blit(self.fn.render(title, True, C_DIM), (x + 6, y + 4))

        pl, pt, pr_, pb = 8, 20, 6, 16
        cx0, cy0 = x + pl, y + pt
        cw, ch   = w - pl - pr_, h - pt - pb

        if not data or len(data) < 2:
            surf.blit(self.fn.render("Đang thu thập...", True, C_DIM),
                      (cx0 + 4, cy0 + ch // 2 - 6))
            return

        mn  = y_min if y_min is not None else min(data)
        mx_ = y_max if y_max is not None else max(data)
        rng = mx_ - mn if mx_ != mn else 1

        def py_(v):
            return int(cy0 + ch - (v - mn) / rng * ch)

        if show_zero and mn < 0 < mx_:
            zy = py_(0)
            pygame.draw.line(surf, (55, 70, 100), (cx0, zy), (cx0 + cw, zy), 1)

        for frac in [0.25, 0.5, 0.75, 1.0]:
            ly = cy0 + int((1 - frac) * ch)
            pygame.draw.line(surf, BDR, (cx0, ly), (cx0 + cw, ly), 1)
            surf.blit(self.fn.render(f"{mn + frac * rng:.0f}", True, C_DIM), (x + 1, ly - 6))

        pts_data = data[-200:] if len(data) > 200 else data
        n = len(pts_data)
        pts = []
        for i, v in enumerate(pts_data):
            ppx = cx0 + int(i / (n - 1) * cw)
            ppy = max(cy0, min(cy0 + ch, py_(v)))
            pts.append((ppx, ppy))

        if fill and len(pts) >= 2:
            poly = [(pts[0][0], cy0 + ch)] + pts + [(pts[-1][0], cy0 + ch)]
            fs   = pygame.Surface((surf.get_width(), surf.get_height()), pygame.SRCALPHA)
            pygame.draw.polygon(fs, (*col, 28), poly)
            surf.blit(fs, (0, 0))

        if len(pts) >= 2:
            pygame.draw.lines(surf, col, False, pts, 2)
        if pts:
            pygame.draw.circle(surf, col,             pts[-1], 4)
            pygame.draw.circle(surf, (255, 255, 255), pts[-1], 2)


# ══════════════════════════════════════════════════════
#  BẢNG KẾT QUẢ TERMINAL
# ══════════════════════════════════════════════════════
def print_table(results):
    RST = "\033[0m"; B = "\033[1m"; DIM = "\033[2m"
    G = "\033[92m";  R = "\033[91m"; C = "\033[96m"; W = "\033[97m"

    cols = ["Train WR","Test WR","Gen Gap","Avg Reward","Best","Avg Steps","Time(s)"]
    cw   = [10, 9, 9, 12, 10, 11, 9]
    aw   = 14

    def sep(l, m, ri, f="─"):
        return l + (f * (aw + 2)) + m + "".join(f * (w + 2) + m for w in cw) + ri

    print(f"\n{B}{C}{'═' * 112}{RST}")
    print(f"  {B}{W}RL BENCHMARK — TRAIN MAP vs TEST MAP-2 GENERALIZATION (v2){RST}")
    print(f"{B}{C}{'═' * 112}{RST}\n")
    print(sep("  ┌","┬","┐"))
    print(f"{B}  │ {'ALGORITHM':<{aw}} │" +
          "".join(f" {c:^{w}} │" for c, w in zip(cols, cw)) + f"{RST}")
    print(sep("  ├","┼","┤"))

    for algo, res in results.items():
        row  = f"  │ {B}{C}{algo:<{aw}}{RST} │"
        vals = [
            f"{res.get('wr',0)*100:.1f}%",
            f"{res.get('test_wr',0)*100:.1f}%",
            f"{res.get('gen_gap',0)*100:+.1f}%",
            f"{res.get('avg_r',0):+.2f}",
            f"{res.get('best',0):+.1f}",
            f"{res.get('steps',0):.1f}",
            f"{res.get('time',0):.1f}s",
        ]
        for txt, w in zip(vals, cw):
            row += f" {W}{txt:^{w}}{RST} │"
        print(row)

    print(sep("  └","┴","┘"))
    print(f"\n  {B}Train WR cao nhưng Test WR thấp ⇒ overfitting vào map train.{RST}\n")


# ══════════════════════════════════════════════════════
#  TRAINING HELPERS
# ══════════════════════════════════════════════════════
def make_agent(algo):
    if algo in ["Q-Learning","SARSA"]:
        return TabularAgent(algo)
    if algo in ["DQN","Double DQN","Dueling DQN","PER"]:
        return DQNAgent(algo)
    return PolicyAgent(algo)


def eval_action(agent, algo, state):
    if algo in ["Q-Learning","SARSA"]:
        k = agent._key(state)
        return int(np.argmax(agent._q(k)))
    if algo in ["DQN","Double DQN","Dueling DQN","PER"]:
        agent.net.eval()
        with torch.no_grad():
            q = agent.net(torch.FloatTensor(state))
            a = int(torch.argmax(q).item())
        agent.net.train()
        return a
    agent.net.eval()
    with torch.no_grad():
        x = torch.FloatTensor(state).view(1, -1)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        logits, _ = agent.net(x)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        logits = torch.clamp(logits, -20.0, 20.0)
        a = int(torch.argmax(logits.squeeze(0)).item())
    agent.net.train()
    return a


def handle_pygame_controls():
    global FAST_MODE
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            return "quit"
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:  return "quit"
            if ev.key == pygame.K_SPACE:   FAST_MODE = not FAST_MODE
            if ev.key == pygame.K_RETURN:  return "skip"
    return None


def render_dashboard(rend, algo, acol, ep, total_ep, ep_rewards, wins, losses,
                     epsilon, algo_idx, n_algos, train_state, train_env,
                     test_env, eval_history, eval_active=False):
    train_sub = f"Train ep {ep}/{total_ep}"
    if eval_history:
        last      = eval_history[-1]
        test_sub  = (f"Last test@ep{last['ep']}: "
                     f"WR={last['success_rate']*100:.1f}% "
                     f"Dist={last['avg_final_dist']:.1f}")
    else:
        test_sub = f"Chưa test — mỗi {EVAL_INTERVAL_EP} ep"

    rend.draw_maze(train_env, acol, x0=TRAIN_X,
                   title="TRAIN MAP", subtitle=train_sub, eval_active=False)
    rend.draw_maze(test_env, C_YELL if eval_active else C_BLUE, x0=TEST_X,
                   title="TEST MAP-2 / GENERALIZATION", subtitle=test_sub,
                   eval_active=eval_active)
    rend.draw_charts(algo, acol, ep, total_ep, ep_rewards, wins, losses,
                     epsilon, FAST_MODE, algo_idx, n_algos, train_state, train_env,
                     eval_history=eval_history, eval_active=eval_active)
    pygame.display.flip()


def run_generalization_eval(agent, algo, ep, train_env, test_env,
                             screen, rend, clock, acol, algo_idx,
                             ep_rewards, wins, losses, train_state, eval_history):
    rewards = []; final_dists = []; step_counts = []; successes = 0

    for k in range(EVAL_EPISODES):
        # eval: start cố định, không noise
        state = test_env.reset(random_start=False)
        done  = False
        ep_r  = 0.0
        steps = 0
        info  = {"is_success": False, "distance_to_goal": test_env._dist_to_goal()}

        while not done and steps < MAX_STEPS_PER_EP:
            ctrl = handle_pygame_controls()
            if ctrl == "quit":  return None, True, False
            if ctrl == "skip":  return None, False, True

            action = eval_action(agent, algo, state)
            # eval: training=False → không noise
            state, r, done, info = test_env.step(action, training=False)
            ep_r  += r
            steps += 1

            if not FAST_MODE:
                eps_v = getattr(agent, "epsilon", 0.0)
                render_dashboard(rend, algo, acol, ep, EPISODES, ep_rewards,
                                 wins, losses, eps_v, algo_idx, len(ALL_ALGOS),
                                 train_state, train_env, test_env, eval_history,
                                 eval_active=True)
                pygame.time.delay(max(10, STEP_DELAY // 2))
                clock.tick(FPS)

        rewards.append(ep_r)
        final_dists.append(float(info.get("distance_to_goal", test_env._dist_to_goal())))
        step_counts.append(steps)
        successes += int(bool(info.get("is_success", False)))

    stats = {
        "ep":            ep,
        "success_rate":  successes / max(1, EVAL_EPISODES),
        "avg_reward":    float(np.mean(rewards))     if rewards     else 0.0,
        "avg_steps":     float(np.mean(step_counts)) if step_counts else 0.0,
        "avg_final_dist":float(np.mean(final_dists)) if final_dists else 0.0,
    }
    eval_history.append(stats)
    return stats, False, False


# ══════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════
def run_algo(algo, algo_idx, screen, rend, clock):
    global FAST_MODE
    FAST_MODE = False

    acol     = ALGO_COL[algo]
    env      = Maze()
    test_env = GeneralizationMaze()
    test_env.reset(random_start=False)

    agent      = make_agent(algo)
    ep_rewards = []
    eval_history = []
    ep_steps   = []
    wins = losses = 0
    best = -999
    conv = EPISODES
    win_win  = deque(maxlen=50)
    start_t  = time.time()
    skip     = False
    last_state = env.reset(random_start=True, add_noise_walls=True)

    pygame.display.set_caption(
        f"RL Benchmark v2 — {algo} [{algo_idx+1}/{len(ALL_ALGOS)}]")
    print(f"\n{'─'*60}")
    print(f"  [{algo_idx+1:02d}/{len(ALL_ALGOS)}]  {algo}")

    for ep in range(1, EPISODES + 1):
        # fix #1,#3: random start + domain rand mỗi episode
        state = env.reset(random_start=True, add_noise_walls=True)
        done  = False
        ep_r  = 0.0
        steps = 0
        ppo_s = []; ppo_a = []; ppo_lp = []; ppo_r = []
        info  = {"event": None, "is_success": False}

        if algo in ["Q-Learning","SARSA"]:
            action = agent.act(state)

        while not done:
            ctrl = handle_pygame_controls()
            if ctrl == "quit":  return None, skip
            if ctrl == "skip":  skip = True; done = True; break
            if skip:            break

            if algo in ["Q-Learning","SARSA"]:
                # training=True → noise on
                ns, r, done, info = env.step(action, training=True)
                na = agent.act(ns)
                agent.update(state, action, r, ns, float(done), na)
                state  = ns
                action = na

            elif algo in ["DQN","Double DQN","Dueling DQN","PER"]:
                action = agent.act(state)
                ns, r, done, info = env.step(action, training=True)
                agent.buf.append(state, action, r, ns, float(done))
                agent.update()
                state = ns

            else:  # policy agents
                action, lp = agent.act(state)
                ns, r, done, info = env.step(action, training=True)

                if algo == "REINFORCE":
                    agent.traj.append((state, lp, r, ns))
                elif algo == "A2C":
                    agent.upd_a2c(state, lp, r, ns, float(done))
                elif algo == "PPO":
                    ppo_s.append(state); ppo_a.append(action)
                    ppo_lp.append(lp.item()); ppo_r.append(r)
                elif algo == "SAC":
                    agent.upd_sac(state, action, r, ns, float(done))

                state = ns

            ep_r  += r
            steps += 1
            last_state = state

            if not FAST_MODE:
                eps_v = getattr(agent, "epsilon", 0.0)
                render_dashboard(rend, algo, acol, ep, EPISODES, ep_rewards,
                                 wins, losses, eps_v, algo_idx, len(ALL_ALGOS),
                                 state, env, test_env, eval_history, eval_active=False)
                pygame.time.delay(STEP_DELAY)
                clock.tick(FPS)

        if skip:
            break

        # ── Post-episode updates ──────────────────────
        if algo == "REINFORCE" and agent.traj:
            # fix #13: push episode, update nếu đủ batch
            agent.push_reinforce_episode(agent.traj)
            agent.traj = []
            agent.upd_reinforce(force=(ep == EPISODES))

        elif algo == "PPO" and ppo_s:
            st_  = torch.FloatTensor(np.array(ppo_s))
            at_  = torch.LongTensor(ppo_a)
            lpt_ = torch.FloatTensor(ppo_lp)

            agent.net.eval()
            with torch.no_grad():
                _, vs = agent.net(st_)
            agent.net.train()
            vs = vs.squeeze(-1).numpy()

            G = 0.0; ret_ = []
            for rv in reversed(ppo_r):
                G = rv + agent.gamma * G
                ret_.insert(0, G)

            ret_  = np.array(ret_, dtype=np.float32)
            adv_  = ret_ - vs
            if len(adv_) > 1:
                adv_std = adv_.std()
                if adv_std > 1e-8:
                    adv_ = (adv_ - adv_.mean()) / (adv_std + 1e-8)

            agent.upd_ppo(st_, at_, lpt_,
                          torch.FloatTensor(ret_),
                          torch.FloatTensor(adv_))

        won = bool(info.get("is_success", False))
        if won:  wins   += 1
        else:    losses += 1
        if ep_r > best: best = ep_r

        ep_rewards.append(ep_r)
        ep_steps.append(steps)
        win_win.append(1 if won else 0)

        if conv == EPISODES and len(win_win) == 50 and sum(win_win) / 50 >= 0.5:
            conv = ep

        eps_v = getattr(agent, "epsilon", 0.0)
        render_dashboard(rend, algo, acol, ep, EPISODES, ep_rewards,
                         wins, losses, eps_v, algo_idx, len(ALL_ALGOS),
                         last_state, env, test_env, eval_history, eval_active=False)
        clock.tick(FPS)

        if ep % EVAL_INTERVAL_EP == 0:
            stats, quit_eval, skip_eval = run_generalization_eval(
                agent, algo, ep, env, test_env, screen, rend, clock,
                acol, algo_idx, ep_rewards, wins, losses, last_state, eval_history)
            if quit_eval:  return None, skip
            if skip_eval:  skip = True; break

        if ep % 50 == 0:
            avg = np.mean(ep_rewards[-50:])
            wr  = wins / ep * 100
            if eval_history:
                te  = eval_history[-1]
                gap = wr - te["success_rate"] * 100
                print(f"  Ep {ep:>4}  TrainAvgR={avg:+6.2f}  TrainWR={wr:5.1f}%  "
                      f"TestWR={te['success_rate']*100:5.1f}%  Gap={gap:+5.1f}%  "
                      f"TestDist={te['avg_final_dist']:.2f}  ε={eps_v:.3f}")
            else:
                print(f"  Ep {ep:>4}  AvgR={avg:+6.2f}  WR={wr:5.1f}%  ε={eps_v:.3f}")

    last       = ep_rewards[-100:] if ep_rewards else [0]
    last_eval  = eval_history[-1] if eval_history else None
    train_wr_f = wins / max(1, len(ep_rewards))

    result = {
        "wr":       train_wr_f,
        "avg_r":    float(np.mean(last)),
        "best":     float(best),
        "steps":    float(np.mean(ep_steps[-100:])) if ep_steps else 0.0,
        "conv":     conv,
        "time":     time.time() - start_t,
        "test_wr":  float(last_eval["success_rate"])   if last_eval else 0.0,
        "test_dist":float(last_eval["avg_final_dist"]) if last_eval else float("inf"),
        "gen_gap":  float(train_wr_f - last_eval["success_rate"]) if last_eval else 0.0,
    }

    print(f"  DONE  TrainWR={result['wr']*100:.1f}%  TestWR={result['test_wr']*100:.1f}%  "
          f"Gap={result['gen_gap']*100:+.1f}%  Best={result['best']:+.1f}  "
          f"Time={result['time']:.0f}s")

    return result, skip


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
FAST_MODE = False


def main():
    global FAST_MODE
    pygame.init()

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("RL Local-Obs Benchmark v2")
    clock  = pygame.time.Clock()
    rend   = Renderer(screen)

    print(f"\n{'='*65}")
    print("  RL BENCHMARK v2 — Anti-Overfitting Edition")
    print(f"  STATE_DIM={STATE_DIM}  |  Episodes={EPISODES}")
    print(f"  Cửa sổ {OBS_SIDE}×{OBS_SIDE} | Random Start | Rand Obs Phase | Explore Bonus")
    print(f"{'='*65}")

    results = {}
    for idx, algo in enumerate(ALL_ALGOS):
        res, stop = run_algo(algo, idx, screen, rend, clock)
        if res is None:
            break
        results[algo] = res
        if stop and idx + 1 < len(ALL_ALGOS):
            continue

    if results:
        print_table(results)

    best_algo = max(results, key=lambda k: results[k]["wr"]) if results else "N/A"
    fn = pygame.font.SysFont("Segoe UI", 18, bold=True)
    fs = pygame.font.SysFont("Segoe UI", 13)

    while True:
        screen.fill(BG)
        cx, cy = WIN_W // 2, WIN_H // 2

        title = fn.render("BENCHMARK HOÀN TẤT (v2)", True, C_GREEN)
        screen.blit(title, (cx - title.get_width() // 2, 40))
        sub = fs.render(f"Tốt nhất (Train WR): {best_algo}", True, C_BLUE)
        screen.blit(sub, (cx - sub.get_width() // 2, 76))

        y_ = 110
        for algo, res in sorted(results.items(), key=lambda x: -x[1]["wr"]):
            ac = ALGO_COL.get(algo, C_TEXT)
            bw = int(res["wr"] * 280)
            pygame.draw.rect(screen, (28, 38, 56), (cx - 280, y_, 280, 16), border_radius=3)
            if bw > 0:
                pygame.draw.rect(screen, ac,        (cx - 280, y_, bw,  16), border_radius=3)
            lbl = fs.render(
                f"{algo:<14}  TrainWR={res['wr']*100:5.1f}%  "
                f"TestWR={res['test_wr']*100:5.1f}%  "
                f"Gap={res['gen_gap']*100:+.1f}%", True, ac)
            screen.blit(lbl, (cx + 20, y_))
            y_ += 24

        esc = fs.render("ESC để thoát", True, C_DIM)
        screen.blit(esc, (cx - esc.get_width() // 2, WIN_H - 28))
        pygame.display.flip()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                pygame.quit(); return

        clock.tick(30)


if __name__ == "__main__":
    main()