"""
MiniWorld World Model Benchmark - FINAL single-file experiment
==============================================================

This script collects a replay buffer from a Gymnasium MiniWorld environment,
trains four simplified world-model architectures, evaluates model quality,
and optionally evaluates control using the SAME CEM planner for all models.

Architectures included:
  1. ha          : Ha-style VAE latent + GRU dynamics
  2. planet      : PlaNet-style RSSM latent dynamics
  3. dreamer     : DreamerV1-style RSSM latent dynamics, larger recurrent state
  4. transformer : CNN latent + causal Transformer dynamics

Important notes:
  - This is a runnable research benchmark, not an exact reproduction of the
    official implementations of Ha et al., PlaNet, DreamerV1, or Transformer WM.
  - For fair world-model comparison, all models use the same replay buffer.
  - For fair control comparison, all models use the same CEM planner.
  - This version adds:
      * replay-buffer positive-label statistics
      * pos_weight for reward/done BCE during training
      * best-threshold F1 for sparse reward/done
      * normalized latent MSE
      * latent collapse diagnostics
      * decoded free-rollout image MSE

Install:
  pip install torch gymnasium miniworld pillow numpy pandas tqdm

Quick smoke test:
  python miniworld_wm_benchmark_final.py --quick

Recommended first run:
  python miniworld_wm_benchmark_final.py --env_id MiniWorld-OneRoom-v0 --rebuild_buffer --collect_steps 30000 --wm_epochs 10 --skip_control

Hallway run:
  python miniworld_wm_benchmark_final.py --env_id MiniWorld-Hallway-v0 --rebuild_buffer --collect_steps 80000 --wm_epochs 20 --forward_prob 0.80 --eval_episodes 10
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import gymnasium as gym
except Exception as e:
    raise ImportError("Please install gymnasium: pip install gymnasium") from e

# Register MiniWorld envs. New Farama package is `miniworld`; older forks may use gym_miniworld.
try:
    import miniworld  # noqa: F401
except Exception:
    try:
        import gym_miniworld  # type: ignore # noqa: F401
    except Exception as e:
        raise ImportError("Please install MiniWorld: pip install miniworld") from e


# -----------------------------
# Reproducibility and utilities
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def get_obs_array(obs):
    """Handle ndarray observations and dict-style wrappers."""
    if isinstance(obs, dict):
        if "image" in obs:
            return obs["image"]
        if "observation" in obs:
            return obs["observation"]
        for v in obs.values():
            if isinstance(v, np.ndarray):
                return v
        raise ValueError(f"Cannot find image observation in dict keys={list(obs.keys())}")
    return obs


def resize_obs(obs, image_size: int) -> np.ndarray:
    obs = get_obs_array(obs)
    if obs.dtype != np.uint8:
        obs = np.clip(obs, 0, 255).astype(np.uint8)
    img = Image.fromarray(obs)
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    img = img.resize((image_size, image_size), resample)
    return np.asarray(img, dtype=np.uint8)


def obs_uint8_to_float_tensor(obs_uint8: np.ndarray) -> torch.Tensor:
    """obs shape (..., H, W, C) uint8 -> (..., C, H, W) float32 [0,1]."""
    arr = torch.from_numpy(obs_uint8).float() / 255.0
    if arr.ndim == 3:
        arr = arr.permute(2, 0, 1)
    elif arr.ndim == 4:
        arr = arr.permute(0, 3, 1, 2)
    elif arr.ndim == 5:
        arr = arr.permute(0, 1, 4, 2, 3)
    else:
        raise ValueError(f"Unexpected obs ndim: {arr.ndim}")
    return arr.contiguous()


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not dicts:
        return {}
    keys = sorted(set().union(*[d.keys() for d in dicts]))
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(d[k]) for d in dicts if k in d and np.isfinite(d[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def normalized_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    var = torch.var(target.detach())
    return mse / (var + eps)


def batch_pos_weight(targets: torch.Tensor, max_weight: float = 50.0) -> torch.Tensor:
    """Return scalar pos_weight for BCEWithLogitsLoss on the same device."""
    targets = (targets >= 0.5).float()
    pos = targets.sum()
    neg = targets.numel() - pos
    weight = neg / (pos + 1e-6)
    weight = torch.clamp(weight, min=1.0, max=max_weight)
    return weight.detach()


def bce_logits_sparse(
    logits: torch.Tensor,
    targets: torch.Tensor,
    use_pos_weight: bool,
    max_pos_weight: float = 50.0,
) -> torch.Tensor:
    targets = (targets >= 0.5).float()
    if use_pos_weight:
        pw = batch_pos_weight(targets, max_weight=max_pos_weight)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
    return F.binary_cross_entropy_with_logits(logits, targets)


@torch.no_grad()
def binary_f1_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits).flatten()
    pred = (probs >= threshold).float()
    tgt = (targets.flatten() >= 0.5).float()
    tp = (pred * tgt).sum()
    fp = (pred * (1.0 - tgt)).sum()
    fn = ((1.0 - pred) * tgt).sum()
    return float((2.0 * tp / (2.0 * tp + fp + fn + 1e-8)).cpu())


@torch.no_grad()
def binary_best_f1_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    probs = torch.sigmoid(logits).flatten()
    tgt = (targets.flatten() >= 0.5).float()
    if float(tgt.sum().cpu()) == 0.0:
        return 0.0, 0.5

    best_f1 = torch.tensor(0.0, device=logits.device)
    best_th = torch.tensor(0.5, device=logits.device)
    for th in torch.linspace(0.01, 0.99, 99, device=logits.device):
        pred = (probs >= th).float()
        tp = (pred * tgt).sum()
        fp = (pred * (1.0 - tgt)).sum()
        fn = ((1.0 - pred) * tgt).sum()
        f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return float(best_f1.cpu()), float(best_th.cpu())


# -----------------------------
# Replay buffer collection
# -----------------------------


def make_env(env_id: str, seed: Optional[int] = None):
    env = gym.make(env_id)
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            try:
                env.seed(seed)
            except Exception:
                pass
    return env


def sample_exploration_action(env, forward_prob: float = 0.60) -> int:
    """Forward-biased random policy for common MiniWorld Discrete(3): 0 left, 1 right, 2 forward."""
    if hasattr(env.action_space, "n") and int(env.action_space.n) == 3:
        left_right = (1.0 - forward_prob) / 2.0
        return int(np.random.choice([0, 1, 2], p=[left_right, left_right, forward_prob]))
    return int(env.action_space.sample())


def collect_replay_buffer(
    env_id: str,
    total_steps: int,
    image_size: int,
    seed: int,
    max_episode_steps: int,
    forward_prob: float,
) -> List[Dict[str, np.ndarray]]:
    env = make_env(env_id, seed=seed)
    episodes: List[Dict[str, np.ndarray]] = []
    steps_collected = 0
    ep_idx = 0

    pbar = tqdm(total=total_steps, desc="Collect replay", ncols=100)
    while steps_collected < total_steps:
        obs, info = env.reset(seed=seed + ep_idx)
        obs = resize_obs(obs, image_size)

        ep_obs = [obs]
        ep_actions: List[int] = []
        ep_rewards: List[float] = []
        ep_dones: List[float] = []
        ep_success = False

        for _ in range(max_episode_steps):
            action = sample_exploration_action(env, forward_prob=forward_prob)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = resize_obs(next_obs, image_size)
            terminal = bool(terminated)  # environment success/failure terminal, not our artificial max-step cutoff
            stop = bool(terminated or truncated)

            ep_obs.append(next_obs)
            ep_actions.append(int(action))
            ep_rewards.append(float(reward))
            ep_dones.append(float(terminal))
            ep_success = ep_success or (float(reward) > 0.0)

            steps_collected += 1
            pbar.update(1)
            obs = next_obs

            if stop or steps_collected >= total_steps:
                break

        if len(ep_actions) > 0:
            episodes.append(
                {
                    "obs": np.stack(ep_obs, axis=0).astype(np.uint8),
                    "actions": np.asarray(ep_actions, dtype=np.int64),
                    "rewards": np.asarray(ep_rewards, dtype=np.float32),
                    "dones": np.asarray(ep_dones, dtype=np.float32),
                    "success": bool(ep_success),
                }
            )
        ep_idx += 1
    pbar.close()
    env.close()

    stats = compute_buffer_stats(episodes)
    print_buffer_stats(stats, title="COLLECTED REPLAY BUFFER")
    return episodes


def compute_buffer_stats(episodes: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
    total_steps = int(sum(len(ep["actions"]) for ep in episodes))
    total_reward_pos = int(sum(int((ep["rewards"] > 0.0).sum()) for ep in episodes))
    total_done_pos = int(sum(int((ep["dones"] > 0.0).sum()) for ep in episodes))
    success_eps = int(sum(bool(ep.get("success", False)) for ep in episodes))
    ep_lengths = [len(ep["actions"]) for ep in episodes]
    return {
        "episodes": float(len(episodes)),
        "transitions": float(total_steps),
        "success_episodes": float(success_eps),
        "success_episode_rate": float(success_eps / max(1, len(episodes))),
        "reward_positive": float(total_reward_pos),
        "reward_pos_rate": float(total_reward_pos / max(1, total_steps)),
        "done_positive": float(total_done_pos),
        "done_pos_rate": float(total_done_pos / max(1, total_steps)),
        "episode_len_mean": float(np.mean(ep_lengths) if ep_lengths else 0.0),
        "episode_len_max": float(np.max(ep_lengths) if ep_lengths else 0.0),
    }


def print_buffer_stats(stats: Dict[str, float], title: str = "REPLAY BUFFER STATS") -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for k, v in stats.items():
        if "rate" in k:
            print(f"{k:24s}: {v:.8f}")
        else:
            print(f"{k:24s}: {v}")


def split_episodes(episodes: List[Dict[str, np.ndarray]], seed: int, train_frac=0.70, val_frac=0.15):
    idx = list(range(len(episodes)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = [episodes[i] for i in idx[:n_train]]
    val = [episodes[i] for i in idx[n_train:n_train + n_val]]
    test = [episodes[i] for i in idx[n_train + n_val:]]
    return train, val, test


class SequenceReplayDataset(Dataset):
    def __init__(self, episodes: List[Dict[str, np.ndarray]], seq_len: int):
        self.episodes = episodes
        self.seq_len = seq_len
        self.index: List[Tuple[int, int]] = []
        for ei, ep in enumerate(episodes):
            T = len(ep["actions"])
            for start in range(0, T - seq_len + 1):
                self.index.append((ei, start))
        if not self.index:
            raise ValueError("No valid sequences. Increase collect_steps/max_episode_steps or decrease seq_len.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ei, start = self.index[idx]
        ep = self.episodes[ei]
        L = self.seq_len
        obs = ep["obs"][start:start + L + 1]
        actions = ep["actions"][start:start + L]
        rewards = ep["rewards"][start:start + L]
        dones = ep["dones"][start:start + L]
        return {
            "obs": obs_uint8_to_float_tensor(obs),
            "actions": torch.from_numpy(actions).long(),
            "rewards": torch.from_numpy(rewards).float(),
            "dones": torch.from_numpy(dones).float(),
        }


# -----------------------------
# Neural building blocks
# -----------------------------


class ConvEncoderVAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x).reshape(x.shape[0], -1)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -8.0, 4.0)
        return mu, logvar


class ConvEncoderDet(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(nn.Linear(256 * 4 * 4, latent_dim), nn.LayerNorm(latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).reshape(x.shape[0], -1)
        return self.fc(h)


class ConvDecoder64(nn.Module):
    """Decoder returns 64x64 image. Keep image_size=64 for clean image-MSE interpretation."""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).reshape(z.shape[0], 256, 4, 4)
        return self.net(h)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, training: bool) -> torch.Tensor:
    if not training:
        return mu
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


# -----------------------------
# World models
# -----------------------------


class HaWorldModel(nn.Module):
    def __init__(self, n_actions: int, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.n_actions = n_actions
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder = ConvEncoderVAE(latent_dim)
        self.decoder = ConvDecoder64(latent_dim)
        self.rnn = nn.GRUCell(latent_dim + n_actions, hidden_dim)
        self.next_z_head = nn.Linear(hidden_dim, latent_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)

    def encode_mu(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(obs)
        return mu

    def compute_loss(self, batch: Dict[str, torch.Tensor], use_pos_weight: bool = False) -> Tuple[torch.Tensor, Dict[str, float]]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = (batch["rewards"] > 0.0).float()
        dones = batch["dones"].float()
        B, Lp1, C, H, W = obs.shape
        L = Lp1 - 1

        flat_obs = obs.reshape(B * Lp1, C, H, W)
        mu, logvar = self.encoder(flat_obs)
        z = reparameterize(mu, logvar, self.training)
        z_seq = z.reshape(B, Lp1, self.latent_dim)
        mu_seq = mu.reshape(B, Lp1, self.latent_dim)

        recon = self.decoder(z).reshape(B, Lp1, C, H, W)
        recon_loss = F.mse_loss(recon, obs)
        kl_loss = kl_standard_normal(mu, logvar)

        h = torch.zeros(B, self.hidden_dim, device=obs.device)
        pred_zs, reward_logits, done_logits = [], [], []
        for t in range(L):
            a = F.one_hot(actions[:, t], self.n_actions).float()
            h = self.rnn(torch.cat([z_seq[:, t], a], dim=-1), h)
            pred_zs.append(self.next_z_head(h))
            reward_logits.append(self.reward_head(h).squeeze(-1))
            done_logits.append(self.done_head(h).squeeze(-1))

        pred_z = torch.stack(pred_zs, dim=1)
        reward_logits = torch.stack(reward_logits, dim=1)
        done_logits = torch.stack(done_logits, dim=1)
        target_z = mu_seq[:, 1:].detach()

        next_z_loss = F.mse_loss(pred_z, target_z)
        next_z_nmse = normalized_mse(pred_z, target_z)
        reward_loss = bce_logits_sparse(reward_logits, rewards, use_pos_weight=use_pos_weight)
        done_loss = bce_logits_sparse(done_logits, dones, use_pos_weight=use_pos_weight)
        loss = recon_loss + 0.01 * kl_loss + next_z_loss + reward_loss + 0.5 * done_loss

        reward_best_f1, reward_best_th = binary_best_f1_from_logits(reward_logits, rewards)
        done_best_f1, done_best_th = binary_best_f1_from_logits(done_logits, dones)
        metrics = {
            "loss": float(loss.detach().cpu()),
            "recon_mse": float(recon_loss.detach().cpu()),
            "kl": float(kl_loss.detach().cpu()),
            "one_step_latent_mse": float(next_z_loss.detach().cpu()),
            "one_step_latent_nmse": float(next_z_nmse.detach().cpu()),
            "reward_bce": float(reward_loss.detach().cpu()),
            "done_bce": float(done_loss.detach().cpu()),
            "reward_f1": binary_f1_from_logits(reward_logits, rewards, threshold=0.5),
            "done_f1": binary_f1_from_logits(done_logits, dones, threshold=0.5),
            "reward_best_f1": reward_best_f1,
            "reward_best_threshold": reward_best_th,
            "done_best_f1": done_best_f1,
            "done_best_threshold": done_best_th,
            "batch_reward_pos_rate": float(rewards.mean().detach().cpu()),
            "batch_done_pos_rate": float(dones.mean().detach().cpu()),
        }
        return loss, metrics

    @torch.no_grad()
    def rollout_metrics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs, actions = batch["obs"], batch["actions"]
        B, Lp1, C, H, W = obs.shape
        L = Lp1 - 1
        true_z = self.encode_mu(obs.reshape(B * Lp1, C, H, W)).reshape(B, Lp1, self.latent_dim)
        z = true_z[:, 0]
        h = torch.zeros(B, self.hidden_dim, device=obs.device)
        pred_zs = []
        for t in range(L):
            a = F.one_hot(actions[:, t], self.n_actions).float()
            h = self.rnn(torch.cat([z, a], dim=-1), h)
            z = self.next_z_head(h)
            pred_zs.append(z)
        pred_z = torch.stack(pred_zs, dim=1)
        target_z = true_z[:, 1:]
        pred_img = self.decoder(pred_z.reshape(B * L, self.latent_dim)).reshape(B, L, C, H, W)
        target_img = obs[:, 1:]
        mse = F.mse_loss(pred_z, target_z)
        nmse = normalized_mse(pred_z, target_z)
        image_mse = F.mse_loss(pred_img, target_img)
        return {
            "free_rollout_latent_mse": float(mse.detach().cpu()),
            "free_rollout_latent_nmse": float(nmse.detach().cpu()),
            "free_rollout_image_mse": float(image_mse.detach().cpu()),
        }

    @torch.no_grad()
    def imagine(self, obs: torch.Tensor, action_seq: torch.Tensor, discount: float = 0.99) -> torch.Tensor:
        B, horizon = action_seq.shape
        z = self.encode_mu(obs)
        h = torch.zeros(B, self.hidden_dim, device=obs.device)
        scores = torch.zeros(B, device=obs.device)
        gamma = 1.0
        for t in range(horizon):
            a = F.one_hot(action_seq[:, t], self.n_actions).float()
            h = self.rnn(torch.cat([z, a], dim=-1), h)
            z = self.next_z_head(h)
            reward_prob = torch.sigmoid(self.reward_head(h).squeeze(-1))
            done_prob = torch.sigmoid(self.done_head(h).squeeze(-1))
            scores = scores + gamma * reward_prob
            gamma = gamma * discount
            scores = scores - 0.01 * done_prob * float(t)
        return scores


class RSSMWorldModel(nn.Module):
    def __init__(
        self,
        n_actions: int,
        latent_dim: int = 32,
        deter_dim: int = 128,
        kl_weight: float = 1.0,
        model_name: str = "rssm",
    ):
        super().__init__()
        self.n_actions = n_actions
        self.latent_dim = latent_dim
        self.deter_dim = deter_dim
        self.kl_weight = kl_weight
        self.model_name = model_name
        self.encoder = ConvEncoderVAE(latent_dim)
        self.decoder = ConvDecoder64(latent_dim)
        self.rnn = nn.GRUCell(latent_dim + n_actions, deter_dim)
        self.prior_mu = nn.Linear(deter_dim, latent_dim)
        self.prior_logvar = nn.Linear(deter_dim, latent_dim)
        self.reward_head = nn.Sequential(nn.Linear(deter_dim + latent_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.done_head = nn.Sequential(nn.Linear(deter_dim + latent_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def encode_mu(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(obs)
        return mu

    def compute_loss(self, batch: Dict[str, torch.Tensor], use_pos_weight: bool = False) -> Tuple[torch.Tensor, Dict[str, float]]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = (batch["rewards"] > 0.0).float()
        dones = batch["dones"].float()
        B, Lp1, C, H, W = obs.shape
        L = Lp1 - 1

        flat_obs = obs.reshape(B * Lp1, C, H, W)
        post_mu, post_logvar = self.encoder(flat_obs)
        z = reparameterize(post_mu, post_logvar, self.training)
        z_seq = z.reshape(B, Lp1, self.latent_dim)
        post_mu_seq = post_mu.reshape(B, Lp1, self.latent_dim)
        post_logvar_seq = post_logvar.reshape(B, Lp1, self.latent_dim)

        recon = self.decoder(z).reshape(B, Lp1, C, H, W)
        recon_loss = F.mse_loss(recon, obs)
        post_kl = kl_standard_normal(post_mu, post_logvar)

        h = torch.zeros(B, self.deter_dim, device=obs.device)
        prior_mus, prior_logvars, reward_logits, done_logits = [], [], [], []
        for t in range(L):
            a = F.one_hot(actions[:, t], self.n_actions).float()
            h = self.rnn(torch.cat([z_seq[:, t], a], dim=-1), h)
            p_mu = self.prior_mu(h)
            p_logvar = torch.clamp(self.prior_logvar(h), -8.0, 4.0)
            prior_mus.append(p_mu)
            prior_logvars.append(p_logvar)
            pred_state = torch.cat([h, p_mu], dim=-1)
            reward_logits.append(self.reward_head(pred_state).squeeze(-1))
            done_logits.append(self.done_head(pred_state).squeeze(-1))

        prior_mus = torch.stack(prior_mus, dim=1)
        prior_logvars = torch.stack(prior_logvars, dim=1)
        reward_logits = torch.stack(reward_logits, dim=1)
        done_logits = torch.stack(done_logits, dim=1)
        target_mu = post_mu_seq[:, 1:].detach()
        target_logvar = post_logvar_seq[:, 1:].detach()

        prior_mse = F.mse_loss(prior_mus, target_mu)
        prior_nmse = normalized_mse(prior_mus, target_mu)
        prior_var_loss = F.mse_loss(prior_logvars, target_logvar)
        reward_loss = bce_logits_sparse(reward_logits, rewards, use_pos_weight=use_pos_weight)
        done_loss = bce_logits_sparse(done_logits, dones, use_pos_weight=use_pos_weight)
        loss = recon_loss + self.kl_weight * 0.01 * post_kl + prior_mse + 0.05 * prior_var_loss + reward_loss + 0.5 * done_loss

        reward_best_f1, reward_best_th = binary_best_f1_from_logits(reward_logits, rewards)
        done_best_f1, done_best_th = binary_best_f1_from_logits(done_logits, dones)
        metrics = {
            "loss": float(loss.detach().cpu()),
            "recon_mse": float(recon_loss.detach().cpu()),
            "kl": float(post_kl.detach().cpu()),
            "one_step_latent_mse": float(prior_mse.detach().cpu()),
            "one_step_latent_nmse": float(prior_nmse.detach().cpu()),
            "reward_bce": float(reward_loss.detach().cpu()),
            "done_bce": float(done_loss.detach().cpu()),
            "reward_f1": binary_f1_from_logits(reward_logits, rewards, threshold=0.5),
            "done_f1": binary_f1_from_logits(done_logits, dones, threshold=0.5),
            "reward_best_f1": reward_best_f1,
            "reward_best_threshold": reward_best_th,
            "done_best_f1": done_best_f1,
            "done_best_threshold": done_best_th,
            "batch_reward_pos_rate": float(rewards.mean().detach().cpu()),
            "batch_done_pos_rate": float(dones.mean().detach().cpu()),
        }
        return loss, metrics

    @torch.no_grad()
    def rollout_metrics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs, actions = batch["obs"], batch["actions"]
        B, Lp1, C, H, W = obs.shape
        L = Lp1 - 1
        true_z = self.encode_mu(obs.reshape(B * Lp1, C, H, W)).reshape(B, Lp1, self.latent_dim)
        z = true_z[:, 0]
        h = torch.zeros(B, self.deter_dim, device=obs.device)
        pred_zs = []
        for t in range(L):
            a = F.one_hot(actions[:, t], self.n_actions).float()
            h = self.rnn(torch.cat([z, a], dim=-1), h)
            z = self.prior_mu(h)
            pred_zs.append(z)
        pred_z = torch.stack(pred_zs, dim=1)
        target_z = true_z[:, 1:]
        pred_img = self.decoder(pred_z.reshape(B * L, self.latent_dim)).reshape(B, L, C, H, W)
        target_img = obs[:, 1:]
        mse = F.mse_loss(pred_z, target_z)
        nmse = normalized_mse(pred_z, target_z)
        image_mse = F.mse_loss(pred_img, target_img)
        return {
            "free_rollout_latent_mse": float(mse.detach().cpu()),
            "free_rollout_latent_nmse": float(nmse.detach().cpu()),
            "free_rollout_image_mse": float(image_mse.detach().cpu()),
        }

    @torch.no_grad()
    def imagine(self, obs: torch.Tensor, action_seq: torch.Tensor, discount: float = 0.99) -> torch.Tensor:
        B, horizon = action_seq.shape
        z = self.encode_mu(obs)
        h = torch.zeros(B, self.deter_dim, device=obs.device)
        scores = torch.zeros(B, device=obs.device)
        gamma = 1.0
        for t in range(horizon):
            a = F.one_hot(action_seq[:, t], self.n_actions).float()
            h = self.rnn(torch.cat([z, a], dim=-1), h)
            z = self.prior_mu(h)
            state = torch.cat([h, z], dim=-1)
            reward_prob = torch.sigmoid(self.reward_head(state).squeeze(-1))
            done_prob = torch.sigmoid(self.done_head(state).squeeze(-1))
            scores = scores + gamma * reward_prob
            gamma = gamma * discount
            scores = scores - 0.01 * done_prob * float(t)
        return scores


class TransformerWorldModel(nn.Module):
    def __init__(
        self,
        n_actions: int,
        latent_dim: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        context_len: int = 16,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.context_len = context_len
        self.encoder = ConvEncoderDet(latent_dim)
        self.decoder = ConvDecoder64(latent_dim)
        self.z_proj = nn.Linear(latent_dim, d_model)
        self.a_embed = nn.Embedding(n_actions, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, context_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.next_z_head = nn.Linear(d_model, latent_dim)
        self.reward_head = nn.Linear(d_model, 1)
        self.done_head = nn.Linear(d_model, 1)
        nn.init.normal_(self.pos_embed, std=0.02)

    def encode_mu(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    @staticmethod
    def causal_mask(T: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((T, T), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def compute_loss(self, batch: Dict[str, torch.Tensor], use_pos_weight: bool = False) -> Tuple[torch.Tensor, Dict[str, float]]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = (batch["rewards"] > 0.0).float()
        dones = batch["dones"].float()
        B, Lp1, C, H, W = obs.shape
        L = Lp1 - 1

        if L > self.context_len:
            obs = obs[:, :self.context_len + 1]
            actions = actions[:, :self.context_len]
            rewards = rewards[:, :self.context_len]
            dones = dones[:, :self.context_len]
            L = self.context_len
            Lp1 = L + 1

        flat_obs = obs.reshape(B * Lp1, C, H, W)
        z_all = self.encoder(flat_obs).reshape(B, Lp1, self.latent_dim)
        recon = self.decoder(z_all.reshape(B * Lp1, self.latent_dim)).reshape(B, Lp1, C, H, W)
        recon_loss = F.mse_loss(recon, obs)

        z_in = z_all[:, :L]
        token = self.z_proj(z_in) + self.a_embed(actions) + self.pos_embed[:, :L]
        h = self.transformer(token, mask=self.causal_mask(L, obs.device))
        pred_z = self.next_z_head(h)
        reward_logits = self.reward_head(h).squeeze(-1)
        done_logits = self.done_head(h).squeeze(-1)
        target_z = z_all[:, 1:].detach()

        next_z_loss = F.mse_loss(pred_z, target_z)
        next_z_nmse = normalized_mse(pred_z, target_z)
        reward_loss = bce_logits_sparse(reward_logits, rewards, use_pos_weight=use_pos_weight)
        done_loss = bce_logits_sparse(done_logits, dones, use_pos_weight=use_pos_weight)
        loss = recon_loss + next_z_loss + reward_loss + 0.5 * done_loss

        reward_best_f1, reward_best_th = binary_best_f1_from_logits(reward_logits, rewards)
        done_best_f1, done_best_th = binary_best_f1_from_logits(done_logits, dones)
        metrics = {
            "loss": float(loss.detach().cpu()),
            "recon_mse": float(recon_loss.detach().cpu()),
            "kl": 0.0,
            "one_step_latent_mse": float(next_z_loss.detach().cpu()),
            "one_step_latent_nmse": float(next_z_nmse.detach().cpu()),
            "reward_bce": float(reward_loss.detach().cpu()),
            "done_bce": float(done_loss.detach().cpu()),
            "reward_f1": binary_f1_from_logits(reward_logits, rewards, threshold=0.5),
            "done_f1": binary_f1_from_logits(done_logits, dones, threshold=0.5),
            "reward_best_f1": reward_best_f1,
            "reward_best_threshold": reward_best_th,
            "done_best_f1": done_best_f1,
            "done_best_threshold": done_best_th,
            "batch_reward_pos_rate": float(rewards.mean().detach().cpu()),
            "batch_done_pos_rate": float(dones.mean().detach().cpu()),
        }
        return loss, metrics

    @torch.no_grad()
    def rollout_metrics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs, actions = batch["obs"], batch["actions"]
        B, Lp1, C, H, W = obs.shape
        L = min(Lp1 - 1, self.context_len)
        obs = obs[:, :L + 1]
        actions = actions[:, :L]
        true_z = self.encoder(obs.reshape(B * (L + 1), C, H, W)).reshape(B, L + 1, self.latent_dim)

        z_context = [true_z[:, 0]]
        pred_zs = []
        for t in range(L):
            ctx_z = torch.stack(z_context, dim=1)
            ctx_a = actions[:, :t + 1]
            token = self.z_proj(ctx_z) + self.a_embed(ctx_a) + self.pos_embed[:, :t + 1]
            h = self.transformer(token, mask=self.causal_mask(t + 1, obs.device))
            z_next = self.next_z_head(h[:, -1])
            pred_zs.append(z_next)
            z_context.append(z_next)

        pred_z = torch.stack(pred_zs, dim=1)
        target_z = true_z[:, 1:]
        pred_img = self.decoder(pred_z.reshape(B * L, self.latent_dim)).reshape(B, L, C, H, W)
        target_img = obs[:, 1:]
        mse = F.mse_loss(pred_z, target_z)
        nmse = normalized_mse(pred_z, target_z)
        image_mse = F.mse_loss(pred_img, target_img)
        return {
            "free_rollout_latent_mse": float(mse.detach().cpu()),
            "free_rollout_latent_nmse": float(nmse.detach().cpu()),
            "free_rollout_image_mse": float(image_mse.detach().cpu()),
        }

    @torch.no_grad()
    def imagine(self, obs: torch.Tensor, action_seq: torch.Tensor, discount: float = 0.99) -> torch.Tensor:
        B, horizon = action_seq.shape
        z0 = self.encoder(obs)
        z_context = [z0]
        scores = torch.zeros(B, device=obs.device)
        gamma = 1.0
        for t in range(horizon):
            start = max(0, len(z_context) - self.context_len)
            ctx_z = torch.stack(z_context[start:], dim=1)
            ctx_actions = action_seq[:, start:t + 1]
            T = ctx_z.shape[1]
            if T > self.context_len:
                ctx_z = ctx_z[:, -self.context_len:]
                ctx_actions = ctx_actions[:, -self.context_len:]
                T = self.context_len
            token = self.z_proj(ctx_z) + self.a_embed(ctx_actions) + self.pos_embed[:, :T]
            h = self.transformer(token, mask=self.causal_mask(T, obs.device))
            last = h[:, -1]
            z_next = self.next_z_head(last)
            reward_prob = torch.sigmoid(self.reward_head(last).squeeze(-1))
            done_prob = torch.sigmoid(self.done_head(last).squeeze(-1))
            scores = scores + gamma * reward_prob
            gamma = gamma * discount
            scores = scores - 0.01 * done_prob * float(t)
            z_context.append(z_next)
        return scores


# -----------------------------
# Training and evaluation
# -----------------------------


def build_model(name: str, n_actions: int, seq_len: int) -> nn.Module:
    name = name.lower()
    if name == "ha":
        return HaWorldModel(n_actions=n_actions, latent_dim=32, hidden_dim=128)
    if name == "planet":
        return RSSMWorldModel(n_actions=n_actions, latent_dim=32, deter_dim=128, kl_weight=0.5, model_name="planet")
    if name == "dreamer":
        return RSSMWorldModel(n_actions=n_actions, latent_dim=48, deter_dim=200, kl_weight=1.0, model_name="dreamer")
    if name == "transformer":
        return TransformerWorldModel(n_actions=n_actions, latent_dim=64, d_model=128, n_heads=4, n_layers=2, context_len=max(seq_len, 8))
    raise ValueError(f"Unknown model name: {name}")


@torch.no_grad()
def compute_latent_stats(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 10) -> Dict[str, float]:
    model.eval()
    zs = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        obs = batch["obs"]
        B, Lp1, C, H, W = obs.shape
        z = model.encode_mu(obs.reshape(B * Lp1, C, H, W))
        zs.append(z.detach().cpu())
    if not zs:
        return {
            "latent_mean_abs": float("nan"),
            "latent_std_mean": float("nan"),
            "latent_std_min": float("nan"),
            "latent_std_max": float("nan"),
            "latent_var_mean": float("nan"),
        }
    z = torch.cat(zs, dim=0)
    std = z.std(dim=0)
    return {
        "latent_mean_abs": float(z.mean(dim=0).abs().mean()),
        "latent_std_mean": float(std.mean()),
        "latent_std_min": float(std.min()),
        "latent_std_max": float(std.max()),
        "latent_var_mean": float(z.var(dim=0).mean()),
    }


@torch.no_grad()
def evaluate_world_model(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 20) -> Dict[str, float]:
    model.eval()
    metrics_list = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        _, metrics = model.compute_loss(batch, use_pos_weight=False)
        rollout = model.rollout_metrics(batch)
        metrics.update(rollout)
        metrics_list.append(metrics)
    out = mean_dict(metrics_list)
    out.update(compute_latent_stats(model, loader, device, max_batches=min(10, max_batches)))
    return out


def train_world_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    grad_clip: float,
    out_dir: Path,
    use_pos_weight: bool,
) -> Dict[str, float]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_metrics: Dict[str, float] = {}
    ckpt_path = out_dir / f"{name}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_metrics = []
        pbar = tqdm(train_loader, desc=f"Train {name} epoch {epoch}/{epochs}", ncols=120)
        for batch in pbar:
            batch = move_batch_to_device(batch, device)
            loss, metrics = model.compute_loss(batch, use_pos_weight=use_pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            train_metrics.append(metrics)
            pbar.set_postfix(loss=f"{metrics['loss']:.3f}", recon=f"{metrics['recon_mse']:.4f}")

        train_mean = mean_dict(train_metrics)
        val_metrics = evaluate_world_model(model, val_loader, device, max_batches=20)
        print(f"\n[{name}] epoch={epoch}")
        print("train:", train_mean)
        print("val  :", val_metrics)

        # Select by validation loss. You may change this to recon + rollout if preferred.
        if val_metrics.get("loss", float("inf")) < best_val:
            best_val = val_metrics["loss"]
            best_metrics = val_metrics
            torch.save(model.state_dict(), ckpt_path)

    if ckpt_path.exists():
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
    return best_metrics


# -----------------------------
# CEM planner and control eval
# -----------------------------


@torch.no_grad()
def cem_plan_action(
    model: nn.Module,
    obs_np: np.ndarray,
    n_actions: int,
    device: torch.device,
    image_size: int,
    horizon: int,
    candidates: int,
    elites: int,
    iterations: int,
    discount: float,
) -> int:
    model.eval()
    obs_uint8 = resize_obs(obs_np, image_size)
    obs_t = obs_uint8_to_float_tensor(obs_uint8).unsqueeze(0).to(device)

    logits = torch.zeros(horizon, n_actions, device=device)
    best_action = 0
    best_score = -1e9

    for _ in range(iterations):
        probs = torch.softmax(logits, dim=-1)
        seq_parts = [torch.multinomial(probs[t], num_samples=candidates, replacement=True) for t in range(horizon)]
        action_seq = torch.stack(seq_parts, dim=1).long()
        obs_batch = obs_t.repeat(candidates, 1, 1, 1)
        scores = model.imagine(obs_batch, action_seq, discount=discount)

        top = torch.topk(scores, k=min(elites, candidates)).indices
        elite_seq = action_seq[top]
        if float(scores[top[0]].detach().cpu()) > best_score:
            best_score = float(scores[top[0]].detach().cpu())
            best_action = int(elite_seq[0, 0].detach().cpu())

        new_probs = torch.zeros_like(probs) + 1e-3
        for t in range(horizon):
            counts = torch.bincount(elite_seq[:, t], minlength=n_actions).float().to(device)
            new_probs[t] += counts
        new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)
        logits = torch.log(new_probs + 1e-8)

    return int(best_action)


def evaluate_control_with_planner(
    name: str,
    model: nn.Module,
    env_id: str,
    n_actions: int,
    device: torch.device,
    image_size: int,
    eval_episodes: int,
    max_episode_steps: int,
    seed: int,
    planner_horizon: int,
    planner_candidates: int,
    planner_elites: int,
    planner_iterations: int,
    discount: float,
) -> Dict[str, float]:
    returns, successes, steps_list, efficiency = [], [], [], []
    env = make_env(env_id)
    for ep in tqdm(range(eval_episodes), desc=f"Control eval {name}", ncols=100):
        obs, info = env.reset(seed=seed + 10000 + ep)
        total_reward = 0.0
        success = False
        steps = 0
        for _ in range(max_episode_steps):
            action = cem_plan_action(
                model=model,
                obs_np=obs,
                n_actions=n_actions,
                device=device,
                image_size=image_size,
                horizon=planner_horizon,
                candidates=planner_candidates,
                elites=planner_elites,
                iterations=planner_iterations,
                discount=discount,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if float(reward) > 0.0:
                success = True
            if terminated or truncated:
                break
        returns.append(total_reward)
        successes.append(1.0 if success else 0.0)
        steps_list.append(steps)
        # Proxy for SPL when shortest path is unavailable from MiniWorld API.
        efficiency.append((1.0 if success else 0.0) * max(0.0, 1.0 - steps / max_episode_steps))
    env.close()
    return {
        "model": name,
        "success_rate": float(np.mean(successes)),
        "avg_return": float(np.mean(returns)),
        "avg_steps": float(np.mean(steps_list)),
        "success_efficiency_proxy": float(np.mean(efficiency)),
    }


def evaluate_random_policy(
    env_id: str,
    eval_episodes: int,
    max_episode_steps: int,
    seed: int,
    forward_prob: float,
) -> Dict[str, float]:
    env = make_env(env_id)
    returns, successes, steps_list, efficiency = [], [], [], []
    for ep in tqdm(range(eval_episodes), desc="Control eval random", ncols=100):
        obs, info = env.reset(seed=seed + 20000 + ep)
        total_reward, success, steps = 0.0, False, 0
        for _ in range(max_episode_steps):
            action = sample_exploration_action(env, forward_prob=forward_prob)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            success = success or (float(reward) > 0.0)
            steps += 1
            if terminated or truncated:
                break
        returns.append(total_reward)
        successes.append(1.0 if success else 0.0)
        steps_list.append(steps)
        efficiency.append((1.0 if success else 0.0) * max(0.0, 1.0 - steps / max_episode_steps))
    env.close()
    return {
        "model": "random_forward_policy",
        "success_rate": float(np.mean(successes)),
        "avg_return": float(np.mean(returns)),
        "avg_steps": float(np.mean(steps_list)),
        "success_efficiency_proxy": float(np.mean(efficiency)),
    }


# -----------------------------
# Main
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="MiniWorld-Hallway-v0")
    parser.add_argument("--models", type=str, default="ha,planet,dreamer,transformer")
    parser.add_argument("--output_dir", type=str, default="runs_miniworld_wm_final")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=64, help="Keep 64 for decoder image-MSE compatibility.")
    parser.add_argument("--collect_steps", type=int, default=5000)
    parser.add_argument("--rebuild_buffer", action="store_true")
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--wm_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_clip", type=float, default=100.0)
    parser.add_argument("--eval_episodes", type=int, default=3)
    parser.add_argument("--max_episode_steps", type=int, default=120)
    parser.add_argument("--forward_prob", type=float, default=0.60)
    parser.add_argument("--planner_horizon", type=int, default=8)
    parser.add_argument("--planner_candidates", type=int, default=64)
    parser.add_argument("--planner_elites", type=int, default=8)
    parser.add_argument("--planner_iterations", type=int, default=2)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--skip_control", action="store_true")
    parser.add_argument("--no_pos_weight", action="store_true", help="Disable pos_weight for reward/done BCE during training.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true", help="Tiny smoke-test settings.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.image_size != 64:
        raise ValueError("This final benchmark keeps decoder output at 64x64. Please use --image_size 64.")

    if args.quick:
        args.collect_steps = 1000
        args.wm_epochs = 1
        args.eval_episodes = 1
        args.batch_size = 16
        args.seq_len = 6
        args.planner_candidates = 24
        args.planner_elites = 6
        args.planner_iterations = 1
        args.max_episode_steps = 80

    set_seed(args.seed)
    out_dir = ensure_dir(args.output_dir)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    env = make_env(args.env_id, seed=args.seed)
    if not hasattr(env.action_space, "n"):
        raise ValueError("This benchmark expects a Discrete action space.")
    n_actions = int(env.action_space.n)
    env.close()
    print(f"Env: {args.env_id}, n_actions={n_actions}")

    buffer_path = out_dir / f"buffer_{args.env_id.replace('/', '_')}_{args.collect_steps}_{args.image_size}_fwd{args.forward_prob:.2f}.pkl"
    if buffer_path.exists() and not args.rebuild_buffer:
        print(f"Loading replay buffer: {buffer_path}")
        with open(buffer_path, "rb") as f:
            episodes = pickle.load(f)
        print_buffer_stats(compute_buffer_stats(episodes), title="LOADED REPLAY BUFFER")
    else:
        episodes = collect_replay_buffer(
            env_id=args.env_id,
            total_steps=args.collect_steps,
            image_size=args.image_size,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            forward_prob=args.forward_prob,
        )
        with open(buffer_path, "wb") as f:
            pickle.dump(episodes, f)
        print(f"Saved replay buffer: {buffer_path}")

    all_buffer_stats = compute_buffer_stats(episodes)
    with open(out_dir / "buffer_stats.json", "w", encoding="utf-8") as f:
        json.dump(all_buffer_stats, f, indent=2, ensure_ascii=False)

    train_eps, val_eps, test_eps = split_episodes(episodes, seed=args.seed)
    if len(val_eps) == 0:
        val_eps = train_eps
    if len(test_eps) == 0:
        test_eps = val_eps

    split_stats = {
        "train": compute_buffer_stats(train_eps),
        "val": compute_buffer_stats(val_eps),
        "test": compute_buffer_stats(test_eps),
    }
    with open(out_dir / "split_buffer_stats.json", "w", encoding="utf-8") as f:
        json.dump(split_stats, f, indent=2, ensure_ascii=False)

    print(f"Split episodes: train={len(train_eps)}, val={len(val_eps)}, test={len(test_eps)}")
    print_buffer_stats(split_stats["train"], title="TRAIN BUFFER STATS")
    print_buffer_stats(split_stats["val"], title="VAL BUFFER STATS")
    print_buffer_stats(split_stats["test"], title="TEST BUFFER STATS")

    train_ds = SequenceReplayDataset(train_eps, seq_len=args.seq_len)
    val_ds = SequenceReplayDataset(val_eps, seq_len=args.seq_len)
    test_ds = SequenceReplayDataset(test_eps, seq_len=args.seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model_names = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    wm_rows = []
    control_rows = []

    if not args.skip_control:
        rand_row = evaluate_random_policy(
            env_id=args.env_id,
            eval_episodes=args.eval_episodes,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            forward_prob=args.forward_prob,
        )
        control_rows.append(rand_row)
        pd.DataFrame(control_rows).to_csv(out_dir / "control_results.csv", index=False)

    for name in model_names:
        print("\n" + "=" * 80)
        print(f"MODEL: {name}")
        print("=" * 80)
        model = build_model(name, n_actions=n_actions, seq_len=args.seq_len)
        n_params = count_parameters(model)
        print(f"Parameters: {n_params:,}")

        start = time.time()
        best_val = train_world_model(
            name=name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.wm_epochs,
            lr=args.lr,
            grad_clip=args.grad_clip,
            out_dir=out_dir,
            use_pos_weight=(not args.no_pos_weight),
        )
        train_seconds = time.time() - start
        test_metrics = evaluate_world_model(model, test_loader, device, max_batches=50)

        wm_row = {
            "model": name,
            "env_id": args.env_id,
            "params": n_params,
            "train_seconds": train_seconds,
            "buffer_reward_pos_rate": all_buffer_stats["reward_pos_rate"],
            "buffer_done_pos_rate": all_buffer_stats["done_pos_rate"],
            "buffer_success_episode_rate": all_buffer_stats["success_episode_rate"],
            **{f"val_{k}": v for k, v in best_val.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        wm_rows.append(wm_row)
        pd.DataFrame(wm_rows).to_csv(out_dir / "world_model_results.csv", index=False)
        print(f"World-model test metrics for {name}: {test_metrics}")

        if not args.skip_control:
            ctrl = evaluate_control_with_planner(
                name=name + "+CEM",
                model=model,
                env_id=args.env_id,
                n_actions=n_actions,
                device=device,
                image_size=args.image_size,
                eval_episodes=args.eval_episodes,
                max_episode_steps=args.max_episode_steps,
                seed=args.seed,
                planner_horizon=args.planner_horizon,
                planner_candidates=args.planner_candidates,
                planner_elites=args.planner_elites,
                planner_iterations=args.planner_iterations,
                discount=args.discount,
            )
            control_rows.append(ctrl)
            pd.DataFrame(control_rows).to_csv(out_dir / "control_results.csv", index=False)
            print(f"Control metrics for {name}: {ctrl}")

    wm_df = pd.DataFrame(wm_rows)
    ctrl_df = pd.DataFrame(control_rows)

    print("\n" + "=" * 80)
    print("WORLD MODEL RESULTS")
    print("=" * 80)
    print(wm_df.to_string(index=False))
    print(f"Saved: {out_dir / 'world_model_results.csv'}")
    print(f"Saved: {out_dir / 'buffer_stats.json'}")
    print(f"Saved: {out_dir / 'split_buffer_stats.json'}")

    if not args.skip_control:
        print("\n" + "=" * 80)
        print("CONTROL RESULTS")
        print("=" * 80)
        print(ctrl_df.to_string(index=False))
        print(f"Saved: {out_dir / 'control_results.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
