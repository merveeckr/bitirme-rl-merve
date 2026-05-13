"""
Phase 1 Eğitim — JSBSim Helikopter (Temel Navigasyon)
=======================================================
Ajan engel olmadan hedefe ulaşmayı öğrenir.
JSBSim gerçek 6-DOF katı cisim dinamiğini sağlar.

Kullanım:
  python helicopter_rl_jsbsim/phase1_train.py --train
  python helicopter_rl_jsbsim/phase1_train.py --eval --episodes 10
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback, EvalCallback)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helicopter_rl_jsbsim.envs.jsbsim_base_env import JSBSimHelicopterEnv

MODEL_DIR = "models/jsbsim_phase1"
LOG_DIR   = "logs/jsbsim_phase1"
FINAL_ZIP = f"{MODEL_DIR}/ppo_jsbsim_phase1_final"
VEC_NORM  = f"{MODEL_DIR}/vec_normalize.pkl"

PPO_KWARGS = dict(
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 256,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    verbose       = 1,
    policy_kwargs = dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
)


def _make_env(log_dir: str, wind_max: float = 0.0):
    def _inner():
        env = JSBSimHelicopterEnv(
            max_steps=1000,
            wind_strength_max=wind_max,
        )
        return Monitor(env, log_dir)
    return _inner


def plot_trajectories(trajectories, targets, starts,
                      title="Trajectories", obstacles=None,
                      save_path=None):
    fig = plt.figure(figsize=(12, 5))

    # XY görünümü
    ax1 = fig.add_subplot(121)
    for i, (traj, tgt, st) in enumerate(zip(trajectories, targets, starts)):
        traj = np.array(traj)
        ax1.plot(traj[:, 0], traj[:, 1], alpha=0.7)
        ax1.plot(st[0],  st[1],  'go', ms=6)
        ax1.plot(tgt[0], tgt[1], 'r*', ms=10)

    if obstacles:
        for obs in obstacles:
            c = plt.Circle((obs["pos"][0], obs["pos"][1]),
                            obs["radius"], color='red', alpha=0.3)
            ax1.add_patch(c)

    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
    ax1.set_title(f"{title} — XY")
    ax1.set_aspect('equal'); ax1.grid(True)

    # İrtifa profili
    ax2 = fig.add_subplot(122)
    for traj in trajectories:
        traj = np.array(traj)
        dist = np.cumsum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
        dist = np.insert(dist, 0, 0)
        ax2.plot(dist, traj[:, 2], alpha=0.7)
    ax2.set_xlabel("Mesafe (m)"); ax2.set_ylabel("İrtifa (m)")
    ax2.set_title("İrtifa Profili"); ax2.grid(True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
        print(f"  → Grafik kaydedildi: {save_path}")
    plt.close()


def train(total_timesteps: int = 800_000, n_envs: int = 4):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    factory   = _make_env(LOG_DIR, wind_max=3.0)
    train_env = DummyVecEnv([factory] * n_envs)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env  = DummyVecEnv([factory])
    eval_env  = VecNormalize(eval_env,  norm_obs=True, norm_reward=False,
                              clip_obs=10.0, training=False)

    model = PPO("MlpPolicy", train_env,
                tensorboard_log=f"{LOG_DIR}/tensorboard", **PPO_KWARGS)

    callbacks = CallbackList([
        EvalCallback(eval_env,
                     best_model_save_path=MODEL_DIR,
                     log_path=LOG_DIR,
                     eval_freq=max(10_000 // n_envs, 1),
                     n_eval_episodes=10,
                     deterministic=True, verbose=1),
        CheckpointCallback(save_freq=max(100_000 // n_envs, 1),
                           save_path=MODEL_DIR,
                           name_prefix="ppo_jsbsim_p1"),
    ])

    print("=" * 60)
    print(f"JSBSim PHASE 1 — Temel Navigasyon")
    print(f"  Timesteps : {total_timesteps:,}")
    print(f"  Envs      : {n_envs}")
    print("=" * 60)

    model.learn(total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=True)

    model.save(FINAL_ZIP)
    train_env.save(VEC_NORM)
    print(f"\n✓ Model kaydedildi → {FINAL_ZIP}.zip")
    return model, train_env


def evaluate(model_path: str = FINAL_ZIP, n_episodes: int = 10):
    for path in [f"{MODEL_DIR}/best_model", model_path]:
        if os.path.exists(path + ".zip"):
            model_path = path
            break
    else:
        print("[!] Model bulunamadı — önce eğit.")
        return

    model = PPO.load(model_path)
    env   = JSBSimHelicopterEnv(max_steps=1000, wind_strength_max=3.0)

    trajectories, targets, starts = [], [], []
    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        starts.append(env.pos.copy())
        targets.append(env.target.copy())
        done = False; total_r = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            done = term or trunc

        trajectories.append(env.get_trajectory())
        final_dist = float(np.linalg.norm(env.target - env.pos))
        success    = info.get("success", False)
        results.append({"ep": ep+1, "reward": total_r,
                        "final_dist": final_dist, "success": success})
        print(f"  Ep {ep+1:2d}: ödül={total_r:8.1f}  "
              f"son_mesafe={final_dist:6.1f}m  "
              f"{'✓ BAŞARILI' if success else '✗ BAŞARISIZ'}")

    sr = sum(r["success"] for r in results) / n_episodes
    print(f"\n  Başarı oranı: {sr*100:.0f}%")

    plot_trajectories(trajectories, targets, starts,
                      title="JSBSim Phase 1 — Navigasyon",
                      save_path=f"{MODEL_DIR}/trajectories.png")
    env.close()
    return sr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",     action="store_true")
    parser.add_argument("--eval",      action="store_true")
    parser.add_argument("--timesteps", type=int, default=800_000)
    parser.add_argument("--n_envs",    type=int, default=4)
    parser.add_argument("--episodes",  type=int, default=10)
    args = parser.parse_args()

    if args.train:
        train(total_timesteps=args.timesteps, n_envs=args.n_envs)

    if args.eval or not args.train:
        evaluate(n_episodes=args.episodes)
