"""
Phase 1 Training — Basic 3D Navigation
=======================================
Helikopter ajanı engel olmadan 3B uzayda rastgele bir hedefe uçmayı öğrenir.

Curriculum:
  - Basit ortam, rüzgar minimal
  - PPO (Proximal Policy Optimization)  ←  Design Doc 4.1.3 / Work Package 4

Kullanım:
  python phase1_train.py --train --timesteps 1000000
  python phase1_train.py --eval  --episodes 5
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ── local imports ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helicopter_rl.envs.base_env import FlightControlEnv3D

# ── paths ────────────────────────────────────────────────────────────────
MODEL_DIR = "models/phase1"
LOG_DIR   = "logs/phase1"
FINAL_ZIP = f"{MODEL_DIR}/ppo_phase1_final"
VEC_NORM  = f"{MODEL_DIR}/vec_normalize.pkl"

# ── PPO hyperparameters (Design Doc 3.3.2) ───────────────────────────────
PPO_KWARGS = dict(
    learning_rate  = 3e-4,
    n_steps        = 2048,
    batch_size     = 256,
    n_epochs       = 10,
    gamma          = 0.99,
    gae_lambda     = 0.95,
    clip_range     = 0.2,
    ent_coef       = 0.01,
    vf_coef        = 0.5,
    max_grad_norm  = 0.5,
    verbose        = 1,
    policy_kwargs  = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
)


# ─────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────

def train(total_timesteps: int = 1_000_000, n_envs: int = 4):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    def make_env():
        env = FlightControlEnv3D(max_steps=1000, wind_strength_max=3.0)
        return Monitor(env, LOG_DIR)

    # Vectorised training envs
    train_env = DummyVecEnv([make_env] * n_envs)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Separate eval env (no reward normalisation for fair comparison)
    eval_env = DummyVecEnv([lambda: Monitor(FlightControlEnv3D(max_steps=1000))])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    model = PPO(
        "MlpPolicy",
        train_env,
        tensorboard_log=f"{LOG_DIR}/tensorboard",
        **PPO_KWARGS,
    )

    callbacks = CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=MODEL_DIR,
            log_path=LOG_DIR,
            eval_freq=max(10_000 // n_envs, 1),
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(100_000 // n_envs, 1),
            save_path=MODEL_DIR,
            name_prefix="ppo_phase1",
        ),
    ])

    print("=" * 60)
    print("PHASE 1: Basic 3D Navigation")
    print(f"  Total timesteps : {total_timesteps:,}")
    print(f"  Parallel envs   : {n_envs}")
    print("=" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=True,
    )

    model.save(FINAL_ZIP)
    train_env.save(VEC_NORM)
    print(f"\n✓ Model saved → {FINAL_ZIP}.zip")
    return model, train_env


# ─────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────

def evaluate(model_path: str = FINAL_ZIP, n_episodes: int = 5):
    # Try loading best_model first, fall back to final
    for path in [f"{MODEL_DIR}/best_model", model_path]:
        if os.path.exists(path + ".zip"):
            model_path = path
            break
    else:
        print(f"[!] No model found at {model_path}.zip – train first.")
        return

    print(f"Loading model: {model_path}.zip")
    model = PPO.load(model_path)

    raw_env = FlightControlEnv3D(max_steps=1000)
    eval_env = DummyVecEnv([lambda: raw_env])
    if os.path.exists(VEC_NORM):
        eval_env = VecNormalize.load(VEC_NORM, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
        print(f"Loaded normalization stats: {VEC_NORM}")
    else:
        print(f"[!] vec normalize not found at {VEC_NORM}, evaluating without normalization.")

    trajectories, targets, starts = [], [], []
    results = []

    for ep in range(n_episodes):
        obs = eval_env.reset()
        starts.append(raw_env.pos.copy())
        targets.append(raw_env.target.copy())

        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward_arr, done_arr, info_arr = eval_env.step(action)
            r = float(reward_arr[0])
            info = info_arr[0]
            total_r += r
            done = bool(done_arr[0])

        traj = raw_env.get_trajectory()
        trajectories.append(traj)
        final_dist = float(np.linalg.norm(raw_env.target - raw_env.pos))
        success = info.get("success", False)
        results.append({"ep": ep + 1, "reward": total_r,
                        "final_dist": final_dist, "success": success})

        status = "✓ REACHED" if success else "✗ missed"
        print(f"  Ep {ep+1:2d}: reward={total_r:8.1f}  "
              f"final_dist={final_dist:6.1f}m  {status}")

    success_rate = sum(r["success"] for r in results) / n_episodes
    avg_reward   = np.mean([r["reward"] for r in results])
    print(f"\n  Success rate : {success_rate*100:.0f}%  ({sum(r['success'] for r in results)}/{n_episodes})")
    print(f"  Avg reward   : {avg_reward:.1f}")

    # 3-D plot
    plot_trajectories(
        trajectories, targets, starts,
        title="Phase 1 – Basic 3D Navigation",
        save_path=f"{MODEL_DIR}/phase1_trajectories.png",
    )
    return success_rate


# ─────────────────────────────────────────────────────────────────────────
# Plotting helper (shared across all phases)
# ─────────────────────────────────────────────────────────────────────────

def plot_trajectories(trajectories, targets, starts,
                      title="Helicopter Trajectories",
                      obstacles=None,
                      save_path=None):
    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(trajectories)))

    for i, (traj, tgt, st) in enumerate(zip(trajectories, targets, starts)):
        traj = np.asarray(traj)
        c    = colors[i]

        # Flight path (colour-coded by time)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=c, linewidth=1.5, alpha=0.85, label=f"Ep {i+1}")

        # Start  (green sphere)
        ax.scatter(*st,  color="lime",   s=120, marker="o", zorder=6,
                   edgecolors="black", linewidth=0.5)
        # End    (small cross)
        ax.scatter(*traj[-1], color=c, s=80, marker="x", zorder=6)
        # Target (gold star)
        ax.scatter(*tgt, color="gold",  s=250, marker="*", zorder=7,
                   edgecolors="black", linewidth=0.5)

    # Draw obstacles
    if obstacles:
        for obs in obstacles:
            _draw_cylinder(ax, obs["pos"], obs["radius"], obs["height"])

    ax.set_xlabel("X  [m]", labelpad=8)
    ax.set_ylabel("Y  [m]", labelpad=8)
    ax.set_zlabel("Altitude  [m]", labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)

    # Legend proxies
    from matplotlib.lines import Line2D
    proxies = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lime",
               markersize=10, label="Start"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markersize=14, label="Target"),
    ]
    if obstacles:
        import matplotlib.patches as mpatches
        proxies.append(mpatches.Patch(color="red", alpha=0.35, label="Obstacle"))
    ax.legend(handles=proxies, loc="upper right", fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    plt.close(fig)


def _draw_cylinder(ax, centre, radius, height, color="red", alpha=0.30):
    theta = np.linspace(0.0, 2.0 * np.pi, 32)
    z_vals = np.linspace(centre[2], centre[2] + height, 12)
    theta_g, z_g = np.meshgrid(theta, z_vals)
    x = centre[0] + radius * np.cos(theta_g)
    y = centre[1] + radius * np.sin(theta_g)
    ax.plot_surface(x, y, z_g, color=color, alpha=alpha, linewidth=0)


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Basic Navigation")
    parser.add_argument("--train",      action="store_true", help="Run training")
    parser.add_argument("--eval",       action="store_true", help="Run evaluation")
    parser.add_argument("--timesteps",  type=int, default=1_000_000)
    parser.add_argument("--n_envs",     type=int, default=4)
    parser.add_argument("--episodes",   type=int, default=5)
    args = parser.parse_args()

    if args.train:
        train(total_timesteps=args.timesteps, n_envs=args.n_envs)

    if args.eval or not args.train:
        evaluate(n_episodes=args.episodes)
