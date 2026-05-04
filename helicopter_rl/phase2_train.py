"""
Phase 2 Training — Single Static Obstacle Avoidance
====================================================
Ajan, Phase 1'de öğrendiği temel uçuş becerilerini koruyarak
önüne çıkan tek sabit bir silindiri fark edip manevra yaparak
hedefe ulaşmayı öğrenir.

Design Doc references:
  Work Package 5, Stage 2: "Add simple obstacles, light wind"
  Section 2.1.1: 25-metre safety margin
  Section 4.1.4: Safety penalty -1000 on collision

Curriculum:
  Phase 2 ajan, Phase 1 ağırlıklarından başlatılır (transfer learning).
  Bu sayede temel hedefe gidiş politikası korunur, engelden kaçınma eklenir.

Kullanım:
  python phase2_train.py --train --timesteps 800000
  python phase2_train.py --eval  --episodes 5
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helicopter_rl.envs.obstacle_env import ObstacleHelicopterEnv
from helicopter_rl.phase1_train import plot_trajectories

# ── paths ────────────────────────────────────────────────────────────────
MODEL_DIR_P1 = "models/phase1"
MODEL_DIR    = "models/phase2"
LOG_DIR      = "logs/phase2"
FINAL_ZIP    = f"{MODEL_DIR}/ppo_phase2_final"
VEC_NORM     = f"{MODEL_DIR}/vec_normalize.pkl"

PPO_KWARGS = dict(
    learning_rate = 2e-4,      # lower LR for fine-tuning on top of Phase 1
    n_steps       = 2048,
    batch_size    = 256,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.005,     # slightly less exploration needed
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    verbose       = 1,
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
)


def _make_env():
    env = ObstacleHelicopterEnv(
        n_obstacles     = 1,
        obstacle_radius = 15.0,
        obstacle_height = 120.0,
        safety_margin   = 25.0,
        max_steps       = 1200,
        wind_strength_max = 4.0,
    )
    return Monitor(env, LOG_DIR)


# ─────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────

def train(total_timesteps: int = 800_000, n_envs: int = 4):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    train_env = DummyVecEnv([_make_env] * n_envs)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([_make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    # ── Transfer learning: load Phase 1 weights if available ─────────────
    p1_model = f"{MODEL_DIR_P1}/best_model.zip"
    if not os.path.exists(p1_model):
        p1_model = f"{MODEL_DIR_P1}/ppo_phase1_final.zip"

    if os.path.exists(p1_model):
        print(f"  ↳ Loading Phase 1 weights from {p1_model}")
        model = PPO.load(p1_model, env=train_env, **PPO_KWARGS)
    else:
        print("  ↳ No Phase 1 model found – training from scratch")
        model = PPO("MlpPolicy", train_env,
                    tensorboard_log=f"{LOG_DIR}/tensorboard", **PPO_KWARGS)

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
            save_freq=max(80_000 // n_envs, 1),
            save_path=MODEL_DIR,
            name_prefix="ppo_phase2",
        ),
    ])

    print("=" * 60)
    print("PHASE 2: Single Static Obstacle Avoidance")
    print(f"  Total timesteps : {total_timesteps:,}")
    print(f"  Parallel envs   : {n_envs}")
    print(f"  Safety margin   : 25 m")
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
    for path in [f"{MODEL_DIR}/best_model", model_path]:
        if os.path.exists(path + ".zip"):
            model_path = path
            break
    else:
        print(f"[!] No model found – train first.")
        return

    print(f"Loading model: {model_path}.zip")
    model = PPO.load(model_path)

    env = ObstacleHelicopterEnv(
        n_obstacles=1, obstacle_radius=15.0,
        obstacle_height=120.0, safety_margin=25.0,
        max_steps=1200, wind_strength_max=4.0,
    )

    trajectories, targets, starts, all_obstacles = [], [], [], []
    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        starts.append(env.pos.copy())
        targets.append(env.target.copy())
        all_obstacles.append(list(env.get_obstacles()))

        done = False
        total_r = 0.0
        min_dist = float("inf")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            done = terminated or truncated
            md, _ = env._obstacle_distances()
            min_dist = min(min_dist, md)

        traj = env.get_trajectory()
        trajectories.append(traj)
        final_dist = float(np.linalg.norm(env.target - env.pos))
        success   = info.get("success",   False)
        collision = info.get("collision",  False)
        results.append({"ep": ep + 1, "reward": total_r,
                        "final_dist": final_dist, "success": success,
                        "collision": collision, "min_dist": min_dist})

        status = "✓ REACHED" if success else ("✗ COLLISION" if collision else "✗ timeout")
        print(f"  Ep {ep+1:2d}: reward={total_r:8.1f}  "
              f"final_dist={final_dist:6.1f}m  min_obs_dist={min_dist:5.1f}m  {status}")

    success_rate   = sum(r["success"]   for r in results) / n_episodes
    collision_rate = sum(r["collision"] for r in results) / n_episodes
    avg_min_dist   = np.mean([r["min_dist"] for r in results])
    print(f"\n  Success rate   : {success_rate*100:.0f}%")
    print(f"  Collision rate : {collision_rate*100:.0f}%")
    print(f"  Avg min obs dist: {avg_min_dist:.1f} m")

    # Use the last episode's obstacles for the plot
    plot_trajectories(
        trajectories, targets, starts,
        title="Phase 2 – Single Obstacle Avoidance",
        obstacles=all_obstacles[-1],
        save_path=f"{MODEL_DIR}/phase2_trajectories.png",
    )
    return success_rate


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Single Obstacle Avoidance")
    parser.add_argument("--train",     action="store_true")
    parser.add_argument("--eval",      action="store_true")
    parser.add_argument("--timesteps", type=int, default=800_000)
    parser.add_argument("--n_envs",    type=int, default=4)
    parser.add_argument("--episodes",  type=int, default=5)
    args = parser.parse_args()

    if args.train:
        train(total_timesteps=args.timesteps, n_envs=args.n_envs)

    if args.eval or not args.train:
        evaluate(n_episodes=args.episodes)
