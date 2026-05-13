"""
Phase 3 Eğitim — JSBSim + Çoklu Engel Kaçınma
===============================================
Curriculum:
  3A → 2 engel   (Phase 1 ağırlıklarından başla)
  3B → 3 engel
  3C → 5 engel + domain randomisation

Kullanım:
  python helicopter_rl_jsbsim/phase3_train.py --train --curriculum
  python helicopter_rl_jsbsim/phase3_train.py --eval --episodes 8
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback, EvalCallback)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helicopter_rl_jsbsim.envs.obstacle_env import ObstacleHelicopterEnv
from helicopter_rl_jsbsim.phase1_train import plot_trajectories

MODEL_DIR_P1 = "models/jsbsim_phase1"
MODEL_DIR    = "models/jsbsim_phase3"
LOG_DIR      = "logs/jsbsim_phase3"
FINAL_ZIP    = f"{MODEL_DIR}/ppo_jsbsim_phase3_final"
VEC_NORM     = f"{MODEL_DIR}/vec_normalize.pkl"

PPO_KWARGS = dict(
    learning_rate = 1e-4,
    n_steps       = 2048,
    batch_size    = 256,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.02,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    verbose       = 1,
    policy_kwargs = dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
)

CURRICULUM_STAGES = [
    {"n_obstacles": 2, "wind_strength_max": 0.0, "randomize_radius": False, "timesteps": 600_000},
    {"n_obstacles": 3, "wind_strength_max": 0.0, "randomize_radius": True,  "timesteps": 600_000},
    {"n_obstacles": 5, "wind_strength_max": 0.0, "randomize_radius": True,  "timesteps": 600_000},
]


class ObstacleStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._ep_collisions = 0
        self._ep_count      = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("collision"):
                self._ep_collisions += 1
            if info.get("episode"):
                self._ep_count += 1
        return True

    def _on_rollout_end(self):
        if self._ep_count > 0:
            self.logger.record("obstacle/collision_rate",
                               self._ep_collisions / self._ep_count)
        self._ep_collisions = 0
        self._ep_count      = 0


def _make_env(n_obstacles, wind_max, randomize_radius, log_dir):
    def _inner():
        env = ObstacleHelicopterEnv(
            n_obstacles       = n_obstacles,
            obstacle_radius   = 15.0,
            obstacle_height   = 120.0,
            safety_margin     = 25.0,
            randomize_radius  = randomize_radius,
            max_steps         = 1500,
            wind_strength_max = wind_max,
        )
        return Monitor(env, log_dir)
    return _inner


def train(total_timesteps: int = 1_200_000, n_envs: int = 4,
          n_obstacles: int = 4, wind_strength_max: float = 0.0,
          randomize_radius: bool = True):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    factory   = _make_env(n_obstacles, wind_strength_max, randomize_radius, LOG_DIR)
    train_env = DummyVecEnv([factory] * n_envs)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env  = DummyVecEnv([factory])
    eval_env  = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                              clip_obs=10.0, training=False)

    # Phase 1 ağırlıklarından başla
    for src in [f"{MODEL_DIR_P1}/best_model", f"{MODEL_DIR_P1}/ppo_jsbsim_phase1_final"]:
        if os.path.exists(src + ".zip"):
            print(f"  ↳ Phase 1 ağırlıkları yüklendi: {src}.zip")
            kw = {k: v for k, v in PPO_KWARGS.items() if k != 'policy_kwargs'}
            model = PPO.load(src, env=train_env, **kw)
            break
    else:
        print("  ↳ Phase 1 modeli bulunamadı — sıfırdan başlıyor")
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
                           name_prefix="ppo_jsbsim_p3"),
        ObstacleStatsCallback(),
    ])

    print("=" * 60)
    print(f"JSBSim PHASE 3 — Çoklu Engel ({n_obstacles} engel)")
    print(f"  Timesteps : {total_timesteps:,}")
    print(f"  Envs      : {n_envs}")
    print("=" * 60)

    model.learn(total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=True)

    model.save(FINAL_ZIP)
    train_env.save(VEC_NORM)
    print(f"\n✓ Phase 3 modeli kaydedildi → {FINAL_ZIP}.zip")
    return model, train_env


def train_curriculum(n_envs: int = 4):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    prev_save = None
    for src in [f"{MODEL_DIR_P1}/best_model", f"{MODEL_DIR_P1}/ppo_jsbsim_phase1_final"]:
        if os.path.exists(src + ".zip"):
            prev_save = src
            break

    model = None
    for stage_idx, stage in enumerate(CURRICULUM_STAGES):
        stage_name = f"3{'ABC'[stage_idx]}"
        stage_dir  = f"{MODEL_DIR}/stage_{stage_name}"
        stage_log  = f"{LOG_DIR}/stage_{stage_name}"
        os.makedirs(stage_dir, exist_ok=True)
        os.makedirs(stage_log, exist_ok=True)

        factory   = _make_env(stage["n_obstacles"], stage["wind_strength_max"],
                               stage["randomize_radius"], stage_log)
        train_env = DummyVecEnv([factory] * n_envs)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env  = DummyVecEnv([factory])
        eval_env  = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                                  clip_obs=10.0, training=False)

        if prev_save and os.path.exists(prev_save + ".zip"):
            print(f"\n  Aşama {stage_name}: {prev_save}.zip yükleniyor")
            kw = {k: v for k, v in PPO_KWARGS.items() if k != 'policy_kwargs'}
            model = PPO.load(prev_save, env=train_env, **kw)
        else:
            print(f"\n  Aşama {stage_name}: sıfırdan")
            model = PPO("MlpPolicy", train_env,
                        tensorboard_log=f"{stage_log}/tensorboard", **PPO_KWARGS)

        print(f"  Aşama {stage_name}: {stage['n_obstacles']} engel, "
              f"rüzgar≤{stage['wind_strength_max']}m/s, "
              f"adım={stage['timesteps']:,}")

        callbacks = CallbackList([
            EvalCallback(eval_env,
                         best_model_save_path=stage_dir,
                         log_path=stage_log,
                         eval_freq=max(8_000 // n_envs, 1),
                         n_eval_episodes=8,
                         deterministic=True, verbose=0),
            ObstacleStatsCallback(),
        ])

        model.learn(total_timesteps=stage["timesteps"],
                    callback=callbacks,
                    progress_bar=True,
                    reset_num_timesteps=True)

        stage_path = f"{stage_dir}/ppo_jsbsim_phase{stage_name}_final"
        model.save(stage_path)
        train_env.save(f"{stage_dir}/vec_normalize.pkl")
        prev_save = stage_path
        print(f"  ✓ Aşama {stage_name} kaydedildi → {stage_path}.zip")

    model.save(FINAL_ZIP)
    print(f"\n✓ Curriculum tamamlandı → {FINAL_ZIP}.zip")
    return model


def evaluate(model_path: str = FINAL_ZIP, n_episodes: int = 8, n_obstacles: int = 4):
    for path in [f"{MODEL_DIR}/best_model", model_path]:
        if os.path.exists(path + ".zip"):
            model_path = path
            break
    else:
        print("[!] Model bulunamadı — önce eğit.")
        return

    model = PPO.load(model_path)
    env   = ObstacleHelicopterEnv(
        n_obstacles=n_obstacles, obstacle_radius=15.0,
        obstacle_height=120.0, safety_margin=25.0,
        randomize_radius=True, max_steps=1500, wind_strength_max=7.0)

    trajectories, targets, starts = [], [], []
    all_obstacles_last = []
    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        starts.append(env.pos.copy())
        targets.append(env.target.copy())
        ep_obstacles = list(env.get_obstacles())
        done = False; total_r = 0.0; min_dist = float("inf")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            done = term or trunc
            md, _ = env._obstacle_distances()
            min_dist = min(min_dist, md)

        trajectories.append(env.get_trajectory())
        final_dist = float(np.linalg.norm(env.target - env.pos))
        success    = info.get("success",   False)
        collision  = info.get("collision",  False)
        results.append({"ep": ep+1, "reward": total_r, "final_dist": final_dist,
                        "success": success, "collision": collision, "min_dist": min_dist})
        all_obstacles_last = ep_obstacles

        status = "✓ HEDEFE ULAŞTI" if success else ("✗ ÇARPIŞMA" if collision else "✗ timeout")
        print(f"  Ep {ep+1:2d}: ödül={total_r:8.1f}  "
              f"son_mesafe={final_dist:6.1f}m  min_engel={min_dist:5.1f}m  {status}")

    sr = sum(r["success"]   for r in results) / n_episodes
    cr = sum(r["collision"] for r in results) / n_episodes
    ad = np.mean([r["min_dist"] for r in results])
    print(f"\n  Başarı oranı    : {sr*100:.0f}%")
    print(f"  Çarpışma oranı  : {cr*100:.0f}%")
    print(f"  Ort. min engel  : {ad:.1f} m")

    plot_trajectories(trajectories, targets, starts,
                      title=f"JSBSim Phase 3 — {n_obstacles} Engel",
                      obstacles=all_obstacles_last,
                      save_path=f"{MODEL_DIR}/trajectories.png")
    env.close()
    return sr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",       action="store_true")
    parser.add_argument("--eval",        action="store_true")
    parser.add_argument("--curriculum",  action="store_true")
    parser.add_argument("--timesteps",   type=int, default=1_200_000)
    parser.add_argument("--n_envs",      type=int, default=4)
    parser.add_argument("--n_obstacles", type=int, default=4)
    parser.add_argument("--episodes",    type=int, default=8)
    args = parser.parse_args()

    if args.train:
        if args.curriculum:
            train_curriculum(n_envs=args.n_envs)
        else:
            train(total_timesteps=args.timesteps,
                  n_envs=args.n_envs,
                  n_obstacles=args.n_obstacles)

    if args.eval or not args.train:
        evaluate(n_episodes=args.episodes, n_obstacles=args.n_obstacles)
