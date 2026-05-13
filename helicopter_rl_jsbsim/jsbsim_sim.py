"""
JSBSim Helikopter Simülasyonu — Demo Görselleştirici
=====================================================
JSBSim fiziğiyle üretilen trajektoryayı PyBullet'te görselleştirir.

Kullanim:
  python helicopter_rl_jsbsim/jsbsim_sim.py --demo --seed 5
  python helicopter_rl_jsbsim/jsbsim_sim.py --live --seed 5   (model gerektirir)
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helicopter_rl_jsbsim.envs.obstacle_env import ObstacleHelicopterEnv
from helicopter_rl.pybullet_sim import (
    build_scene, _update_heli, simulate_live as _base_simulate_live,
)
import pybullet as p
import pybullet_data

MODEL_DIR = "models/jsbsim_phase3"
MODEL_DIR_P1 = "models/jsbsim_phase1"


def _load_model():
    from stable_baselines3 import PPO
    candidates = [
        f"{MODEL_DIR}/best_model",
        f"{MODEL_DIR}/stage_3C/best_model",
        f"{MODEL_DIR}/stage_3B/best_model",
        f"{MODEL_DIR}/ppo_jsbsim_phase3_final",
        f"{MODEL_DIR_P1}/best_model",
        f"{MODEL_DIR_P1}/ppo_jsbsim_phase1_final",
    ]
    for path in candidates:
        if os.path.exists(path + ".zip"):
            model = PPO.load(path)
            print(f"Model yuklendi: {path}.zip")
            return model
    raise FileNotFoundError(
        "JSBSim modeli bulunamadi. Once egitimi calistir:\n"
        "  python helicopter_rl_jsbsim/phase1_train.py --train")


def _heuristic_action(env) -> np.ndarray:
    delta = env.target - env.pos
    dist  = float(np.linalg.norm(delta)) + 1e-6
    dx, dy, dz = delta
    yr    = np.deg2rad(env.yaw)
    fwd   =  dx * np.cos(yr) + dy * np.sin(yr)
    right = -dx * np.sin(yr) + dy * np.cos(yr)
    K = 0.06
    pitch_cmd = float(np.clip( fwd   / max(dist, 1.0) * K * 16, -1.0, 1.0))
    roll_cmd  = float(np.clip( right / max(dist, 1.0) * K * 16, -1.0, 1.0))
    heading_err = float(np.degrees(np.arctan2(dy, dx))) - env.yaw
    heading_err = (heading_err + 180) % 360 - 180
    yaw_cmd  = float(np.clip(heading_err / 60.0, -1.0, 1.0))
    vert_cmd = float(np.clip(dz / 20.0, -1.0, 1.0))
    return np.array([roll_cmd, pitch_cmd, yaw_cmd, vert_cmd], dtype=np.float32)


def _run_jsbsim_episode(seed=None, demo=False, model=None, n_obstacles=3):
    env = ObstacleHelicopterEnv(
        n_obstacles=n_obstacles, obstacle_radius=15.0, obstacle_height=120.0,
        safety_margin=25.0, randomize_radius=True,
        max_steps=1200, wind_strength_max=4.0,
    )
    obs, _ = env.reset(seed=seed)

    traj    = [env.pos.copy()]
    rolls   = [env.roll]
    pitches = [env.pitch]
    yaws    = [env.yaw]

    done = False
    while not done:
        if demo:
            action = _heuristic_action(env)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        traj.append(env.pos.copy())
        rolls.append(env.roll)
        pitches.append(env.pitch)
        yaws.append(env.yaw)
        done = term or trunc

    traj    = np.array(traj)
    success = info.get("success",   False)
    collision = info.get("collision", False)
    tag = "HEDEFE ULASTI" if success else ("CARPISME" if collision else "TIMEOUT")
    src = "JSBSim DEMO" if demo else "JSBSim PPO"
    print(f"[{src}] {tag}  |  Adim: {len(traj)}")
    obstacles = list(env.get_obstacles())
    target    = env.target.copy()
    env.close()
    return traj, rolls, pitches, yaws, obstacles, target, success


def simulate_jsbsim(seed=42, demo=False):
    import time

    if demo:
        print("JSBSim DEMO MODU — heuristik kontrolcu")
        def _load(s): return _run_jsbsim_episode(seed=s, demo=True)
    else:
        model = _load_model()
        def _load(s):
            for att in range(10):
                r = _run_jsbsim_episode(seed=s + att, demo=False, model=model)
                if r[6]: return r
            return r

    tag = "JSBSim DEMO  |  Heuristik" if demo else "JSBSim PPO  |  Egitilmis Ajan"

    p.connect(p.GUI,
              options="--background_color_red=0.04 "
                      "--background_color_green=0.06 "
                      "--background_color_blue=0.12")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,                       0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,                   1)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,        0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,      0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

    traj, rolls, pitches, yaws, obstacles, target, _ = _load(seed)
    heli_id, main_idx, tail_idx = build_scene(traj, obstacles, target)

    cam_yaw    = 45.0
    cam_pitch  = -60.0
    cam_dist   = 350.0
    cam_target = list(traj[0])
    _ml = None; _lb = False; _rb = False

    p.resetDebugVisualizerCamera(
        cameraDistance=cam_dist, cameraYaw=cam_yaw,
        cameraPitch=cam_pitch, cameraTargetPosition=cam_target)

    p.addUserDebugText(
        f"HELIKOPTER RL  |  {tag}",
        [0, 0, 190], textColorRGB=[0.35, 0.85, 1.0], textSize=2.4)
    p.addUserDebugText(
        "SPACE=Dur  R=Yeni  F=Takip  +/-=Hiz  Sol-Surukle=Dondur  Sag-Surukle=Zoom",
        [0, 0, -80], textColorRGB=[0.35, 0.35, 0.35], textSize=1.1)

    print("\n  SPACE=Dur  R=Yeni bolum  F=Takip  +/-=Hiz")
    print("  Fare: Sol-surukle=Dondur  Sag-surukle=Zoom\n")

    current_seed = seed
    paused = False; follow_cam = False; speed = 1.0
    rotor_angle = 0.0; trail_start = None; txt_hud = None; i = 0

    while p.isConnected():
        for ev in p.getMouseEvents():
            et, mx, my, bi, bs = ev
            if et == 2:
                if bi == 0:
                    _lb = (bs == 3);  _ml = (mx, my) if bs == 3 else _ml
                elif bi == 2:
                    _rb = (bs == 3);  _ml = (mx, my) if bs == 3 else _ml
                elif bi == 3 and bs == 3:
                    cam_dist = max(20.0, cam_dist - 25.0)
                elif bi == 4 and bs == 3:
                    cam_dist = min(800.0, cam_dist + 25.0)
            elif et == 1 and _ml:
                dx, dy = mx - _ml[0], my - _ml[1]
                if _lb:
                    cam_yaw  += dx * 0.5
                    cam_pitch = float(np.clip(cam_pitch - dy * 0.3, -89.0, -2.0))
                elif _rb:
                    cam_dist = float(np.clip(cam_dist + dy * 2.0, 20.0, 800.0))
                _ml = (mx, my)

        keys = p.getKeyboardEvents()
        if 32 in keys and keys[32] & p.KEY_WAS_TRIGGERED:
            paused = not paused
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            current_seed += 1
            traj, rolls, pitches, yaws, obstacles, target, _ = _load(current_seed)
            p.resetSimulation()
            heli_id, main_idx, tail_idx = build_scene(traj, obstacles, target)
            p.addUserDebugText(f"HELIKOPTER RL  |  {tag}",
                               [0, 0, 190], textColorRGB=[0.35, 0.85, 1.0], textSize=2.4)
            cam_target = list(traj[0]); i = 0; trail_start = None
            rotor_angle = 0.0; txt_hud = None
        if ord('f') in keys and keys[ord('f')] & p.KEY_WAS_TRIGGERED:
            follow_cam = not follow_cam
        if (ord('+') in keys and keys[ord('+')] & p.KEY_WAS_TRIGGERED) or \
           (ord('=') in keys and keys[ord('=')] & p.KEY_WAS_TRIGGERED):
            speed = min(5.0, round(speed + 0.5, 1))
        if ord('-') in keys and keys[ord('-')] & p.KEY_WAS_TRIGGERED:
            speed = max(0.1, round(speed - 0.5, 1))

        if paused:
            p.resetDebugVisualizerCamera(
                cameraDistance=cam_dist, cameraYaw=cam_yaw,
                cameraPitch=cam_pitch, cameraTargetPosition=cam_target)
            p.stepSimulation(); time.sleep(0.05); continue

        if i >= len(traj):
            i = 0; trail_start = None; rotor_angle = 0.0

        pos = traj[i]
        rotor_angle += 0.35 * speed
        _update_heli(heli_id, main_idx, tail_idx,
                     pos, rolls[i], pitches[i], yaws[i], rotor_angle)

        if trail_start:
            p.addUserDebugLine(trail_start, pos.tolist(),
                               [0.10, 0.55, 1.0], 2.0, lifeTime=0)
        trail_start = pos.tolist()

        p.addUserDebugLine(pos.tolist(), [pos[0], pos[1], 0.5],
                           [0.20, 0.45, 0.80], 0.8, lifeTime=0.12)

        dist = float(np.linalg.norm(np.array(target) - pos))
        pct  = 100.0 * i / max(1, len(traj) - 1)
        hud  = f"JSBSim  Adim {i}/{len(traj)-1} ({pct:.0f}%)   {dist:.0f}m   {speed:.1f}x"
        if txt_hud is None:
            txt_hud = p.addUserDebugText(
                hud, (pos + np.array([0, 0, 20])).tolist(),
                textColorRGB=[0.15, 1.0, 0.5], textSize=1.4)
        else:
            p.addUserDebugText(
                hud, (pos + np.array([0, 0, 20])).tolist(),
                textColorRGB=[0.15, 1.0, 0.5], textSize=1.4,
                lifeTime=0.15, replaceItemUniqueId=txt_hud)

        if follow_cam:
            cam_target = pos.tolist()
            cam_yaw    = float(yaws[i]) - 30
            cam_pitch  = -28.0
            cam_dist   = 200.0
        p.resetDebugVisualizerCamera(
            cameraDistance=cam_dist, cameraYaw=cam_yaw,
            cameraPitch=cam_pitch, cameraTargetPosition=cam_target)

        p.stepSimulation()
        time.sleep(max(0.008, 0.04 / speed))
        i += 1

    p.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",  action="store_true",
                        help="JSBSim fizigiyle modelsiz heuristik demo")
    parser.add_argument("--live",  action="store_true",
                        help="JSBSim fizigiyle egitilmis model")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    if args.live:
        simulate_jsbsim(seed=args.seed, demo=False)
    else:
        simulate_jsbsim(seed=args.seed, demo=True)
