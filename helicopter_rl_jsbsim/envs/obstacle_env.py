"""
ObstacleHelicopterEnv (JSBSim versiyonu)
=========================================
JSBSimHelicopterEnv üzerine silindirik engeller ve 8 sektörlü LiDAR ekler.
helicopter_rl/envs/obstacle_env.py ile özdeş mantık —
sadece base sınıfı değişti.
"""

import numpy as np
from .jsbsim_base_env import JSBSimHelicopterEnv

_N_SECTORS   = 8
_SECTOR_DEG  = 360.0 / _N_SECTORS
_LIDAR_RANGE = 150.0
_LIDAR_NOISE = 0.5


class ObstacleHelicopterEnv(JSBSimHelicopterEnv):

    def __init__(
        self,
        n_obstacles: int = 1,
        obstacle_radius: float = 15.0,
        obstacle_height: float = 120.0,
        safety_margin: float = 25.0,
        randomize_radius: bool = False,
        **kwargs,
    ):
        self.n_obstacles          = n_obstacles
        self.obstacle_radius_base = obstacle_radius
        self.obstacle_height      = obstacle_height
        self.safety_margin        = safety_margin
        self.randomize_radius     = randomize_radius
        self.obstacles: list[dict] = []
        super().__init__(**kwargs)

    # ── Reset ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        obs_vec, info = super().reset(seed=seed, options=options)
        self._place_obstacles()
        return self._obs(), info

    # ── Engel yerleştirme ─────────────────────────────────────────────

    def _place_obstacles(self):
        self.obstacles = []
        start    = self.pos[:2].copy()
        end      = self.target[:2].copy()
        path_vec = end - start
        path_len = float(np.linalg.norm(path_vec))
        if path_len < 1e-3:
            return

        path_dir = path_vec / path_len
        perp_dir = np.array([-path_dir[1], path_dir[0]])

        for i in range(self.n_obstacles):
            seg  = 0.50 / max(self.n_obstacles, 1)
            t_lo = 0.25 + i * seg
            t_hi = t_lo + seg

            placed = False
            for _ in range(100):
                t       = float(self.np_random.uniform(t_lo, t_hi))
                lateral = float(self.np_random.uniform(-50.0, 50.0))
                centre_xy = start + t * path_vec + lateral * perp_dir

                r = (
                    float(self.np_random.uniform(
                        self.obstacle_radius_base * 0.7,
                        self.obstacle_radius_base * 1.4))
                    if self.randomize_radius
                    else self.obstacle_radius_base
                )

                if (float(np.linalg.norm(centre_xy - start)) < 50.0 or
                        float(np.linalg.norm(centre_xy - end)) < 50.0):
                    continue

                if any(
                    float(np.linalg.norm(centre_xy - ex["pos"][:2]))
                    < r + ex["radius"] + 20.0
                    for ex in self.obstacles
                ):
                    continue

                self.obstacles.append({
                    "pos":    np.append(centre_xy, 0.0),
                    "radius": r,
                    "height": self.obstacle_height,
                })
                placed = True
                break

            if not placed:
                t = 0.3 + 0.4 * len(self.obstacles) / max(1, self.n_obstacles)
                self.obstacles.append({
                    "pos":    np.append(start + t * path_vec, 0.0),
                    "radius": self.obstacle_radius_base,
                    "height": self.obstacle_height,
                })

    # ── Konik dağ modeli ──────────────────────────────────────────────

    def _cone_radius(self, obs: dict, z: float) -> float:
        """Konik dağ modelinde yükseklik z'deki etkin yarıçap."""
        t = max(0.0, 1.0 - z / obs["height"])
        return obs["radius"] * t

    # ── LiDAR ─────────────────────────────────────────────────────────

    def _lidar_scan(self) -> np.ndarray:
        ranges = np.ones(_N_SECTORS, dtype=np.float32)
        yr = np.deg2rad(self.yaw)

        for obs in self.obstacles:
            if self.pos[2] > obs["pos"][2] + obs["height"]:
                continue
            r_eff      = self._cone_radius(obs, self.pos[2])
            to_obs     = obs["pos"][:2] - self.pos[:2]
            horiz_dist = float(np.linalg.norm(to_obs))
            surf_dist  = horiz_dist - r_eff
            if surf_dist > _LIDAR_RANGE:
                continue

            angle_world = np.degrees(np.arctan2(to_obs[1], to_obs[0]))
            angle_rel   = (angle_world - np.degrees(yr) + 360.0) % 360.0

            if horiz_dist > r_eff and r_eff > 0:
                ang_half = np.degrees(np.arcsin(min(r_eff / horiz_dist, 1.0)))
            elif r_eff > 0:
                ang_half = 90.0
            else:
                continue

            measured   = max(0.0, surf_dist + float(
                self.np_random.normal(0.0, _LIDAR_NOISE)))
            normalised = float(np.clip(measured / _LIDAR_RANGE, 0.0, 1.0))

            for s in range(_N_SECTORS):
                sector_centre = s * _SECTOR_DEG
                diff = abs((angle_rel - sector_centre + 180.0) % 360.0 - 180.0)
                if diff < (_SECTOR_DEG / 2.0 + ang_half):
                    if normalised < ranges[s]:
                        ranges[s] = normalised

        return ranges

    # ── Engel mesafeleri ──────────────────────────────────────────────

    def _obstacle_distances(self) -> tuple[float, float]:
        if not self.obstacles:
            return self.world_size, self.world_size
        z = self.pos[2]
        dists = [
            float(np.linalg.norm(obs["pos"][:2] - self.pos[:2])) - self._cone_radius(obs, z)
            for obs in self.obstacles
            if z <= obs["pos"][2] + obs["height"]
        ]
        if not dists:
            return self.world_size, self.world_size
        dists.sort()
        return dists[0], dists[1] if len(dists) > 1 else self.world_size

    def _is_collision(self) -> bool:
        z = self.pos[2]
        for obs in self.obstacles:
            if z > obs["pos"][2] + obs["height"]:
                continue
            r_eff = self._cone_radius(obs, z)
            if r_eff > 0 and float(np.linalg.norm(self.pos[:2] - obs["pos"][:2])) < r_eff:
                return True
        return False

    # ── Gözlem ────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        obs = super()._obs()
        obs[18:26] = self._lidar_scan()
        return obs

    # ── Step ──────────────────────────────────────────────────────────

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if not terminated:
            r_obs, t_obs, i_obs = self._obstacle_reward()
            reward    += r_obs
            terminated = terminated or t_obs
            info.update(i_obs)

        return obs, reward, terminated, truncated, info

    # ── Engel ödülü ───────────────────────────────────────────────────

    def _obstacle_reward(self) -> tuple[float, bool, dict]:
        info: dict = {}

        if self._is_collision():
            return -1000.0, True, {"collision": True}

        z = self.pos[2]
        all_dists = [
            float(np.linalg.norm(obs["pos"][:2] - self.pos[:2])) - self._cone_radius(obs, z)
            for obs in self.obstacles
            if z <= obs["pos"][2] + obs["height"]
        ]
        if not all_dists:
            return 0.0, False, info

        def _penalty(d: float) -> float:
            if d < self.safety_margin:
                f      = (self.safety_margin - d) / self.safety_margin
                linger = 1.0 - d / self.safety_margin
                return -50.0 * (f ** 2) - 0.3 * linger
            return 0.0

        reward = sum(_penalty(d) for d in all_dists)
        if any(d < self.safety_margin for d in all_dists):
            info["near_obstacle"] = True

        return reward, False, info

    def get_obstacles(self) -> list[dict]:
        return self.obstacles
