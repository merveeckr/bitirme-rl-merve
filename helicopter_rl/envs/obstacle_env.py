"""
ObstacleHelicopterEnv — Phase 2 / Phase 3

Extends FlightControlEnv3D with cylindrical obstacles and a simulated
8-sector LiDAR sensor.

Sensor model
------------
  LiDAR range   : 150 m
  Sectors       : 8 × 45° (relative to helicopter heading)
  Noise         : Gaussian σ = 1.5 m (models real sensor uncertainty)
  Observation   : obs[18-25] = normalised distance per sector (1.0 = clear)

This mirrors what a real on-board sensor (LiDAR / radar altimeter array)
would provide, making the learned policy transferable to physical platforms
via sim-to-real calibration.

Design Document references
--------------------------
  Section 2.1.2 : cylindrical obstacle model
  Section 2.1.1 : 25-metre safety margin
  Section 4.1.4 : collision penalty −1000
  Section 4.3   : ProactiveEvasion below 25 m
"""

import numpy as np
from .base_env import FlightControlEnv3D

# ── LiDAR constants ───────────────────────────────────────────────────────
_N_SECTORS    = 8
_SECTOR_DEG   = 360.0 / _N_SECTORS   # 45°
_LIDAR_RANGE  = 150.0                 # metres
_LIDAR_NOISE  = 0.5                   # Gaussian σ [m]


class ObstacleHelicopterEnv(FlightControlEnv3D):
    """
    Phase 2 / Phase 3 environment with cylindrical obstacles.

    Observation slots [18-25] are filled by a simulated LiDAR scan
    instead of hand-crafted scalar distances, so the agent learns from
    the same type of signal a real sensor delivers.

    Args:
        n_obstacles      : number of cylindrical obstacles
        obstacle_radius  : cylinder radius [m]
        obstacle_height  : cylinder height [m]
        safety_margin    : proximity-penalty zone [m]
        randomize_radius : vary radius per episode (domain randomisation)
    """

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

    # ──────────────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        obs_vec, info = super().reset(seed=seed, options=options)
        self._place_obstacles()
        return self._obs(), info

    # ──────────────────────────────────────────────────────────────────────
    # Obstacle placement
    # ──────────────────────────────────────────────────────────────────────

    def _place_obstacles(self):
        """
        Place n_obstacles along the start→target corridor with random
        lateral offsets. Obstacles are spread across the path so the agent
        encounters them sequentially rather than as a wall.
        """
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
            # Each obstacle gets its own segment along the path (no clustering)
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
                        self.obstacle_radius_base * 1.4,
                    ))
                    if self.randomize_radius
                    else self.obstacle_radius_base
                )

                d_start = float(np.linalg.norm(centre_xy - start))
                d_end   = float(np.linalg.norm(centre_xy - end))
                if d_start < 50.0 or d_end < 50.0:
                    continue

                overlap = any(
                    float(np.linalg.norm(centre_xy - ex["pos"][:2])) < r + ex["radius"] + 20.0
                    for ex in self.obstacles
                )
                if overlap:
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

    # ──────────────────────────────────────────────────────────────────────
    # Simulated LiDAR sensor
    # ──────────────────────────────────────────────────────────────────────

    def _lidar_scan(self) -> np.ndarray:
        """
        Simulates an 8-sector radial LiDAR mounted on the helicopter.

        Each sector covers 45° relative to the current heading:
          sector 0 = directly ahead (0°)
          sector 1 = 45° left, … sector 7 = 45° right (315°)

        Returns normalised distances in [0, 1]:
          1.0 → no obstacle detected within _LIDAR_RANGE
          0.0 → obstacle surface at the helicopter

        Gaussian noise (σ = _LIDAR_NOISE) models real sensor uncertainty.
        A cylinder can illuminate multiple adjacent sectors when it subtends
        a large angular width.
        """
        ranges = np.ones(_N_SECTORS, dtype=np.float32)
        yr     = np.deg2rad(self.yaw)

        for obs in self.obstacles:
            if self.pos[2] > obs["pos"][2] + obs["height"]:
                continue

            to_obs     = obs["pos"][:2] - self.pos[:2]
            horiz_dist = float(np.linalg.norm(to_obs))
            surf_dist  = horiz_dist - obs["radius"]

            if surf_dist > _LIDAR_RANGE:
                continue

            # Bearing of obstacle centre relative to heading (0° = ahead)
            angle_world = np.degrees(np.arctan2(to_obs[1], to_obs[0]))
            angle_rel   = (angle_world - np.degrees(yr) + 360.0) % 360.0

            # Angular half-width of the cylinder from this distance
            if horiz_dist > obs["radius"]:
                ang_half = np.degrees(np.arcsin(
                    min(obs["radius"] / horiz_dist, 1.0)))
            else:
                ang_half = 90.0

            # Sensor reading with Gaussian noise
            measured   = max(0.0, surf_dist + float(
                self.np_random.normal(0.0, _LIDAR_NOISE)))
            normalised = float(np.clip(measured / _LIDAR_RANGE, 0.0, 1.0))

            # Illuminate every sector the cylinder touches
            for s in range(_N_SECTORS):
                sector_centre = s * _SECTOR_DEG
                diff = abs((angle_rel - sector_centre + 180.0) % 360.0 - 180.0)
                if diff < (_SECTOR_DEG / 2.0 + ang_half):
                    if normalised < ranges[s]:
                        ranges[s] = normalised

        return ranges

    # ──────────────────────────────────────────────────────────────────────
    # Obstacle geometry helpers (used internally for reward / evaluation)
    # ──────────────────────────────────────────────────────────────────────

    def _obstacle_distances(self) -> tuple[float, float]:
        """
        Returns (min_surface_dist, second_min_surface_dist).
        Used by reward shaping and evaluation scripts — NOT exposed in obs.
        Using exact distances for training reward is intentional: privileged
        information during training improves sample efficiency while the
        policy input (LiDAR) stays sensor-realistic.
        """
        if not self.obstacles:
            return self.world_size, self.world_size

        dists = []
        for obs in self.obstacles:
            if self.pos[2] > obs["pos"][2] + obs["height"]:
                continue
            h = float(np.linalg.norm(obs["pos"][:2] - self.pos[:2]))
            dists.append(h - obs["radius"])

        if not dists:
            return self.world_size, self.world_size

        dists.sort()
        return dists[0], dists[1] if len(dists) > 1 else self.world_size

    _COLLISION_BUFFER = 2.0  # discrete-time safety buffer [m]

    def _is_collision(self) -> bool:
        for obs in self.obstacles:
            if self.pos[2] > obs["pos"][2] + obs["height"]:
                continue
            if float(np.linalg.norm(self.pos[:2] - obs["pos"][:2])) < obs["radius"] + self._COLLISION_BUFFER:
                return True
        return False

    # ──────────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        obs = super()._obs()          # base 26-dim (slots 18-25 = 1.0)
        obs[18:26] = self._lidar_scan()
        return obs

    # ──────────────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────────────

    def step(self, action):
        self._physics(action)
        self.step_count += 1
        self.trajectory.append(self.pos.copy())

        reward, terminated, info = self._reward()

        if not terminated:
            r_obs, t_obs, i_obs = self._obstacle_reward()
            reward    += r_obs
            terminated = terminated or t_obs
            info.update(i_obs)

        truncated = self.step_count >= self.max_steps
        return self._obs(), reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────
    # Obstacle reward (uses exact distances — privileged training signal)
    # ──────────────────────────────────────────────────────────────────────

    def _obstacle_reward(self) -> tuple[float, bool, dict]:
        """
        Design Doc 4.1.4:
          Collision    : −1000, episode ends
          Proximity    : graduated penalty inside safety_margin
                         applied to EVERY obstacle independently
        """
        info: dict = {}

        if self._is_collision():
            return -1000.0, True, {"collision": True}

        all_dists = []
        for obs in self.obstacles:
            if self.pos[2] > obs["pos"][2] + obs["height"]:
                continue
            h = float(np.linalg.norm(obs["pos"][:2] - self.pos[:2]))
            all_dists.append(h - obs["radius"])

        if not all_dists:
            return 0.0, False, info

        def _penalty(d: float) -> float:
            if d < 5.0:
                # steep inner zone — strongly discourages threading
                return -120.0 * (1.0 - d / 5.0)
            if d < self.safety_margin:
                f      = (self.safety_margin - d) / self.safety_margin
                linger = 1.0 - d / self.safety_margin
                return -50.0 * (f ** 2) - 0.3 * linger
            return 0.0

        reward = sum(_penalty(d) for d in all_dists)

        if any(d < self.safety_margin for d in all_dists):
            info["near_obstacle"] = True

        return reward, False, info

    # ──────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────

    def get_obstacles(self) -> list[dict]:
        return self.obstacles
