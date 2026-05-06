"""
ObstacleHelicopterEnv - Phase 2 (1 obstacle) & Phase 3 (N obstacles)

Extends FlightControlEnv3D with cylindrical obstacles.

Design Document Reference:
  - Section 2.1.2: Obstacles are cylindrical geometric structures
  - Section 2.1.1: 25-metre safety margin
  - Section 4.1.4: Safety penalty = -1000 on collision
  - Section 4.3:   ProactiveEvasion state triggered below 25m

Obstacle model:
  - Vertical cylinder anchored at ground level
  - radius, height configurable per episode
  - Proactive safety: graduated penalty 0…50 in [25m, 0] zone
"""

import numpy as np
from .base_env import FlightControlEnv3D


class ObstacleHelicopterEnv(FlightControlEnv3D):
    """
    Phase 2 / Phase 3 environment with cylindrical obstacles.

    Args:
        n_obstacles      : number of obstacles (1 = Phase 2, >1 = Phase 3)
        obstacle_radius  : cylinder radius  [m]
        obstacle_height  : cylinder height  [m]
        safety_margin    : proactive penalty zone radius  [m]
        randomize_radius : if True, radius varies per episode
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
        self.n_obstacles = n_obstacles
        self.obstacle_radius_base = obstacle_radius
        self.obstacle_height = obstacle_height
        self.safety_margin = safety_margin
        self.randomize_radius = randomize_radius

        # Will be populated in reset()
        self.obstacles: list[dict] = []

        super().__init__(**kwargs)

    # ──────────────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        obs_vec, info = super().reset(seed=seed, options=options)
        self._place_obstacles()
        return self._obs(), info   # re-compute obs with obstacle info

    # ──────────────────────────────────────────────────────────────────────
    # Obstacle placement
    # ──────────────────────────────────────────────────────────────────────

    def _place_obstacles(self):
        """
        Place n_obstacles cylindrical obstacles along the start→target path
        with random lateral offsets, ensuring they don't overlap and are not
        too close to start / target.
        """
        self.obstacles = []

        start = self.pos[:2].copy()
        end   = self.target[:2].copy()
        path_vec = end - start
        path_len = float(np.linalg.norm(path_vec))

        if path_len < 1e-3:
            return

        path_dir = path_vec / path_len
        perp_dir = np.array([-path_dir[1], path_dir[0]])   # 90° CCW

        for _ in range(self.n_obstacles):
            placed = False
            for _ in range(100):
                # Position along path
                t = float(self.np_random.uniform(0.25, 0.75))
                # Random lateral offset
                lateral = float(self.np_random.uniform(-60.0, 60.0))

                centre_xy = start + t * path_vec + lateral * perp_dir

                r = (
                    float(self.np_random.uniform(
                        self.obstacle_radius_base * 0.7,
                        self.obstacle_radius_base * 1.4,
                    ))
                    if self.randomize_radius
                    else self.obstacle_radius_base
                )

                # Minimum clearance from start / target
                d_start = float(np.linalg.norm(centre_xy - start))
                d_end   = float(np.linalg.norm(centre_xy - end))
                if d_start < 50.0 or d_end < 50.0:
                    continue

                # No overlap with already-placed obstacles
                overlap = False
                for ex in self.obstacles:
                    d = float(np.linalg.norm(centre_xy - ex["pos"][:2]))
                    if d < (r + ex["radius"] + 10.0):
                        overlap = True
                        break
                if overlap:
                    continue

                self.obstacles.append({
                    "pos":    np.append(centre_xy, 0.0),   # ground-anchored
                    "radius": r,
                    "height": self.obstacle_height,
                })
                placed = True
                break

            if not placed:
                # Fallback: place on straight path midpoint
                t = 0.3 + 0.4 * len(self.obstacles) / max(1, self.n_obstacles)
                centre_xy = start + t * path_vec
                self.obstacles.append({
                    "pos":    np.append(centre_xy, 0.0),
                    "radius": self.obstacle_radius_base,
                    "height": self.obstacle_height,
                })

    # ──────────────────────────────────────────────────────────────────────
    # Obstacle geometry helpers
    # ──────────────────────────────────────────────────────────────────────

    def _obstacle_distances(self) -> tuple[float, float]:
        """
        Returns:
            surf_dist_nearest : surface distance to the nearest in-range cylinder [m]
            bearing_sin       : sin(angle from heading to nearest obstacle centre)
                                > 0 → obstacle is to the LEFT  of current heading
                                < 0 → obstacle is to the RIGHT of current heading
                                0.0 when no obstacle is in range
        When the agent clears the nearest obstacle it automatically tracks the next,
        so two obstacles are handled sequentially with full directional awareness.
        """
        if not self.obstacles:
            return self.world_size, 0.0

        yr = np.deg2rad(self.yaw)
        fwd = np.array([np.cos(yr), np.sin(yr)])   # unit heading vector

        best_dist    = self.world_size
        best_bearing = 0.0

        for obs in self.obstacles:
            if self.pos[2] > obs["pos"][2] + obs["height"]:
                continue
            to_obs_xy  = obs["pos"][:2] - self.pos[:2]
            horiz_dist = float(np.linalg.norm(to_obs_xy))
            surf_dist  = horiz_dist - obs["radius"]
            if surf_dist < best_dist:
                best_dist = surf_dist
                if horiz_dist > 1e-3:
                    d_norm = to_obs_xy / horiz_dist
                    # 2-D cross product: fwd × d_norm = sin of angle (left > 0)
                    best_bearing = float(fwd[0] * d_norm[1] - fwd[1] * d_norm[0])
                else:
                    best_bearing = 0.0

        return best_dist, best_bearing

    def _is_collision(self) -> bool:
        for obs in self.obstacles:
            if self.pos[2] > obs["pos"][2] + obs["height"]:
                continue
            horiz = float(np.linalg.norm(self.pos[:2] - obs["pos"][:2]))
            if horiz < obs["radius"]:
                return True
        return False

    # ──────────────────────────────────────────────────────────────────────
    # Overridden observation (adds obstacle distances at slots 18-19)
    # ──────────────────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        obs = super()._obs()

        surf_dist, bearing_sin = self._obstacle_distances()
        obs[18] = float(np.clip(surf_dist / 150.0, 0.0, 1.0))
        obs[19] = float(np.clip(bearing_sin, -1.0, 1.0))   # ±1: left/right of heading

        return obs

    # ──────────────────────────────────────────────────────────────────────
    # Overridden step (adds obstacle penalty)
    # ──────────────────────────────────────────────────────────────────────

    def step(self, action):
        self._physics(action)
        self.step_count += 1
        self.trajectory.append(self.pos.copy())

        reward, terminated, info = self._reward()          # base reward

        if not terminated:
            r_obs, t_obs, i_obs = self._obstacle_reward()
            reward    += r_obs
            terminated = terminated or t_obs
            info.update(i_obs)

        truncated = self.step_count >= self.max_steps
        return self._obs(), reward, terminated, truncated, info

    def _obstacle_reward(self) -> tuple[float, bool, dict]:
        """
        Design Doc 4.1.4:
          Safety Penalty : -1000 on collision
          Proximity zone : graduated penalty in [safety_margin, 0] band
                           applied independently to EACH obstacle so the agent
                           learns to stay clear of all of them simultaneously
          Safe zone      : small reward for maintaining distance > safety_margin
        """
        info: dict = {}
        terminated = False

        if self._is_collision():
            return -1000.0, True, {"collision": True}

        # Collect raw surface distances for ALL obstacles (independent of obs encoding)
        all_surf_dists = []
        for obs in self.obstacles:
            if self.pos[2] > obs["pos"][2] + obs["height"]:
                continue
            horiz = float(np.linalg.norm(obs["pos"][:2] - self.pos[:2]))
            all_surf_dists.append(horiz - obs["radius"])

        if not all_surf_dists:
            return 0.0, False, info

        def _proximity_penalty(d: float) -> float:
            if d < self.safety_margin:
                factor = (self.safety_margin - d) / self.safety_margin
                linger = 1.0 - d / self.safety_margin
                return -50.0 * (factor ** 2) - 0.3 * linger
            return 0.0

        reward = sum(_proximity_penalty(d) for d in all_surf_dists)

        if any(d < self.safety_margin for d in all_surf_dists):
            info["near_obstacle"] = True

        return reward, terminated, info

    # ──────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────

    def get_obstacles(self) -> list[dict]:
        return self.obstacles
