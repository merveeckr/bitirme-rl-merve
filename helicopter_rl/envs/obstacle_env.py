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
            for _attempt in range(100):
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
            min_surface_dist : minimum surface distance to any cylinder [m]
            forward_dist     : distance to nearest obstacle along flight direction [m]
        """
        if not self.obstacles:
            return self.world_size, self.world_size

        # Velocity / heading direction for forward projection
        vel_xy = self.vel[:2].copy()
        spd = float(np.linalg.norm(vel_xy))
        if spd > 0.5:
            fwd_dir = vel_xy / spd
        else:
            yr = np.deg2rad(self.yaw)
            fwd_dir = np.array([np.cos(yr), np.sin(yr)])

        min_surf = float("inf")
        fwd_dist = self.world_size

        for obs in self.obstacles:
            # Skip if helicopter is above this obstacle
            if self.pos[2] > obs["pos"][2] + obs["height"]:
                continue

            # Horizontal distance to cylinder axis
            to_obs_xy = obs["pos"][:2] - self.pos[:2]
            horiz_dist = float(np.linalg.norm(to_obs_xy))
            surf_dist  = horiz_dist - obs["radius"]

            if surf_dist < min_surf:
                min_surf = surf_dist

            # Forward distance: projection of obs centre onto fwd_dir
            proj   = float(np.dot(to_obs_xy, fwd_dir))
            if proj > 0.0:
                lateral = float(np.sqrt(max(0.0, np.dot(to_obs_xy, to_obs_xy) - proj * proj)))
                if lateral < obs["radius"] + 15.0:   # on collision course
                    fwd_dist = min(fwd_dist, proj)

        return min_surf, fwd_dist

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

        min_surf, fwd_dist = self._obstacle_distances()
        obs[18] = float(np.clip(min_surf  / 150.0, 0.0, 1.0))
        obs[19] = float(np.clip(fwd_dist  / 250.0, 0.0, 1.0))

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
          Safe zone      : small reward for maintaining distance > safety_margin
        """
        info: dict = {}
        terminated = False

        # Collision
        if self._is_collision():
            return -1000.0, True, {"collision": True}

        min_surf, _fwd = self._obstacle_distances()

        reward = 0.0

        if min_surf < self.safety_margin:
            # Graduated: max -50 at surface, 0 at safety_margin
            factor = (self.safety_margin - min_surf) / self.safety_margin
            reward = -50.0 * (factor ** 2)
            info["near_obstacle"] = True
        elif min_surf < self.safety_margin * 2.0:
            # Small reward for flying safely near (but not too close to) obstacle
            reward = 0.3

        return reward, terminated, info

    # ──────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────

    def get_obstacles(self) -> list[dict]:
        return self.obstacles
