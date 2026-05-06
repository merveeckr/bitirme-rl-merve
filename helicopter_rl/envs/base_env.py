"""
FlightControlEnv3D - Phase 1: Basic 3D Helicopter Navigation

Design Document Reference: Section 4.1 (Module Design)

State vector (20-dim):
  [0-2]   dir_to_target (unit vec 3D)
  [3]     dist_to_target (normalized)
  [4-6]   velocity vx, vy, vz (normalized)
  [7-8]   roll, pitch (normalized by max_tilt)
  [9-10]  sin(yaw), cos(yaw)
  [11-13] angular rates p, q, r (normalized)
  [14-16] wind x, y, z (normalized)
  [17]    step progress (0→1)
  [18-19] obstacle distances (1.0 = no obstacle)

Action vector (4-dim continuous [-1, 1]):
  [0] roll_rate   → scaled to ±45 deg/s
  [1] pitch_rate  → scaled to ±45 deg/s
  [2] yaw_rate    → scaled to ±60 deg/s
  [3] vert_vel    → scaled to ±5 m/s

Reward:
  R_total = 10*R_progress + R_goal + 0.5*R_stability + R_time + R_bounds
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FlightControlEnv3D(gym.Env):
    """
    3D helicopter navigation environment - Phase 1 (no obstacles).
    Helicopter must reach a randomly placed 3D target.
    Uses simplified helicopter kinematics based on tilt-to-velocity mapping.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode=None,
        max_steps: int = 1000,
        world_size: float = 500.0,
        wind_strength_max: float = 3.0,
        target_radius: float = 10.0,
    ):
        super().__init__()

        # ── World parameters ──────────────────────────────────────────────
        self.world_size = world_size
        self.min_altitude = 5.0
        self.max_altitude = 200.0
        self.target_radius = target_radius
        self.max_steps = max_steps
        self.wind_strength_max = wind_strength_max

        # ── Helicopter physics parameters ─────────────────────────────────
        self.dt = 0.1                  # simulation time-step  [s]
        self.max_tilt = 30.0           # max roll / pitch      [deg]
        self.max_tilt_rate = 45.0      # max tilt rate         [deg/s]
        self.max_yaw_rate = 60.0       # max yaw rate          [deg/s]
        self.max_vert_speed = 5.0      # max vertical speed    [m/s]
        self.max_horiz_speed = 20.0    # max horizontal speed  [m/s]

        # ── Gym spaces ───────────────────────────────────────────────────
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )

        self.render_mode = render_mode
        self._init_internals()

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _init_internals(self):
        self.pos = np.zeros(3, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll_rate = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate = 0.0
        self.target = np.zeros(3, dtype=np.float64)
        self.wind = np.zeros(3, dtype=np.float64)
        self.step_count = 0
        self.prev_dist = 0.0
        self.trajectory: list = []

    # ──────────────────────────────────────────────────────────────────────
    # Gym interface
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random start position (some altitude)
        self.pos = np.array([
            self.np_random.uniform(-150.0, 150.0),
            self.np_random.uniform(-150.0, 150.0),
            self.np_random.uniform(20.0, 60.0),
        ], dtype=np.float64)

        # Random target (80–380 m away)
        for _ in range(200):
            self.target = np.array([
                self.np_random.uniform(-200.0, 200.0),
                self.np_random.uniform(-200.0, 200.0),
                self.np_random.uniform(15.0, 100.0),
            ], dtype=np.float64)
            d = np.linalg.norm(self.target - self.pos)
            if 80.0 <= d <= 380.0:
                break

        # Dynamics reset
        self.vel[:] = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = float(self.np_random.uniform(0.0, 360.0))
        self.roll_rate = self.pitch_rate = self.yaw_rate = 0.0

        # Light random wind
        ws = float(self.np_random.uniform(0.0, self.wind_strength_max))
        wd = float(self.np_random.uniform(0.0, 2.0 * np.pi))
        self.wind = np.array([
            ws * np.cos(wd),
            ws * np.sin(wd),
            float(self.np_random.uniform(-0.3, 0.3)),
        ])

        self.step_count = 0
        self.prev_dist = float(np.linalg.norm(self.target - self.pos))
        self.trajectory = [self.pos.copy()]

        return self._obs(), {}

    def step(self, action):
        self._physics(action)
        self.step_count += 1
        self.trajectory.append(self.pos.copy())

        reward, terminated, info = self._reward()
        truncated = self.step_count >= self.max_steps

        return self._obs(), reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        delta = self.target - self.pos
        dist = float(np.linalg.norm(delta)) + 1e-8
        dir_vec = (delta / dist).astype(np.float32)

        obs = np.array([
            dir_vec[0], dir_vec[1], dir_vec[2],               # 0-2  direction to target
            dist / self.world_size,                            # 3    distance (norm)
            self.vel[0] / self.max_horiz_speed,               # 4    vx
            self.vel[1] / self.max_horiz_speed,               # 5    vy
            self.vel[2] / self.max_vert_speed,                # 6    vz
            self.roll  / self.max_tilt,                       # 7    roll
            self.pitch / self.max_tilt,                       # 8    pitch
            np.sin(np.deg2rad(self.yaw)),                     # 9    sin(yaw)
            np.cos(np.deg2rad(self.yaw)),                     # 10   cos(yaw)
            self.roll_rate  / self.max_tilt_rate,             # 11   p
            self.pitch_rate / self.max_tilt_rate,             # 12   q
            self.yaw_rate   / self.max_yaw_rate,              # 13   r
            self.wind[0] / 10.0,                              # 14   wind x
            self.wind[1] / 10.0,                              # 15   wind y
            self.wind[2] / 3.0,                               # 16   wind z
            self.step_count / self.max_steps,                 # 17   progress
            1.0,  # obstacle slot – overridden by subclass    # 18   min obs dist
            1.0,  # obstacle slot – overridden by subclass    # 19   fwd obs dist
        ], dtype=np.float32)

        return obs

    # ──────────────────────────────────────────────────────────────────────
    # Physics
    # ──────────────────────────────────────────────────────────────────────

    def _physics(self, action):
        """
        Simplified tilt-to-velocity helicopter model.
        Pitch forward → move in heading direction.
        Roll right    → move perpendicular (right) to heading.
        Direct vertical velocity control (collective proxy).
        """
        self.roll_rate  = float(action[0]) * self.max_tilt_rate
        self.pitch_rate = float(action[1]) * self.max_tilt_rate
        self.yaw_rate   = float(action[2]) * self.max_yaw_rate
        vert_cmd        = float(action[3]) * self.max_vert_speed

        # Update orientation
        self.roll  = float(np.clip(self.roll  + self.roll_rate  * self.dt,
                                   -self.max_tilt, self.max_tilt))
        self.pitch = float(np.clip(self.pitch + self.pitch_rate * self.dt,
                                   -self.max_tilt, self.max_tilt))
        self.yaw   = (self.yaw + self.yaw_rate * self.dt) % 360.0

        # Tilt fractions → target horizontal velocity (world frame)
        yr = np.deg2rad(self.yaw)
        pf = self.pitch / self.max_tilt   # [-1, 1]
        rf = self.roll  / self.max_tilt   # [-1, 1]

        vx_target = self.max_horiz_speed * ( pf * np.cos(yr) - rf * np.sin(yr))
        vy_target = self.max_horiz_speed * ( pf * np.sin(yr) + rf * np.cos(yr))

        # Low-pass filter on horizontal velocity
        alpha = 0.4
        self.vel[0] = alpha * vx_target + (1.0 - alpha) * self.vel[0]
        self.vel[1] = alpha * vy_target + (1.0 - alpha) * self.vel[1]
        self.vel[2] = vert_cmd

        # Wind perturbation + aerodynamic drag
        self.vel[:2] += self.wind[:2] * 0.05
        self.vel[:2] *= 0.97

        # Integrate position
        self.pos = self.pos + self.vel * self.dt
        self.pos[2] = float(np.clip(self.pos[2], self.min_altitude, self.max_altitude))

    # ──────────────────────────────────────────────────────────────────────
    # Reward (Design Doc Section 4.1.4)
    # ──────────────────────────────────────────────────────────────────────

    def _reward(self):
        """
        Multi-objective reward:
          R_progress  — potential-based shaping (prev_dist - curr_dist)
          R_goal      — large bonus on reaching target
          R_stability — penalise excessive tilt
          R_time      — small per-step penalty for efficiency
          R_bounds    — heavy penalty for leaving world
        """
        curr_dist = float(np.linalg.norm(self.target - self.pos))
        info: dict = {}
        terminated = False

        # Progress (potential shaping)
        r_progress = (self.prev_dist - curr_dist) * 10.0
        self.prev_dist = curr_dist

        # Goal
        r_goal = 0.0
        if curr_dist < self.target_radius:
            r_goal = 500.0
            terminated = True
            info["success"] = True

        # Stability penalty
        r_stability = -(
            abs(self.roll)  / self.max_tilt +
            abs(self.pitch) / self.max_tilt
        ) * 0.5

        # Altitude floor penalty
        r_altitude = -2.0 if self.pos[2] <= self.min_altitude + 2.0 else 0.0

        # Out-of-bounds penalty
        r_bounds = 0.0
        if np.any(np.abs(self.pos[:2]) > self.world_size):
            r_bounds = -200.0
            terminated = True
            info["out_of_bounds"] = True

        # Time penalty (efficiency)
        r_time = -0.1

        total = r_progress + r_goal + r_stability + r_altitude + r_bounds + r_time
        return total, terminated, info

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    def get_trajectory(self) -> np.ndarray:
        return np.array(self.trajectory)

    def render(self):
        pass

    def close(self):
        pass
