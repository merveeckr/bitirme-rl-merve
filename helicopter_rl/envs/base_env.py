"""
FlightControlEnv3D - Phase 1: Basic 3D Helicopter Navigation

Design Document Reference: Section 4.1 (Module Design)

State vector (26-dim):
  [0-2]   dir_to_target (unit vec 3D)
  [3]     dist_to_target (normalized)
  [4-6]   velocity vx, vy, vz (normalized)
  [7-8]   roll, pitch (normalized by max_tilt)
  [9-10]  sin(yaw), cos(yaw)
  [11-13] angular rates p, q, r (normalized)
  [14-16] wind x, y, z (normalized)
  [17]    step progress (0→1)
  [18-25] LiDAR sectors — 8 × normalised distance (1.0 = clear / no obstacle)
           Sector order (relative to heading):
             18: 0°  ahead        22: 180° behind
             19: 45° front-left   23: 225° rear-right
             20: 90° left         24: 270° right
             21: 135° rear-left   25: 315° front-right

Action vector (4-dim continuous [-1, 1]):
  [0] roll_rate   → ±45 deg/s
  [1] pitch_rate  → ±45 deg/s
  [2] yaw_rate    → ±60 deg/s
  [3] vert_vel    → ±5 m/s

Reward:
  R_total = 10*R_progress + R_goal + R_stability + R_time + R_bounds + R_path
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FlightControlEnv3D(gym.Env):
    """
    3D helicopter navigation — Phase 1 (no obstacles).
    Helicopter must reach a randomly placed 3D target.

    Physics: inertial tilt-to-velocity model with aerodynamic drag.
    Reward includes a path-deviation term so the agent learns to return
    to the straight-line route after manoeuvring around obstacles.
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
        self.world_size       = world_size
        self.min_altitude     = 5.0
        self.max_altitude     = 200.0
        self.target_radius    = target_radius
        self.max_steps        = max_steps
        self.wind_strength_max = wind_strength_max

        # ── Helicopter physics parameters ─────────────────────────────────
        self.dt               = 0.1     # simulation time-step [s]
        self.max_tilt         = 30.0    # max roll / pitch     [deg]
        self.max_tilt_rate    = 45.0    # [deg/s]
        self.max_yaw_rate     = 60.0    # [deg/s]
        self.max_vert_speed   = 5.0     # [m/s]
        self.max_horiz_speed  = 20.0    # [m/s]
        # Velocity filter coefficient — lower = more inertia (realistic lag)
        self._vel_alpha       = 0.2

        # ── Gym spaces ───────────────────────────────────────────────────
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
        )

        self.render_mode = render_mode
        self._init_internals()

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _init_internals(self):
        self.pos        = np.zeros(3, dtype=np.float64)
        self.vel        = np.zeros(3, dtype=np.float64)
        self.roll       = 0.0
        self.pitch      = 0.0
        self.yaw        = 0.0
        self.roll_rate  = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate   = 0.0
        self.target     = np.zeros(3, dtype=np.float64)
        self.wind       = np.zeros(3, dtype=np.float64)
        self.step_count = 0
        self.prev_dist  = 0.0
        self.trajectory: list = []
        self.episode_start = np.zeros(2, dtype=np.float64)

    # ──────────────────────────────────────────────────────────────────────
    # Gym interface
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random start position
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
            if 80.0 <= np.linalg.norm(self.target - self.pos) <= 380.0:
                break

        # Dynamics reset
        self.vel[:]  = 0.0
        self.roll    = 0.0
        self.pitch   = 0.0
        self.yaw     = float(self.np_random.uniform(0.0, 360.0))
        self.roll_rate = self.pitch_rate = self.yaw_rate = 0.0

        # Random wind
        ws = float(self.np_random.uniform(0.0, self.wind_strength_max))
        wd = float(self.np_random.uniform(0.0, 2.0 * np.pi))
        self.wind = np.array([
            ws * np.cos(wd),
            ws * np.sin(wd),
            float(self.np_random.uniform(-0.3, 0.3)),
        ])

        self.step_count    = 0
        self.prev_dist     = float(np.linalg.norm(self.target - self.pos))
        self.episode_start = self.pos[:2].copy()
        self.trajectory    = [self.pos.copy()]

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
        delta   = self.target - self.pos
        dist    = float(np.linalg.norm(delta)) + 1e-8
        dir_vec = (delta / dist).astype(np.float32)

        obs = np.array([
            dir_vec[0], dir_vec[1], dir_vec[2],          # 0-2
            dist / self.world_size,                       # 3
            self.vel[0] / self.max_horiz_speed,           # 4
            self.vel[1] / self.max_horiz_speed,           # 5
            self.vel[2] / self.max_vert_speed,            # 6
            self.roll   / self.max_tilt,                  # 7
            self.pitch  / self.max_tilt,                  # 8
            np.sin(np.deg2rad(self.yaw)),                 # 9
            np.cos(np.deg2rad(self.yaw)),                 # 10
            self.roll_rate  / self.max_tilt_rate,         # 11
            self.pitch_rate / self.max_tilt_rate,         # 12
            self.yaw_rate   / self.max_yaw_rate,          # 13
            self.wind[0] / 10.0,                          # 14
            self.wind[1] / 10.0,                          # 15
            self.wind[2] / 3.0,                           # 16
            self.step_count / self.max_steps,             # 17
            # [18-25] LiDAR sectors — subclass fills these; default = 1.0 (clear)
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ], dtype=np.float32)

        return obs

    # ──────────────────────────────────────────────────────────────────────
    # Physics — inertial tilt-to-velocity model
    # ──────────────────────────────────────────────────────────────────────

    def _physics(self, action):
        """
        Helicopter kinematics with velocity lag (first-order inertia).
        Lower _vel_alpha → more inertia → more realistic rotor response.
        """
        self.roll_rate  = float(action[0]) * self.max_tilt_rate
        self.pitch_rate = float(action[1]) * self.max_tilt_rate
        self.yaw_rate   = float(action[2]) * self.max_yaw_rate
        vert_cmd        = float(action[3]) * self.max_vert_speed

        # Update orientation
        self.roll  = float(np.clip(
            self.roll  + self.roll_rate  * self.dt, -self.max_tilt, self.max_tilt))
        self.pitch = float(np.clip(
            self.pitch + self.pitch_rate * self.dt, -self.max_tilt, self.max_tilt))
        self.yaw   = (self.yaw + self.yaw_rate * self.dt) % 360.0

        # Tilt → target horizontal velocity (world frame)
        yr = np.deg2rad(self.yaw)
        pf = self.pitch / self.max_tilt
        rf = self.roll  / self.max_tilt

        vx_target = self.max_horiz_speed * ( pf * np.cos(yr) - rf * np.sin(yr))
        vy_target = self.max_horiz_speed * ( pf * np.sin(yr) + rf * np.cos(yr))

        # First-order inertial filter (α=0.2 → ~0.5 s time constant)
        a = self._vel_alpha
        self.vel[0] = a * vx_target + (1.0 - a) * self.vel[0]
        self.vel[1] = a * vy_target + (1.0 - a) * self.vel[1]
        self.vel[2] = vert_cmd

        # Wind + aerodynamic drag
        self.vel[:2] += self.wind[:2] * 0.05
        self.vel[:2] *= 0.97

        self.pos     = self.pos + self.vel * self.dt
        self.pos[2]  = float(np.clip(self.pos[2], self.min_altitude, self.max_altitude))

    # ──────────────────────────────────────────────────────────────────────
    # Reward
    # ──────────────────────────────────────────────────────────────────────

    def _reward(self):
        curr_dist  = float(np.linalg.norm(self.target - self.pos))
        info: dict = {}
        terminated = False

        # Progress
        r_progress = (self.prev_dist - curr_dist) * 10.0
        self.prev_dist = curr_dist

        # Goal
        r_goal = 0.0
        if curr_dist < self.target_radius:
            r_goal     = 500.0
            terminated = True
            info["success"] = True

        # Stability
        r_stability = -(abs(self.roll) / self.max_tilt +
                        abs(self.pitch) / self.max_tilt) * 0.5

        # Altitude floor
        r_altitude = -2.0 if self.pos[2] <= self.min_altitude + 2.0 else 0.0

        # Out-of-bounds
        r_bounds = 0.0
        if np.any(np.abs(self.pos[:2]) > self.world_size):
            r_bounds   = -200.0
            terminated = True
            info["out_of_bounds"] = True

        # Time penalty
        r_time = -0.1

        # Path deviation — gentle nudge to return to straight-line route
        r_path = self._path_deviation_reward()

        total = r_progress + r_goal + r_stability + r_altitude + r_bounds + r_time + r_path
        return total, terminated, info

    def _path_deviation_reward(self) -> float:
        """
        Penalises lateral distance from the straight line episode_start → target.
        Encourages the agent to return to the optimal route after obstacle manoeuvres.
        Kept small so obstacle avoidance always takes priority.
        """
        path_vec = self.target[:2] - self.episode_start
        path_len = float(np.linalg.norm(path_vec))
        if path_len < 1e-3:
            return 0.0
        path_dir    = path_vec / path_len
        pos_vec     = self.pos[:2] - self.episode_start
        proj        = float(np.dot(pos_vec, path_dir))
        lateral     = pos_vec - proj * path_dir
        lateral_dist = float(np.linalg.norm(lateral))
        return -0.02 * lateral_dist / 100.0

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    def get_trajectory(self) -> np.ndarray:
        return np.array(self.trajectory)

    def render(self):
        pass

    def close(self):
        pass
