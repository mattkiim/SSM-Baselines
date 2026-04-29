from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from jaxrl5.envs.dynamics.quad3d import step_dynamics_3d
from jaxrl5.wrappers.action_rescale import SymmetricActionWrapper


class QuadrotorTracking3DEnv(gym.Env):
    """3D quadrotor tracking a reference trajectory with RAC-style safety info.

    State:      [px, py, pz, vx, vy, vz, phi, theta, psi]
    Reference:  [px_ref, py_ref, pz_ref, vx_ref, vy_ref, vz_ref, phi_ref, theta_ref, psi_ref]
    Observation = [state, ref]
    Action:     [f, phi_dot, theta_dot, psi_dot]

    Notes:
    - This follows the simplified 3D control-affine model you attached.
    - pz is treated as positive upward here to stay consistent with the rest of your new envs.
    - `info["h"]` is a continuous safety signal; `info["cost"]` is binary.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 1.0 / 60.0,
        g: float = 9.81,
        m: float = 1.0,
        max_episode_steps: int = 360,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        # reference trajectory
        trajectory_type: str = "circle",          # {"circle", "figure8", "square"}
        trajectory_plane: str = "xy",             # {"xy", "xz", "yz"}
        num_cycles: int = 1,
        trajectory_scale: float = 0.5,
        trajectory_position_offset: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        # action limits
        max_thrust: float = 20.0,
        max_angle_rate: float = 2.0,              # rad/s
        # safety
        safe_radius: float = 3.0,
        hard_radius: float = 3.5,
        safe_z_min: float = 0.0,
        hard_z_min: float = -0.3,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode

        self.dt = dt
        self.g = g
        self.m = m
        self.max_episode_steps = max_episode_steps

        self.trajectory_type = trajectory_type
        self.trajectory_plane = trajectory_plane
        self.num_cycles = num_cycles
        self.trajectory_scale = trajectory_scale
        self.trajectory_position_offset = np.asarray(
            trajectory_position_offset, dtype=np.float32
        )

        self.max_thrust = max_thrust
        self.max_angle_rate = max_angle_rate

        self.safe_radius = safe_radius
        self.hard_radius = hard_radius
        self.safe_z_min = safe_z_min
        self.hard_z_min = hard_z_min

        # obs = [state(9), ref(9)]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )

        # physical action space; wrap later to symmetric [-1, 1]
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -max_angle_rate, -max_angle_rate, -max_angle_rate], dtype=np.float32),
            high=np.array([max_thrust, max_angle_rate, max_angle_rate, max_angle_rate], dtype=np.float32),
            dtype=np.float32,
        )

        self.dyn_params: Dict[str, float] = {
            "m": self.m,
            "g": self.g,
        }

        # tracking reward weights
        self.Q = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01]).astype(np.float32)
        self.R = np.diag([1e-2, 1e-3, 1e-3, 1e-3]).astype(np.float32)

        # hover / zero-rate reference action
        self.a_ref = np.array([self.m * self.g, 0.0, 0.0, 0.0], dtype=np.float32)

        self.state = np.zeros(9, dtype=np.float32)
        self._waypoints = self._create_waypoints()
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._waypoint_idx = 0

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def _create_waypoints(self) -> np.ndarray:
        traj_length = self.max_episode_steps * self.dt
        times = np.arange(0.0, traj_length + self.dt, self.dt, dtype=np.float32)

        pos_ref = np.zeros((len(times), 3), dtype=np.float32)
        vel_ref = np.zeros((len(times), 3), dtype=np.float32)

        traj_period = traj_length / self.num_cycles
        freq = 2.0 * np.pi / traj_period

        for i, t in enumerate(times):
            pa, pb, va, vb = self._trajectory_coordinates(t, freq)
            pos_ref[i], vel_ref[i] = self._embed_in_plane(pa, pb, va, vb)

        pos_ref = pos_ref + self.trajectory_position_offset[None, :]

        # reference orientation is zero for now
        zeros = np.zeros((len(times),), dtype=np.float32)

        waypoints = np.stack(
            [
                pos_ref[:, 0],  # px_ref
                pos_ref[:, 1],  # py_ref
                pos_ref[:, 2],  # pz_ref
                vel_ref[:, 0],  # vx_ref
                vel_ref[:, 1],  # vy_ref
                vel_ref[:, 2],  # vz_ref
                zeros,          # phi_ref
                zeros,          # theta_ref
                zeros,          # psi_ref
            ],
            axis=1,
        )
        return waypoints.astype(np.float32)

    def _trajectory_coordinates(self, t: float, freq: float):
        s = self.trajectory_scale

        if self.trajectory_type == "circle":
            a = s * np.cos(freq * t)
            b = s * np.sin(freq * t)
            a_dot = -s * freq * np.sin(freq * t)
            b_dot =  s * freq * np.cos(freq * t)
            return a, b, a_dot, b_dot

        if self.trajectory_type == "figure8":
            a = s * np.sin(freq * t)
            b = s * np.sin(freq * t) * np.cos(freq * t)
            a_dot = s * freq * np.cos(freq * t)
            b_dot = s * freq * (np.cos(freq * t) ** 2 - np.sin(freq * t) ** 2)
            return a, b, a_dot, b_dot

        if self.trajectory_type == "square":
            segment_period = (2.0 * np.pi / freq) / 4.0
            traverse_speed = s / segment_period
            cycle_time = t % (4.0 * segment_period)
            segment_time = cycle_time % segment_period
            segment_index = int(np.floor(cycle_time / segment_period))
            seg_pos = traverse_speed * segment_time

            if segment_index == 0:
                return 0.0, seg_pos, 0.0, traverse_speed
            if segment_index == 1:
                return -seg_pos, s, -traverse_speed, 0.0
            if segment_index == 2:
                return -s, s - seg_pos, 0.0, -traverse_speed
            return -s + seg_pos, 0.0, traverse_speed, 0.0

        raise ValueError(f"Unknown trajectory_type: {self.trajectory_type}")

    def _embed_in_plane(self, a: float, b: float, a_dot: float, b_dot: float):
        pos = np.zeros(3, dtype=np.float32)
        vel = np.zeros(3, dtype=np.float32)

        if self.trajectory_plane == "xy":
            pos[0], pos[1] = a, b
            vel[0], vel[1] = a_dot, b_dot
        elif self.trajectory_plane == "xz":
            pos[0], pos[2] = a, b
            vel[0], vel[2] = a_dot, b_dot
        elif self.trajectory_plane == "yz":
            pos[1], pos[2] = a, b
            vel[1], vel[2] = a_dot, b_dot
        else:
            raise ValueError(f"Unknown trajectory_plane: {self.trajectory_plane}")

        return pos, vel

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)

        low = np.array(
            [-0.5, -0.5,  0.5,  -0.2, -0.2, -0.2,  -0.1, -0.1, -0.1],
            dtype=np.float32,
        )
        high = np.array(
            [ 0.5,  0.5,  1.5,   0.2,  0.2,  0.2,   0.1,  0.1,  0.1],
            dtype=np.float32,
        )

        state = self._rng.uniform(low, high).astype(np.float32)

        options = options or {}
        keys = [
            "init_px", "init_py", "init_pz",
            "init_vx", "init_vy", "init_vz",
            "init_phi", "init_theta", "init_psi",
        ]
        for i, k in enumerate(keys):
            if k in options:
                state[i] = float(options[k])

        self.state = state
        self._t = 0

        if "init_waypoint_idx" in options:
            self._waypoint_idx = int(options["init_waypoint_idx"])
        else:
            self._waypoint_idx = self._nearest_waypoint_idx(self.state[:3])

        obs = self._get_observation()
        info: Dict[str, Any] = {"idx": int(self._waypoint_idx)}
        return obs, info

    def _nearest_waypoint_idx(self, pos: np.ndarray) -> int:
        deltas = self._waypoints[:, :3] - pos[None, :]
        idx = int(np.argmin(np.sum(np.square(deltas), axis=1)))
        return idx

    def _get_observation(self) -> np.ndarray:
        ref = self._waypoints[self._waypoint_idx]
        return np.concatenate([self.state, ref]).astype(np.float32)

    def _continuous_safety_value(self, state: np.ndarray) -> float:
        px, py, pz = float(state[0]), float(state[1]), float(state[2])
        radius = np.sqrt(px * px + py * py + pz * pz)

        h_floor = self.safe_z_min - pz          # violated if pz < safe_z_min
        h_radius = radius - self.safe_radius    # violated if radius > safe_radius
        return float(max(h_floor, h_radius))

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.state = step_dynamics_3d(self.state, action, self.dt, self.dyn_params)

        self._waypoint_idx = (self._waypoint_idx + 1) % len(self._waypoints)
        obs = self._get_observation()

        ref = self._waypoints[self._waypoint_idx]
        
        
        action_err = action - self.a_ref
        reward = -float(state_err @ self.Q @ state_err + action_err @ self.R @ action_err)

        px, py, pz = float(self.state[0]), float(self.state[1]), float(self.state[2])
        radius = np.sqrt(px * px + py * py + pz * pz)

        h = self._continuous_safety_value(self.state)
        cost = float((pz < self.safe_z_min) or (radius > self.safe_radius))

        terminated = bool((pz < self.hard_z_min) or (radius > self.hard_radius))
        self._t += 1
        truncated = bool(self._t >= self.max_episode_steps)

        info: Dict[str, Any] = {
            "h": float(h),
            "cost": float(cost),
            "radius": float(radius),
            "pz": float(pz),
            "idx": int(self._waypoint_idx),
        }

        return obs, reward, terminated, truncated, info


def make_quadrotor_tracking_3d_env(**kwargs) -> gym.Env:
    env = QuadrotorTracking3DEnv(**kwargs)
    env = SymmetricActionWrapper(env)
    return env