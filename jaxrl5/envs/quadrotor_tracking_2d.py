from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from jaxrl5.envs.dynamics.quad2d import step_dynamics
from jaxrl5.wrappers.action_rescale import SymmetricActionWrapper


class QuadrotorTracking2DEnv(gym.Env):
    """Planar quadrotor tracking a circular trajectory with altitude constraint.

    The implementation follows the task specification in the RCRL paper (Appendix D.1).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 1.0 / 60.0,
        g: float = 9.81,
        m: float = 1.0,
        I: float = 0.02,
        thrust_scale: float = None,
        torque_scale: float = 0.1,
        waypoint_center: Tuple[float, float] = (0.0, 1.0),
        waypoint_radius: float = 1.0,
        max_episode_steps: int = 360,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode

        self.dt = dt
        self.g = g
        self.m = m
        self.I = I
        self.thrust_scale = thrust_scale if thrust_scale is not None else self.m * self.g
        self.torque_scale = torque_scale
        self.center = waypoint_center
        self.radius = waypoint_radius
        self.max_episode_steps = max_episode_steps

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.dyn_params: Dict[str, float] = {
            "m": self.m,
            "I": self.I,
            "g": self.g,
            "thrust_scale": self.thrust_scale,
            "torque_scale": self.torque_scale,
            "action_low": float(self.action_space.low[0]),
            "action_high": float(self.action_space.high[0]),
        }

        # self.Q = np.diag([10.0, 5.0, 10.0, 5.0, 0.2, 0.2])  # [10.0, 1.0, 10.0, 1.0, 0.2, 0.2]
        self.Q = np.diag([1.0, 0.1, 1.0, 0.1, 0.01, 0.01]) 
        self.R = np.diag([1e-2, 1e-2])  # [1e-4, 1e-4]
        # Hover reference action in normalized units.
        self.a_ref = np.array([0.5, 0.5], dtype=np.float32)

        self.state = np.zeros(6, dtype=np.float32)
        self._waypoints = self._create_waypoints()
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._waypoint_idx = 0

    # def _create_waypoints(self) -> np.ndarray:
    #     angles = np.deg2rad(np.arange(0, 360))
    #     xs = self.center[0] + self.radius * np.cos(angles)
    #     zs = self.center[1] + self.radius * np.sin(angles)
    #     waypoints = np.stack([xs, np.zeros_like(xs), zs, np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs)], axis=1)
    #     return waypoints.astype(np.float32)
    
    def _create_waypoints(self):
        traj_length = self.max_episode_steps * self.dt
        dt = self.dt
        num_cycles = 1

        traj_period = traj_length / num_cycles
        traj_freq = 2.0 * np.pi / traj_period

        times = np.arange(0, traj_length + dt, dt)

        pos = np.zeros((len(times), 2))
        vel = np.zeros((len(times), 2))

        for i, t in enumerate(times):
            x = self.radius * np.cos(traj_freq * t)
            z = self.radius * np.sin(traj_freq * t)

            vx = -self.radius * traj_freq * np.sin(traj_freq * t)
            vz =  self.radius * traj_freq * np.cos(traj_freq * t)

            pos[i] = [self.center[0] + x, self.center[1] + z]
            vel[i] = [vx, vz]

        waypoints = np.stack(
            [
                pos[:, 0],
                vel[:, 0],
                pos[:, 1],
                vel[:, 1],
                np.zeros(len(times)),
                np.zeros(len(times)),
            ],
            axis=1,
        )

        return waypoints.astype(np.float32)

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)

        low = np.array([-1.5, -1.0, 0.25, -1.5, -0.2, -0.1], dtype=np.float32)
        high = np.array([ 1.5,  1.0, 1.75,  1.5,  0.2,  0.1], dtype=np.float32)


        # 1) 先按原逻辑随机初始化
        state = self._rng.uniform(low, high).astype(np.float32)

        # 2) 如果用户传了 options，就覆盖对应维度
        # state 各维含义按你的 low/high 推断应为：
        # [x, x_dot, z, z_dot, theta, theta_dot]  (常见 quad2d)
        options = options or {}
        if "init_x" in options:
            state[0] = float(options["init_x"])
        if "init_vx" in options:
            state[1] = float(options["init_vx"])
        if "init_z" in options:
            state[2] = float(options["init_z"])
        if "init_vz" in options:
            state[3] = float(options["init_vz"])
        if "init_theta" in options:
            state[4] = float(options["init_theta"])
        if "init_omega" in options:
            state[5] = float(options["init_omega"])

        # 3) 写回 state
        self.state = state

        # 4) 重置计时与参考点
        self._t = 0

        # 可选：用户指定 waypoint idx（比如你想让参考轨迹起点也固定）
        if "init_waypoint_idx" in options:
            self._waypoint_idx = int(options["init_waypoint_idx"])
        else:
            self._waypoint_idx = self._nearest_waypoint_idx(self.state[0], self.state[2])

        observation = self._get_observation()
        info: Dict[str, Any] = {"idx": int(self._waypoint_idx)}
        return observation, info


    def _nearest_waypoint_idx(self, x: float, z: float) -> int:
        dx = x - self.center[0]
        dz = z - self.center[1]
        dist_sq = dx * dx + dz * dz
        if dist_sq < 1e-6:
            return int(self._rng.integers(0, len(self._waypoints)))
        deltas = self._waypoints[:, [0, 2]] - np.array([x, z], dtype=np.float32)
        idx = int(np.argmin(np.sum(np.square(deltas), axis=1)))
        return idx

    def _get_observation(self) -> np.ndarray:
        ref = self._waypoints[self._waypoint_idx]
        obs = np.concatenate([self.state, ref]).astype(np.float32)
        return obs

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.state = step_dynamics(self.state, action, self.dt, self.dyn_params)

        self._waypoint_idx = (self._waypoint_idx + 1) % len(self._waypoints)
        obs = self._get_observation()

        x_ref = self._waypoints[self._waypoint_idx]
        state_err = self.state - x_ref
        action_err = action - self.a_ref
        reward = -float(state_err @ self.Q @ state_err + action_err @ self.R @ action_err)

        x = float(self.state[0])
        z = float(self.state[2])
        h = max(0.5 - z, z - 1.5)
        cost = float(h > 0.0)

        terminated = bool(abs(x) > 2.0 or abs(z) > 3.0)
        self._t += 1
        truncated = bool(self._t >= self.max_episode_steps)

        info: Dict[str, Any] = {
            "h": float(h),
            "cost": float(cost),
            "z": float(z),
            "idx": int(self._waypoint_idx),
        }

        return obs, reward, terminated, truncated, info


# def make_quadrotor_tracking_2d_env(**kwargs) -> QuadrotorTracking2DEnv:
#     return QuadrotorTracking2DEnv(**kwargs)

def make_quadrotor_tracking_2d_env(**kwargs) -> gym.Env:
    env = QuadrotorTracking2DEnv(**kwargs)
    env = SymmetricActionWrapper(env)
    return env
