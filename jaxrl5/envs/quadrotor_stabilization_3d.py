from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from jaxrl5.envs.dynamics.quad3d import step_dynamics_3d
from jaxrl5.wrappers.action_rescale import SymmetricActionWrapper


class QuadrotorStabilization3DEnv(gym.Env):
    """Dawson-style Quad3D stabilization task for RAC.

    State:
        [px, py, pz, vx, vy, vz, phi, theta, psi]

    Action:
        [f, phi_dot, theta_dot, psi_dot]

    Goal:
        Stabilize to the origin.

    Safety, matching the attached Quad3D system:
        safe if:
            pz <= 0.0 and ||state|| <= 3.0

        unsafe if:
            pz >= 0.3 or ||state|| >= 3.5

    Note:
        Dawson's file says z is positive downward. This env follows that convention.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 0.01,
        g: float = 9.81,
        m: float = 1.0,
        max_episode_steps: int = 500,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        max_thrust: float = 100.0,
        max_angle_rate: float = 50.0,
        goal_radius: float = 0.3,
        safe_z: float = 0.0,
        unsafe_z: float = 0.3,
        safe_radius: float = 3.0,
        unsafe_radius: float = 3.5,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.dt = dt
        self.g = g
        self.m = m
        self.max_episode_steps = max_episode_steps

        self.goal_radius = goal_radius
        self.safe_z = safe_z
        self.unsafe_z = unsafe_z
        self.safe_radius = safe_radius
        self.unsafe_radius = unsafe_radius

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )

        # self.action_space = gym.spaces.Box(
        #     low=np.array(
        #         [-max_thrust, -max_angle_rate, -max_angle_rate, -max_angle_rate],
        #         dtype=np.float32,
        #     ),
        #     high=np.array(
        #         [max_thrust, max_angle_rate, max_angle_rate, max_angle_rate],
        #         dtype=np.float32,
        #     ),
        #     dtype=np.float32,
        # )
        
        self.action_space = gym.spaces.Box(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

        # self.dyn_params: Dict[str, float] = {
        #     "m": self.m,
        #     "g": self.g,
        #     "action_low": self.action_space.low,
        #     "action_high": self.action_space.high,
        # }
        
        self.dyn_params = {
            "m": self.m,
            "g": self.g,
            "action_low": np.array([0.0, -5.0, -5.0, -5.0], dtype=np.float32),
            "action_high": np.array([2*self.m*self.g, 5.0, 5.0, 5.0], dtype=np.float32),
        }

        # Stabilization reward weights.
        self.Q = np.diag(
            [10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 0.02, 0.02]
        ).astype(np.float32)
        self.R = np.diag([1e-3, 1e-3, 1e-3, 1e-3]).astype(np.float32)

        self.goal = np.zeros(9, dtype=np.float32)

        # Dawson u_eq: [m*g, 0, 0, 0]
        self.a_ref = np.array([self.m * self.g, 0.0, 0.0, 0.0], dtype=np.float32)

        self.state = np.zeros(9, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._t = 0
        
    def _remap_action(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(action, -1.0, 1.0)
        u = np.zeros(4, dtype=np.float32)
        u[0] = self.m * self.g * (1.0 + a[0])  # [0, 2mg], hover at a=0
        u[1] = 5.0 * a[1]                       # [-5, 5] rad/s
        u[2] = 5.0 * a[2]
        u[3] = 5.0 * a[3]
        return u

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        if seed is not None:
            self.seed(seed)

        low = np.array(
            [-1.5, -1.5, -1.0, -0.5, -0.5, -0.5, -0.2, -0.2, -0.2],
            dtype=np.float32,
        )
        high = np.array(
            [1.5, 1.5, -0.05, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2],
            dtype=np.float32,
        )

        # Rejection sampling: discard unsafe or goal states
        state = None
        for _ in range(1000):
            candidate = self._rng.uniform(low, high).astype(np.float32)
            h = max(float(candidate[2]) - self.safe_z,
                    float(np.linalg.norm(candidate)) - self.safe_radius)
            if h < 0:  # safe and not in goal (goal radius 0.3 << safe radius 3.0)
                state = candidate
                break

        if state is None:
            raise RuntimeError("Failed to sample a safe initial state.")

        # Allow override via options (for eval)
        options = options or {}
        keys = [
            "init_px", "init_py", "init_pz",
            "init_vx", "init_vy", "init_vz",
            "init_phi", "init_theta", "init_psi",
        ]
        for i, key in enumerate(keys):
            if key in options:
                state[i] = float(options[key])

        self.state = state
        self._t = 0

        return self._get_observation(), self._get_info()

    def _get_observation(self) -> np.ndarray:
        return self.state.astype(np.float32)

    def _state_norm(self, state: np.ndarray) -> float:
        return float(np.linalg.norm(state))

    def _is_safe(self, state: np.ndarray) -> bool:
        return bool((state[2] <= self.safe_z) and (self._state_norm(state) <= self.safe_radius))

    def _is_unsafe(self, state: np.ndarray) -> bool:
        return bool((state[2] >= self.unsafe_z) or (self._state_norm(state) >= self.unsafe_radius))

    def _is_goal(self, state: np.ndarray) -> bool:
        return bool((self._state_norm(state) <= self.goal_radius) and self._is_safe(state))

    def _h(self, state: np.ndarray) -> float:
        # h <= 0 safe, h > 0 unsafe-ish.
        h_floor = float(state[2] - self.safe_z)
        h_radius = float(self._state_norm(state) - self.safe_radius)
        return max(h_floor, h_radius)

    def _get_info(self) -> Dict[str, Any]:
        h = self._h(self.state)
        unsafe = self._is_unsafe(self.state)
        goal = self._is_goal(self.state)
        return {
            "h": float(h),
            "cost": float(h > 0.0),
            "unsafe": bool(unsafe),
            "goal": bool(goal),
            "state_norm": float(self._state_norm(self.state)),
            "pz": float(self.state[2]),
        }

    def step(self, action: np.ndarray):
        # action = np.asarray(action, dtype=np.float32)
        # action = np.clip(action, self.action_space.low, self.action_space.high)

        # self.state = step_dynamics_3d(self.state, action, self.dt, self.dyn_params)

        # state_err = self.state - self.goal
        # action_err = action - self.a_ref
        # reward = -float(state_err @ self.Q @ state_err + action_err @ self.R @ action_err)
        
        #### 
        action = np.asarray(action, dtype=np.float32)
        u = self._remap_action(action)  # physical control
        
        self.state = step_dynamics_3d(self.state, u, self.dt, self.dyn_params)
        
        state_err = self.state - self.goal
        action_err = action  # penalize normalized action directly, a=0 is hover
        reward = -float(state_err @ self.Q @ state_err + action_err @ self.R @ action_err)
        #### 

        info = self._get_info()
        cost = float(info["cost"])

        self._t += 1

        # For RAC, you may not want to terminate on goal; keeping it as success info is cleaner.
        terminated = bool(info["unsafe"])
        truncated = bool(self._t >= self.max_episode_steps)

        return self._get_observation(), reward, terminated, truncated, info


def make_quadrotor_stabilization_3d_env(**kwargs) -> gym.Env:
    env = QuadrotorStabilization3DEnv(**kwargs)
    # env = SymmetricActionWrapper(env)
    return env