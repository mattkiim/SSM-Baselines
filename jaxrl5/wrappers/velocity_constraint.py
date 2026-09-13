"""Exact signed transition constraint for Safety-Gymnasium velocity tasks."""

import gymnasium as gym
import numpy as np


class VelocityConstraint(gym.Wrapper):
    """Preserve observations/rewards/costs and add info['safety_h'].

    Ant uses torso displacement; Humanoid uses mass-center displacement.
    Both report the exact transition velocities used by their cost function.
    This is h(s, a), not instantaneous qvel reconstructed from observations.
    """

    def __init__(self, env):
        super().__init__(env)
        if env.spec.id not in ('SafetyAntVelocity-v1', 'SafetyHumanoidVelocity-v1'):
            raise ValueError('VelocityConstraint supports Ant/Humanoid velocity-v1 only')
        self.velocity_limit = float(env.unwrapped._velocity_threshold)

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        speed = np.hypot(info['x_velocity'], info['y_velocity'])
        h = float(speed - self.velocity_limit)
        if not np.isfinite(h) or bool(h > 0) != bool(cost > 0):
            raise ValueError('Velocity margin disagrees with the environment safety cost')
        info = dict(info, safety_h=h)
        return obs, reward, cost, terminated, truncated, info
