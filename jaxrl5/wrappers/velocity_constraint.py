"""Exact signed transition constraint for Safety-Gymnasium velocity tasks."""

import gymnasium as gym
import numpy as np


class VelocityConstraint(gym.Wrapper):
    """Preserve observations/rewards/costs and add info['safety_h'].

    Ant uses torso displacement; Humanoid uses mass-center displacement.
    Swimmer, Hopper, HalfCheetah and Walker2d constrain signed forward velocity. Each task reports
    the exact transition velocity used by its cost function.
    This is h(s, a), not instantaneous qvel reconstructed from observations.
    """

    def __init__(self, env):
        super().__init__(env)
        self.planar_speed = env.spec.id in ('SafetyAntVelocity-v1', 'SafetyHumanoidVelocity-v1')
        if env.spec.id not in ('SafetyAntVelocity-v1', 'SafetyHumanoidVelocity-v1',
                               'SafetySwimmerVelocity-v1', 'SafetyHopperVelocity-v1',
                               'SafetyHalfCheetahVelocity-v1', 'SafetyWalker2dVelocity-v1'):
            raise ValueError('VelocityConstraint supports Ant/Humanoid/Swimmer/Hopper/HalfCheetah/Walker2d velocity-v1 only')
        self.velocity_limit = float(env.unwrapped._velocity_threshold)

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        speed = (np.hypot(info['x_velocity'], info['y_velocity'])
                 if self.planar_speed else info['x_velocity'])
        h = float(speed - self.velocity_limit)
        if not np.isfinite(h) or bool(h > 0) != bool(cost > 0):
            raise ValueError('Velocity margin disagrees with the environment safety cost')
        info = dict(info, safety_h=h)
        return obs, reward, cost, terminated, truncated, info
