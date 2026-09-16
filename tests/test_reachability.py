import unittest

import jax.numpy as jnp
import numpy as np
import safety_gymnasium
import gymnasium as gym

from jaxrl5.algorithms.reachability import reachability_target
from jaxrl5.data.replay_buffer import ReplayBuffer
from jaxrl5.wrappers.velocity_constraint import VelocityConstraint
from jaxrl5.agents.rac.rac_learner import RACLearner


class ReachabilityTest(unittest.TestCase):
    def test_learner_uses_margin_not_binary_cost(self):
        space = gym.spaces.Box(-1., 1., shape=(2,))
        agent = RACLearner.create(0, space, space, hidden_dims=(8, 8),
                                 safety_h_mode='reachability_transition')
        batch = dict(observations=jnp.zeros((4, 2)), actions=jnp.zeros((4, 2)),
                     next_observations=jnp.zeros((4, 2)), not_terminated=jnp.zeros(4),
                     safety_h=jnp.array([-2., -1., 1., 2.]), costs=jnp.zeros(4))
        _, metrics = agent.update_safety_critic(batch)
        _, changed = agent.update_safety_critic(dict(batch, costs=jnp.ones(4)*100))
        self.assertEqual(float(metrics['target_qh_mean']), 0.)
        self.assertEqual(float(metrics['safety_critic_loss']), float(changed['safety_critic_loss']))

    def test_signed_maximum_backup_and_terminal_mask(self):
        h = jnp.array([-2., -2., 1., -2.])
        nxt = jnp.array([-3., 1., -3., 100.])
        result = reachability_target(h, nxt, jnp.array([1., 1., 1., 0.]), 0.9)
        np.testing.assert_allclose(result, [-2., 0.7, 1., -2.], atol=1e-6)

    def test_constant_constraint_is_fixed_point(self):
        for h in [-5., 0., 3.]:
            self.assertAlmostEqual(float(reachability_target(h, h, 1., 0.99)), h)

    def test_replay_keeps_transition_margin_aligned_after_wrap(self):
        replay = ReplayBuffer((1,), (1,), capacity=3, store_safety_h=True)
        for i in range(5):
            replay.insert([i], [i], 0., float(i > 2), [i+1], False, False, safety_h=i-2.)
        batch = replay.sample(32)
        np.testing.assert_equal(batch['safety_h'], batch['observations'][:, 0]-2.)
        with self.assertRaises(ValueError):
            replay.insert([0], [0], 0., 0., [0], False, False)

    def test_velocity_margin_matches_real_cost_without_changing_observations(self):
        for name in ['SafetyAntVelocity-v1', 'SafetyHumanoidVelocity-v1',
                     'SafetySwimmerVelocity-v1', 'SafetyHopperVelocity-v1',
                     'SafetyHalfCheetahVelocity-v1', 'SafetyWalker2dVelocity-v1']:
            with self.subTest(env=name):
                env = VelocityConstraint(safety_gymnasium.make(name))
                try:
                    obs, _ = env.reset(seed=0)
                    shape = obs.shape
                    seen = set()
                    for speed in [-10., 0., 10.]:
                        env.reset(seed=0)
                        env.unwrapped.data.qvel[0] = speed
                        obs, _, cost, _, _, info = env.step(np.zeros(env.action_space.shape))
                        seen.add(bool(cost))
                        self.assertEqual(obs.shape, shape)
                        self.assertEqual(info['safety_h'] > 0, cost > 0)
                        velocity = (np.hypot(info['x_velocity'], info['y_velocity'])
                                    if name in ('SafetyAntVelocity-v1', 'SafetyHumanoidVelocity-v1')
                                    else info['x_velocity'])
                        self.assertAlmostEqual(info['safety_h'], velocity-env.velocity_limit)
                    self.assertEqual(seen, {False, True})
                finally:
                    env.close()


if __name__ == '__main__':
    unittest.main()
