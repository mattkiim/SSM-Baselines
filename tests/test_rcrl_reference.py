"""Behavior checks for the optional reference protocol."""
import unittest
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from jaxrl5.agents.rac.rac_learner import RACLearner


class ReferenceTest(unittest.TestCase):
    def setUp(self):
        space = gym.spaces.Box(-1., 1., shape=(2,))
        self.agent = RACLearner.create(
            0, space, space, hidden_dims=(8, 8), reference_protocol=True,
            safety_h_mode='reachability_transition', safety_discount=1.,
            policy_update_period=4, multiplier_update_period=12, lambda_max=-1.)
        self.batch = dict(observations=jnp.zeros((4, 2)), actions=jnp.zeros((4, 2)),
                          next_observations=jnp.zeros((4, 2)), rewards=jnp.ones(4),
                          not_terminated=jnp.zeros(4), costs=jnp.zeros(4),
                          safety_h=jnp.array([-2., 0., 1., 2.]))

    def test_indicator_backup(self):
        _, metrics = self.agent.update_safety_critic(self.batch)
        self.assertEqual(float(metrics['target_qh_mean']), 5.)
        # At zero input, the initial safety prediction is zero: half MSE = 150.
        self.assertAlmostEqual(float(metrics['safety_critic_loss']), 150.)

    def test_schedule_targets_and_checkpoint(self):
        initial = self.agent
        a = initial
        for step in range(1, 13):
            previous = a
            a, metrics = a.update(self.batch)
            self.assertTrue(all(np.isfinite(float(v)) for v in metrics.values()))
            self.assertEqual(int(a.actor.step), step // 4)
            self.assertEqual(int(a.lambda_net.step), step // 12)
            self.assertEqual(int(a.temp.step), step // 4)
            if step % 4:
                for old, new in zip(jax.tree_util.tree_leaves(previous.target_critic.params),
                                    jax.tree_util.tree_leaves(a.target_critic.params)):
                    np.testing.assert_array_equal(old, new)
        restored = flax.serialization.from_bytes(initial, flax.serialization.to_bytes(a))
        self.assertEqual(int(restored.update_step), 12)
        continued, _ = restored.update(self.batch)
        self.assertEqual(int(continued.update_step), 13)

    def test_uncapped_multiplier_and_clipped_signal(self):
        params = jax.tree_util.tree_map(jnp.zeros_like, self.agent.lambda_net.params)
        last = list(params)[-1]
        params[last]['bias'] = jnp.array([200.])
        a = self.agent.replace(lambda_net=self.agent.lambda_net.replace(params=params))
        np.testing.assert_allclose(a._lambda_values(self.batch['observations']), 200.)
        # Two very large Qh values produce identical gradients after residual clipping.
        results = []
        for value in (200., 2000.):
            p = jax.tree_util.tree_map(jnp.zeros_like, a.safety_critic.params)
            p['Dense_0']['bias'] = jnp.array([value])
            updated, _ = a.replace(safety_critic=a.safety_critic.replace(params=p)).update_multiplier(self.batch)
            results.append(updated.lambda_net.params)
        for x, y in zip(*map(jax.tree_util.tree_leaves, results)):
            np.testing.assert_array_equal(x, y)


if __name__ == '__main__':
    unittest.main()
