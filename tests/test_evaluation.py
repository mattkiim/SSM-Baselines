"""Evaluation stopping and aggregation checks without simulator dependencies."""

import unittest

from jaxrl5.evaluation import evaluate


class CountingEnv:
    def __init__(self, stop_at=None, truncate=False):
        self.stop_at = stop_at
        self.truncate = truncate
        self.resets = 0
        self.total_steps = 0
        self.seeds = []

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.resets += 1
        self.steps = 0
        return 0, {}

    def step(self, action):
        self.steps += 1
        self.total_steps += 1
        done = self.steps == self.stop_at
        return 0, 2.0, 1.0, done and not self.truncate, done and self.truncate, {}


class EvaluationTest(unittest.TestCase):
    def test_explicit_evaluation_seeds(self):
        env = CountingEnv(stop_at=1)
        evaluate(env, lambda obs: 0, episodes=3, seed=42)
        self.assertEqual(env.seeds, [42, 43, 44])

    def test_fifty_episodes_capped_at_one_thousand_steps(self):
        env = CountingEnv()
        metrics = evaluate(env, lambda obs: 0, episodes=50, max_episode_steps=1000)
        self.assertEqual(env.resets, 50)
        self.assertEqual(env.total_steps, 50000)
        self.assertEqual(metrics['eval/ep_len_mean'], 1000)
        self.assertEqual(metrics['eval/return_mean'], 2000)
        self.assertEqual(metrics['eval/cost_mean'], 1000)

    def test_environment_can_end_early(self):
        for truncate in (False, True):
            with self.subTest(truncate=truncate):
                env = CountingEnv(stop_at=3, truncate=truncate)
                metrics = evaluate(env, lambda obs: 0, episodes=2, max_episode_steps=1000)
                self.assertEqual(env.total_steps, 6)
                self.assertEqual(metrics['eval/ep_len_mean'], 3)

    def test_default_retains_environment_limit(self):
        env = CountingEnv(stop_at=1001)
        metrics = evaluate(env, lambda obs: 0, episodes=1)
        self.assertEqual(metrics['eval/ep_len_mean'], 1001)

    def test_invalid_limits(self):
        for kwargs in ({'episodes': 0}, {'max_episode_steps': 0}):
            with self.assertRaises(ValueError):
                evaluate(CountingEnv(), lambda obs: 0, **kwargs)


if __name__ == '__main__':
    unittest.main()
