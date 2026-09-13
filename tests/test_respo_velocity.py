"""Check that the RESPO bridge preserves the benchmark dynamics and costs."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import safety_gymnasium

path = Path(__file__).resolve().parents[1] / 'respo-clone/velocity_env.py'
module = None
if path.exists():
    spec = importlib.util.spec_from_file_location('respo_velocity_env', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


@unittest.skipUnless(path.exists(), 'Optional RESPO checkout missing; see docs/portable-setup.md')
class VelocityBridgeTest(unittest.TestCase):
    def test_same_resets_actions_rewards_and_costs(self):
        for name in ['SafetyAntVelocity-v1', 'SafetyHumanoidVelocity-v1']:
            with self.subTest(env=name):
                bridge = module.VelocityEnv(name, seed=42)
                reference = safety_gymnasium.make(name)
                try:
                    obs, _ = reference.reset(seed=42)
                    np.testing.assert_array_equal(bridge.reset(), obs)
                    for magnitude in [0., 0.3, 2.]:
                        action = np.full(bridge.action_space.shape, magnitude)
                        actual = bridge.step(action)
                        normalized = np.clip(action.astype(np.float32), -1., 1.)
                        low = reference.action_space.low.astype(np.float32)
                        high = reference.action_space.high.astype(np.float32)
                        expected = reference.step(low + (normalized+1.)*0.5*(high-low))
                        np.testing.assert_array_equal(actual[0], expected[0])
                        self.assertEqual(actual[1], expected[1])
                        self.assertEqual(actual[2], expected[3] or expected[4])
                        self.assertEqual(actual[3]['cost'], expected[2])
                finally:
                    bridge.close()
                    reference.close()


if __name__ == '__main__':
    unittest.main()
