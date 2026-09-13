"""Maximum-based reachability with exact velocity-transition margins."""

from examples.states.configs.rac_config import get_config as base_config


def get_config():
    config = base_config()
    config.safety_h_mode = 'reachability_transition'
    config.safety_threshold = 0.0
    config.lambda_max = 100.0
    return config
