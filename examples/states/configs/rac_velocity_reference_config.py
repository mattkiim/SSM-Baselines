"""Reference quadrotor optimization protocol adapted to velocity transitions."""
from examples.states.configs.rac_velocity_reachability_config import get_config as base_config


def get_config():
    config = base_config()
    config.reference_protocol = True
    config.safety_discount = 1.0
    config.lambda_max = -1.0  # Sentinel: reference_protocol disables the cap.
    config.actor_lr = 2e-5
    config.critic_lr = 1e-4
    config.safety_lr = 1e-4
    config.lambda_lr = 6e-7
    config.alpha_lr = 8e-5
    config.policy_update_period = 4
    config.multiplier_update_period = 12
    # Map the reference's decay horizon to this experiment's update count.
    config.lr_decay_updates = 2990001
    return config
