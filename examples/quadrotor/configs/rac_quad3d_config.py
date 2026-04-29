from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.model_cls = "RACLearner"

    # network
    config.hidden_dims = (256, 256)

    # learning rates
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.safety_lr = 3e-4
    config.lambda_lr = 3e-4
    config.alpha_lr = 3e-4

    # SAC / RAC discounts and target updates
    config.discount = 0.99
    config.tau = 0.005
    config.safety_discount = 0.99
    config.safety_tau = 0.005

    # if None, RACLearner will set this to -action_dim
    # for Quad3D action_dim = 4, so target_entropy becomes -4.0
    config.target_entropy = None

    # RAC-specific
    config.lambda_max = 100.0
    config.safety_threshold = 0.0
    config.policy_update_period = 1
    config.multiplier_update_period = 1
    config.init_temperature = 1.0

    return config