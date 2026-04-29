from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.model_cls = "RACLearner"
    config.hidden_dims = (256, 256)
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.safety_lr = 3e-4
    config.lambda_lr = 3e-4
    config.alpha_lr = 3e-4
    config.discount = 0.99
    config.tau = 0.005
    config.safety_discount = 0.99
    config.safety_tau = 0.005
    config.target_entropy = None
    config.lambda_max = 100.0
    config.safety_threshold = 0.0
    config.policy_update_period = 1
    config.multiplier_update_period = 1
    config.init_temperature = 1.0
    return config
