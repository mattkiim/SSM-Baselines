from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.model_cls = "SACLagLearner"
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.cost_critic_lr = 3e-4
    config.temp_lr = 3e-4
    config.hidden_dims = (256, 256)
    config.discount = 0.99
    config.tau = 0.005
    config.num_qs = 2
    config.num_min_qs = None
    config.critic_dropout_rate = None
    config.critic_layer_norm = False
    config.target_entropy = None
    config.init_temperature = 1.0
    config.backup_entropy = True
    config.lambda_init = 0.0
    config.lambda_lr = 1e-3
    config.lambda_max = 1000.0
    config.cost_limit = 0.0
    return config
