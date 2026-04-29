from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.model_cls = "CALAgent"
    config.hidden_dims = (256, 256)
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.cost_lr = 5e-4
    config.alpha_lr = 3e-4
    config.gamma = 0.99
    config.gamma_c = 0.99
    config.tau = 0.005
    config.target_entropy = None
    config.k_ucb = 0.5
    config.alm_c = 10.0
    config.lambda_lr = 1e-3
    config.cost_limit = 25.0
    config.init_temperature = 1.0
    config.init_lambda = 1.0
    config.qc_ens_size = 6
    return config
