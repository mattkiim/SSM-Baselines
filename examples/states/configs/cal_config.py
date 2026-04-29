from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    config.model_cls = "CALAgent"

    config.hidden_dims = (256, 256)

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.cost_lr = 3e-4
    config.alpha_lr = 3e-4

    config.gamma = 0.99
    config.gamma_c = 0.99
    config.tau = 0.005

    config.k_ucb = 1.0
    config.alm_c = 10.0
    config.lambda_lr = 1e-4
    config.cost_limit = 10.0

    config.qc_ens_size = 4

    config.init_temperature = 1.0
    config.init_lambda = 1.0

    config.target_entropy = config_dict.placeholder(float)

    return config

