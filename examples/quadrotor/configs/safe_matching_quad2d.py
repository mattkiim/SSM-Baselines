from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.model_cls = "SafeScoreMatchingLearner"
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.safety_lr = 3e-4
    config.actor_hidden_dims = (512, 512)
    config.critic_hidden_dims = (512, 512)
    config.safety_hidden_dims = (512, 512)
    config.discount = 0.99
    config.tau = 0.005
    config.ddpm_temperature = 0.2
    config.T = 5
    config.time_dim = 64
    config.clip_sampler = True
    config.beta_schedule = "vp"
    config.M_q = 10.0
    config.cost_limit = 1.0
    config.safety_discount = 0.99
    config.safety_lambda = 5.0
    config.alpha_coef = 0.25
    config.safety_threshold = 0.0
    config.safety_grad_scale = 10.0
    config.safe_lagrange_coef = 1.5
    config.lambda_lr = 3e-4
    config.lambda_hidden_dims = (256, 256)
    config.lambda_max = 100.0
    config.lambda_update_coef = 1.0
    config.actor_grad_coef = 1.0
    config.actor_safety_grad_coef = 1.0
    config.actor_grad_loss_coef = 0.5
    config.actor_aux_loss_coef = 5.0
    return config
