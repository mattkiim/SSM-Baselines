from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.model_cls = "SACCbfLearner"
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
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
    config.cbf_enabled = True
    config.cbf_mu = 0.2
    config.cbf_dt = 1.0 / 60.0
    config.cbf_fd_eps = 1e-3
    config.cbf_max_iters = 8
    config.cbf_grad_eps = 1e-8
    config.cbf_shrink_factor = 0.5
    config.z_min = 0.5
    config.z_max = 1.5
    config.z_index = 2
    config.dyn_m = 1.0
    config.dyn_I = 0.02
    config.dyn_g = 9.81
    config.thrust_scale = None
    config.torque_scale = 0.1
    config.action_low = 0.0
    config.action_high = 1.0
    return config
