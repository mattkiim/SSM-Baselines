import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.model_cls = "SACLagLearner"

    config.hidden_dims = (256, 256)

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.cost_critic_lr = 3e-4
    config.temp_lr = 3e-4

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

    config.env_name = "SafetyPointGoal1-v0"
    config.seed = 42
    config.max_steps = int(1e6)
    config.batch_size = 256
    config.start_training = int(1e4)
    config.eval_interval = 10_000
    config.eval_episodes = 5
    config.log_interval = 400
    config.utd_ratio = 1
    config.epoch_length = 400

    return config
