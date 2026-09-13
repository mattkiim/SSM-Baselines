import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.model_cls = "RACLearner"

    config.hidden_dims = (256, 256)

    # Multi-time-scale separation per Assumption 5.4 in the RAC paper (critic
    # fastest, actor intermediate, multiplier slowest). lambda_lr=3e-4 (flat
    # with everything else) made the multiplier oscillate between 0 and
    # lambda_max instead of converging (run 64m2016n). This slower rate kept
    # lambda near 0 and let reward grow substantially on the easy static
    # layout (seed 2664, run 8s74ykcs) -- reverted back to this after
    # lambda_lr=2e-5/period=3 looked worse on the harder layout (seed 2663,
    # run 3y9eng8a), though that comparison used a different seed too.
    config.actor_lr = 3e-5
    config.critic_lr = 3e-4
    config.safety_lr = 3e-4
    config.lambda_lr = 1e-6
    config.alpha_lr = 3e-5

    config.discount = 0.99
    config.tau = 0.005
    config.safety_discount = 0.99
    config.safety_tau = 0.005

    config.target_entropy = None
    config.safety_h_mode = "env_cost"
    config.num_qs = 2
    config.num_min_qs = None
    config.lambda_max = 100.0
    config.safety_threshold = 0.0
    config.policy_update_period = 1
    config.multiplier_update_period = 10
    config.init_temperature = 1.0

    config.env_name = "SafetyPointGoal1-v0"
    config.seed = 42
    config.max_steps = int(1e6)
    config.batch_size = 256
    config.start_training = int(1e4)
    config.eval_interval = 2000
    config.eval_episodes = 5
    config.utd_ratio = 1

    return config
