from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.model_cls = "RACLearner"

    # network
    # F16 obs is 26D (aggregate task feats) or 32D (per-axis task feats, default).
    # The default (256, 256) is fine but you may want (256, 256, 256) for the
    # higher-dim obs and richer dynamics.
    config.hidden_dims = (256, 256)

    # learning rates
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.safety_lr = 3e-4
    config.lambda_lr = 3e-4
    config.alpha_lr = 3e-4

    # SAC / RAC discounts and target updates
    # F16 uses dt=0.05 (vs Quad3D dt=0.01) and longer max_episode_steps=640
    # (vs 500), so effective horizons differ. discount=0.99 still gives an
    # effective horizon of ~100 steps which corresponds to ~5 seconds for F16
    # (vs 1 second for Quad3D). If you want a longer effective horizon,
    # bump discount to 0.995.
    config.discount = 0.99
    config.tau = 0.005
    config.safety_discount = 0.99
    config.safety_tau = 0.005

    # if None, RACLearner will set this to -action_dim
    # F16 action_dim = 4, so target_entropy becomes -4.0 (same as Quad3D).
    config.target_entropy = None

    # RAC-specific
    config.lambda_max = 100.0
    config.safety_threshold = 0.0
    config.safety_h_mode = "f16_task_feature"
    config.policy_update_period = 1
    config.multiplier_update_period = 1
    config.init_temperature = 1.0

    return config
