from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def sharpa_reorient_cube_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "mjlab.rl.distributions:SoftplusGaussianDistribution",
        "init_std": 0.5,
        "min_std": 0.2,
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.015,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="sharpa_reorient_cube",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10_000,
    # SoftplusGaussian samples are unbounded; clamp to the [-1, 1] action set
    # the action term expects. Beta was bounded by construction; Gaussian is
    # not, so the wrapper has to clip.
    clip_actions=1.0,
  )
