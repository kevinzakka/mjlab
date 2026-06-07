from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import sharpa_reorient_cube_env_cfg
from .rl_cfg import sharpa_reorient_cube_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Reorient-Cube-Sharpa",
  env_cfg=sharpa_reorient_cube_env_cfg(),
  play_env_cfg=sharpa_reorient_cube_env_cfg(play=True),
  rl_cfg=sharpa_reorient_cube_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
