from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.reorient.rl import ManipulationOnPolicyRunner

from .env_cfgs import sharpa_reorient_cube_env_cfg
from .rl_cfg import sharpa_reorient_cube_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Reorient-Cube-Sharpa",
  env_cfg=sharpa_reorient_cube_env_cfg(),
  play_env_cfg=sharpa_reorient_cube_env_cfg(play=True),
  rl_cfg=sharpa_reorient_cube_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# Same task, but finger colliders use the link/elastomer meshes instead of the
# primitive fits. For benchmarking mesh-vs-primitive collision cost on GPU.
register_mjlab_task(
  task_id="Mjlab-Reorient-Cube-Sharpa-MeshCollision",
  env_cfg=sharpa_reorient_cube_env_cfg(use_mesh_collisions=True),
  play_env_cfg=sharpa_reorient_cube_env_cfg(play=True, use_mesh_collisions=True),
  rl_cfg=sharpa_reorient_cube_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
