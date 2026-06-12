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

# Inverted (palm-down) variant: the hand is bolted upside down, so the cube hangs and
# must be actively gripped against gravity rather than resting in the cup. The full
# sim2real DR stack is kept.
register_mjlab_task(
  task_id="Mjlab-Reorient-Cube-Sharpa-Inverted",
  env_cfg=sharpa_reorient_cube_env_cfg(inverted=True),
  play_env_cfg=sharpa_reorient_cube_env_cfg(play=True, inverted=True),
  rl_cfg=sharpa_reorient_cube_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# "Easy" inverted variant: palm-down geometry with the sim2real DR stripped (no
# friction/inertia/size/encoder/impulse randomization, no mount tilt, clean obs) plus the
# gravity-magnitude curriculum -- gravity points fully down the whole time, but its
# strength ramps from near-weightless up to 9.81 over training, so the policy can learn
# the inverted grasp against progressively more weight rather than against full gravity
# from scratch. The clean, learnable inverted task; flip easy off to add the full DR stack
# back for transfer.
register_mjlab_task(
  task_id="Mjlab-Reorient-Cube-Sharpa-Inverted-Easy",
  env_cfg=sharpa_reorient_cube_env_cfg(
    inverted=True, easy=True, gravity_curriculum=True
  ),
  play_env_cfg=sharpa_reorient_cube_env_cfg(
    play=True, inverted=True, easy=True, gravity_curriculum=True
  ),
  rl_cfg=sharpa_reorient_cube_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
