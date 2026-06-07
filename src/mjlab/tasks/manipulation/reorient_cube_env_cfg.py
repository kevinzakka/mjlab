"""Base configuration for the in-hand cube reorientation task.

This wires the MDP (observations, rewards, terminations, events, command, sim) with
per-robot placeholders (site names, action scale, entities) filled in by the robot
config under ``config/<robot>/``. No domain randomization is included; it will be added
once the task trains reliably.
"""

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.mdp import ReorientationCommandCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

# 50 mm cube. The wrist is fixed (no DOFs), so manipulation is fingertip-driven: the
# cube must ride up in the fingers where the pads can purchase its faces rather than
# sink into the palm. 50 mm sits in the finger cage for this hand without being as
# oversized as a palm-filling cube.
CUBE_HALF_EXTENT = 0.025


def get_reorient_cube_spec(
  cube_size: float = CUBE_HALF_EXTENT,
  mass: float = 0.15,
  rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size,) * 3,
    mass=mass,
    rgba=rgba,
  )
  return spec


def make_reorient_cube_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base in-hand cube reorientation task configuration."""

  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "cube_pos": ObservationTermCfg(
      func=manipulation_mdp.ee_to_object_distance,
      params={
        "object_name": "cube",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "cube_ori": ObservationTermCfg(
      func=manipulation_mdp.object_orientation_6d,
      params={"object_name": "cube"},
    ),
    "cube_to_goal_ori": ObservationTermCfg(
      func=manipulation_mdp.object_to_goal_orientation_6d,
      params={"object_name": "cube", "command_name": "goal"},
    ),
    "cube_lin_vel": ObservationTermCfg(
      func=manipulation_mdp.object_lin_vel_b,
      params={"object_name": "cube"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "cube_ang_vel": ObservationTermCfg(
      func=manipulation_mdp.object_ang_vel_b,
      params={"object_name": "cube"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {**actor_terms}

  observations = {
    # Corruption is OFF while debugging the MDP; per-term noise is kept configured so
    # it can be re-enabled with a single flag once the task trains.
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=False),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }

  # Relative (delta) joint targets: target = current_joint_pos + scale * action. Zero
  # action holds the current pose (good for holding the cube), and the policy can drive
  # joints across their full range over multiple steps rather than being pinned within
  # +-scale of a home offset. The grasp home pose is set by the entity's init_state.
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": RelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.1,  # Per-step delta (rad); ~5 rad/s peak at 50 Hz.
      clip={".*": (-0.15, 0.15)},  # Bound per-step delta against policy outliers.
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "goal": ReorientationCommandCfg(
      entity_name="cube",
      robot_name="robot",
      success_threshold=0.1,
      # Goals change on reset and on success; the timer is an effectively-never fallback.
      resampling_time_range=(1.0e6, 1.0e6),
      debug_vis=True,
      viz=ReorientationCommandCfg.VizCfg(cube_half_extent=CUBE_HALF_EXTENT),
    )
  }

  events = {
    # Position the (fixed-base) hand at the env origin with its configured orientation.
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {}, "velocity_range": {}},
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Place the cube in the hand in a near-canonical (axis-aligned) orientation that the
    # grip fits without penetration. Only a few degrees of jitter; the goal command
    # samples full SO(3), so the policy still learns all rotations.
    "reset_cube": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.005, 0.005),
          "y": (-0.005, 0.005),
          "z": (-0.005, 0.005),
          "roll": (-0.1, 0.1),
          "pitch": (-0.1, 0.1),
          "yaw": (-0.1, 0.1),
        },
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("cube"),
      },
    ),
  }

  rewards = {
    "orientation_tracking": RewardTermCfg(
      func=manipulation_mdp.cube_orientation_tracking,
      weight=1.0,
      params={"command_name": "goal", "std": 1.0},
    ),
    "success_bonus": RewardTermCfg(
      func=manipulation_mdp.cube_orientation_success_bonus,
      weight=5.0,
      params={"command_name": "goal"},
    ),
    "stay_near_palm": RewardTermCfg(
      func=manipulation_mdp.cube_stay_near_palm,
      weight=0.5,
      params={
        "object_name": "cube",
        "std": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "joint_vel_hinge": RewardTermCfg(
      func=manipulation_mdp.joint_velocity_hinge_penalty,
      weight=-0.01,
      params={
        "max_vel": 2.0,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "cube_dropped": TerminationTermCfg(
      func=manipulation_mdp.object_dropped,
      params={
        "object_name": "cube",
        "max_distance": 0.15,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "cube_velocity": TerminationTermCfg(
      func=manipulation_mdp.object_velocity_out_of_bounds,
      params={"object_name": "cube", "max_lin_vel": 5.0, "max_ang_vel": 50.0},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      env_spacing=1.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=0.6,
      elevation=-20.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=150,
      njmax=800,
      mujoco=MujocoCfg(
        timestep=0.005,
        integrator="implicitfast",
        cone="elliptic",
        impratio=1,
        solver="newton",
        iterations=50,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
