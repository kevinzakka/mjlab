"""Base configuration for the in-hand cube reorientation task."""

from mjlab.asset_zoo.props.qwerty_cube import CUBE_HALF_EXTENT
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
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


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
    "fingertip_to_cube": ObservationTermCfg(
      func=manipulation_mdp.fingertip_to_object,
      params={
        "object_name": "cube",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "fingertip_to_palm": ObservationTermCfg(
      func=manipulation_mdp.fingertip_to_palm,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {**actor_terms}

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=False),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": RelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.1,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "goal": ReorientationCommandCfg(
      entity_name="cube",
      robot_name="robot",
      success_threshold=0.2,
      success_hold_steps=5,
      # 50 control steps @ 50 Hz = 1.0 s, matching wuji's 20-step window @ 20 Hz.
      goal_switch_delay=50,
      resampling_time_range=(1.0e6, 1.0e6),
      debug_vis=True,
      viz=ReorientationCommandCfg.VizCfg(cube_half_extent=CUBE_HALF_EXTENT),
    )
  }

  events = {
    # Tilt the hand and nestle the cube in the (tilted) cradle with a random
    # SO(3) orientation. cradle_offset_b is filled in per-robot.
    "reset_hand_and_cube": EventTermCfg(
      func=manipulation_mdp.reset_hand_and_object,
      mode="reset",
      params={
        "hand_pitch_range": (-0.4, 0.1),
        "position_noise": 0.005,
        "cradle_offset_b": (0.0, 0.0, 0.0),  # Set per-robot.
        "object_cfg": SceneEntityCfg("cube"),
      },
    ),
    # Per-robot finger-joint resets are added in the robot config (the joint
    # groups are robot-specific).
  }

  rewards = {
    "orientation_tolerance": RewardTermCfg(
      func=manipulation_mdp.cube_orientation_tolerance,
      weight=5.0,
      params={"command_name": "goal"},
    ),
    "success_bonus": RewardTermCfg(
      func=manipulation_mdp.cube_orientation_success_bonus,
      weight=10.0,
      params={"command_name": "goal"},
    ),
    "drop_penalty": RewardTermCfg(func=mdp.is_terminated, weight=-50.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005),
    "joint_vel_hinge": RewardTermCfg(
      func=manipulation_mdp.joint_velocity_hinge_penalty,
      weight=-0.005,
      params={
        "max_vel": 2.0,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "cube_dropped": TerminationTermCfg(
      func=manipulation_mdp.object_below_height,
      params={"object_name": "cube", "minimum_height": 0.10},
    ),
    "cube_velocity": TerminationTermCfg(
      func=manipulation_mdp.object_velocity_out_of_bounds,
      params={"object_name": "cube", "max_lin_vel": 5.0, "max_ang_vel": 50.0},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=None,
      num_envs=1,
      env_spacing=1.0,
      extent=0.3,
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
        cone="pyramidal",
        impratio=1,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
