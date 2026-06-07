"""Base configuration for the in-hand cube reorientation task."""

from pathlib import Path

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
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

CUBE_HALF_EXTENT = 0.0225

_CUBE_TEXTURE_DIR = Path(__file__).parent / "assets" / "reorientation_cube_textures"

# MjSpec cube-texture face order: right, left, up, down, front, back.
_CUBE_FACES = ("right", "left", "up", "down", "front", "back")


def _make_textured_cube_spec(
  name: str,
  size: float,
  rgba: tuple[float, float, float, float],
  *,
  freejoint: bool,
  collide: bool,
  mass: float | None = None,
) -> mujoco.MjSpec:
  """Build a textured-cube MjSpec used by both the physical cube and goal marker."""
  spec = mujoco.MjSpec()

  spec.add_texture(
    name=name,
    type=mujoco.mjtTexture.mjTEXTURE_CUBE,
    cubefiles=[str(_CUBE_TEXTURE_DIR / f"file{face}.png") for face in _CUBE_FACES],
  )
  mat = spec.add_material(name=name, rgba=rgba)
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = name

  body = spec.worldbody.add_body(name=name)
  if freejoint:
    body.add_freejoint(name=f"{name}_joint")
  geom_kwargs: dict = dict(
    name=f"{name}_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(size, size, size),
    material=name,
  )
  if collide:
    assert mass is not None
    geom_kwargs["mass"] = mass
  else:
    geom_kwargs.update(contype=0, conaffinity=0, density=0.0, group=2)
  body.add_geom(**geom_kwargs)
  return spec


def get_reorient_cube_spec(
  cube_size: float = CUBE_HALF_EXTENT,
  mass: float = 0.15,
) -> mujoco.MjSpec:
  """Cube with a per-face texture so its orientation is readable in any viewer."""
  return _make_textured_cube_spec(
    "cube",
    cube_size,
    rgba=(1.0, 1.0, 1.0, 1.0),
    freejoint=True,
    collide=True,
    mass=mass,
  )


def get_goal_marker_spec(cube_size: float = CUBE_HALF_EXTENT) -> mujoco.MjSpec:
  """Visual-only translucent textured cube used as the goal marker.

  Fixed-base (mjlab wraps it as a mocap body); the reorientation command writes its
  pose each step to show the goal orientation above the hand.
  """
  return _make_textured_cube_spec(
    "goal",
    cube_size,
    rgba=(1.0, 1.0, 1.0, 0.35),
    freejoint=False,
    collide=False,
  )


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
      resampling_time_range=(1.0e6, 1.0e6),
      debug_vis=True,
      viz=ReorientationCommandCfg.VizCfg(cube_half_extent=CUBE_HALF_EXTENT),
    )
  }

  events = {
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
    "orientation_tracking_precise": RewardTermCfg(
      func=manipulation_mdp.cube_orientation_tracking,
      weight=1.0,
      params={"command_name": "goal", "std": 0.15},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.001),
    "joint_vel_hinge": RewardTermCfg(
      func=manipulation_mdp.joint_velocity_hinge_penalty,
      weight=-0.001,
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
