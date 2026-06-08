from dataclasses import replace

import mujoco

from mjlab.asset_zoo.props.qwerty_cube import (
  get_qwerty_cube_goal_marker_spec,
  get_qwerty_cube_spec,
)
from mjlab.asset_zoo.robots.sharpa_wave import get_sharpa_right_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.manipulation.mdp import ReorientationCommandCfg
from mjlab.tasks.manipulation.reorient_cube_env_cfg import (
  make_reorient_cube_env_cfg,
)

# Palm site used for cube-relative observations, the stay-near-palm reward, and the
# drop termination.
PALM_SITE = "wrist_site"

# Fingertip sites used for the fingertip-relative observations.
FINGERTIP_SITES = (
  "right_thumb_fingertip",
  "right_index_fingertip",
  "right_middle_fingertip",
  "right_ring_fingertip",
  "right_pinky_fingertip",
)

# Hand base body, used to center the viewer.
PALM_BODY = "right_hand_C_MC"

# Raise the (fixed-base) hand above the ground plane so no part of it dips below z=0.
HAND_POS = (0.0, 0.0, 0.09)

# Palm-up base orientation.
HAND_ROT = (0.70710678, 0.0, -0.70710678, 0.0)

# Cube cradle, expressed relative to the hand base: center of the cup, between the palm
# and the curled fingertips.
CRADLE_LOCAL = (-0.08, 0.0, 0.052)
CUBE_POS = (CRADLE_LOCAL[0], CRADLE_LOCAL[1], CRADLE_LOCAL[2] + HAND_POS[2])
# Goal ghost sits above the cradled cube (offset is relative to the hand base).
GHOST_OFFSET = (CRADLE_LOCAL[0], CRADLE_LOCAL[1], CRADLE_LOCAL[2] + 0.13)

HAND_HOME_JOINT_POS = {
  "right_thumb_CMC_AA": 0.25,
  "right_thumb_CMC_FE": -0.55,
  "right_thumb_IP": -0.55,
  ".*_MCP_FE": -0.41,
  ".*_PIP": -0.58,
  ".*_DIP": -0.44,
  ".*": 0.0,
}


def _box_from_range(
  center: tuple[float, float, float],
  pose_range: dict[str, tuple[float, float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
  """Translate a uniform (lo, hi) position range into a box center and half-size."""
  axes = ("x", "y", "z")
  box_center = list(center)
  half = []
  for i, axis in enumerate(axes):
    lo, hi = pose_range.get(axis, (0.0, 0.0))
    box_center[i] += 0.5 * (lo + hi)
    half.append(
      max(0.5 * (hi - lo), 1e-4)
    )  # Floor so a zero-width range still renders.
  return (box_center[0], box_center[1], box_center[2]), (half[0], half[1], half[2])


def _make_sampling_viz_spec_fn(
  center: tuple[float, float, float],
  pose_range: dict[str, tuple[float, float]],
):
  """Scene spec hook that draws the cube position-sampling region as a group-5 box.

  Group 5 is hidden by default, so this is a toggle-on debugging aid for sanity
  checking the reset distribution against the cradle geometry.
  """
  box_center, half = _box_from_range(center, pose_range)

  def spec_fn(spec: mujoco.MjSpec) -> None:
    spec.worldbody.add_site(
      name="cube_sampling_region",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=box_center,
      size=half,
      group=5,
      rgba=(0.2, 0.8, 0.2, 0.25),
    )

  return spec_fn


def sharpa_reorient_cube_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_reorient_cube_env_cfg()

  robot_cfg = get_sharpa_right_cfg()
  robot_cfg.init_state = replace(
    robot_cfg.init_state, pos=HAND_POS, rot=HAND_ROT, joint_pos=HAND_HOME_JOINT_POS
  )
  cube_cfg = EntityCfg(
    spec_fn=get_qwerty_cube_spec,
    init_state=EntityCfg.InitialStateCfg(pos=CUBE_POS),
  )
  goal_marker_cfg = EntityCfg(spec_fn=get_qwerty_cube_goal_marker_spec)
  cfg.scene.entities = {
    "robot": robot_cfg,
    "cube": cube_cfg,
    "goal_marker": goal_marker_cfg,
  }

  # Fill per-robot palm site into the terms that need it.
  cfg.observations["actor"].terms["cube_pos"].params["asset_cfg"].site_names = (
    PALM_SITE,
  )

  # Fill per-robot fingertip sites into the fingertip observations.
  cfg.observations["actor"].terms["fingertip_to_cube"].params[
    "asset_cfg"
  ].site_names = FINGERTIP_SITES
  cfg.observations["actor"].terms["fingertip_to_palm"].params[
    "asset_cfg"
  ].site_names = FINGERTIP_SITES

  # Pose the textured goal-marker cube just above the cradled cube.
  goal_cmd = cfg.commands["goal"]
  assert isinstance(goal_cmd, ReorientationCommandCfg)
  goal_cmd.marker_name = "goal_marker"
  goal_cmd.viz.offset = GHOST_OFFSET

  # Draw the cube reset distribution as a group-5 box (hidden until toggled on).
  cube_pose_range = cfg.events["reset_cube"].params["pose_range"]
  cfg.scene.spec_fn = _make_sampling_viz_spec_fn(CUBE_POS, cube_pose_range)

  cfg.viewer.body_name = PALM_BODY

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
