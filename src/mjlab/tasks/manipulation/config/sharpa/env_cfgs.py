from dataclasses import replace

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

# Palm-up base orientation: -90 deg about y rotates the hand's local +x (the grasp
# opening, where the tactile pads face) to world +z, so the cupped fingers form an
# upward-facing bowl that cradles the cube under gravity.
HAND_ROT = (0.70710678, 0.0, -0.70710678, 0.0)

# Cube cradle, expressed relative to the hand base: center of the cup, between the palm
# and the curled fingertips.
CRADLE_LOCAL = (-0.05, 0.0, 0.052)
CUBE_POS = (CRADLE_LOCAL[0], CRADLE_LOCAL[1], CRADLE_LOCAL[2] + HAND_POS[2])
# Goal ghost sits above the cradled cube (offset is relative to the hand base).
GHOST_OFFSET = (CRADLE_LOCAL[0], CRADLE_LOCAL[1], CRADLE_LOCAL[2] + 0.13)

# Cup grasp home pose (also the action offset). Fingers + thumb rest ON the cube faces
# in light contact (not curled inside it), so an axis-aligned cube nestles without
# penetrating. First-match-wins, so specific joints precede the general patterns.
HAND_HOME_JOINT_POS = {
  "right_thumb_CMC_AA": 0.25,
  "right_thumb_CMC_FE": -0.55,
  "right_thumb_IP": -0.55,
  ".*_MCP_FE": -0.41,
  ".*_PIP": -0.58,
  ".*_DIP": -0.44,
  ".*": 0.0,
}


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

  cfg.viewer.body_name = PALM_BODY

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
