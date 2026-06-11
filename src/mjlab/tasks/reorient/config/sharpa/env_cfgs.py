from dataclasses import replace

import mujoco
import numpy as np

from mjlab.asset_zoo.props.qwerty_cube import (
  get_qwerty_cube_goal_marker_spec,
  get_qwerty_cube_spec,
)
from mjlab.asset_zoo.robots.sharpa_wave import get_sharpa_right_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import reset_joints_by_offset
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.reorient.mdp import ReorientationCommandCfg
from mjlab.tasks.reorient.reorient_env_cfg import (
  make_reorient_cube_env_cfg,
)
from mjlab.utils.spec_config import CollisionCfg

# Palm/wrist site used for the cube-relative observations and as a cage point for the
# drop termination + escape penalty.
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
CRADLE_LOCAL = (-0.09, 0.0, 0.052)
CUBE_POS = (CRADLE_LOCAL[0], CRADLE_LOCAL[1], CRADLE_LOCAL[2] + HAND_POS[2])
# Goal ghost sits above the cradled cube (offset is relative to the hand base).
GHOST_OFFSET = (CRADLE_LOCAL[0], CRADLE_LOCAL[1], CRADLE_LOCAL[2] + 0.13)


# Cradle in the hand-base body frame, so the cube tracks the hand under reset pitch.
def _world_to_body(
  vec_w: tuple[float, ...], quat: tuple[float, ...]
) -> tuple[float, ...]:
  conj = np.zeros(4)
  mujoco.mju_negQuat(conj, np.array(quat))
  res = np.zeros(3)
  mujoco.mju_rotVecQuat(res, np.array(vec_w), conj)
  return tuple(res.tolist())


# Spawn the cube lifted along the palm normal so SO(3) corner-down orientations
# drop into the cup instead of penetrating it (corner reaches ~1.65 cm deeper).
CUBE_RESET_LIFT = 0.055
_cradle_b = _world_to_body(
  tuple(c - h for c, h in zip(CUBE_POS, HAND_POS, strict=True)), HAND_ROT
)
_up_b = _world_to_body((0.0, 0.0, 1.0), HAND_ROT)
CRADLE_OFFSET_B = tuple(
  o + CUBE_RESET_LIFT * u for o, u in zip(_cradle_b, _up_b, strict=True)
)

# Reset perturbation about the home grasp: a uniform offset added to the home pose
# and clipped to joint limits. Flexion (curl) gets more range than abduction
# (spread slides the fingertips off the cube faces).
FLEXION_JOINTS = (r".*_FE", r".*_IP", r".*_PIP", r".*_DIP", "right_pinky_CMC")
ABDUCTION_JOINTS = (r".*_AA",)
FLEXION_RESET_RANGE = (-0.1, 0.1)
ABDUCTION_RESET_RANGE = (-0.05, 0.05)


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
    # Contact-force arrow length is force * map.force / meanmass. This is a tiny,
    # light model (meanmass ~58 g), so the default map.force (0.005) renders grip
    # forces as arrows several times the cube size. Shrink the force->length gain
    # so the arrows read at hand scale (meansize handles their width).
    spec.visual.map.force = 0.001

  return spec_fn


def sharpa_reorient_cube_env_cfg(
  play: bool = False, use_mesh_collisions: bool = False
) -> ManagerBasedRlEnvCfg:
  cfg = make_reorient_cube_env_cfg()

  robot_cfg = get_sharpa_right_cfg(use_mesh_collisions=use_mesh_collisions)
  robot_cfg.init_state = replace(robot_cfg.init_state, pos=HAND_POS, rot=HAND_ROT)
  cube_cfg = EntityCfg(
    spec_fn=get_qwerty_cube_spec,
    init_state=EntityCfg.InitialStateCfg(pos=CUBE_POS),
    # Match the stiffened fingertip-pad solref/solimp (see SHARPA_COLLISION). With
    # equal priority the finger<->cube contact is the 0.5 blend of pad and cube
    # params; setting the cube identical to the pads makes that blend deterministic
    # and as stiff as intended. disable_other_geoms=False so only the cube geom is
    # touched (the cube spec has no other geoms, but this guards future ones).
    collisions=(
      CollisionCfg(
        geom_names_expr=("cube_geom",),
        # condim 4 + torsional friction to match the soft-finger pads (see
        # SHARPA_COLLISION). Friction combines by max at equal priority, so the
        # cube's torsional coefficient must also be set/randomized or it would floor
        # the grip's torsional friction. friction is (sliding, torsional); both axes
        # are overridden per-env by the grip-friction DR at startup.
        condim={"cube_geom": 4},
        friction={"cube_geom": (1.0, 0.004)},
        solref={"cube_geom": (0.012, 1.0)},
        solimp={"cube_geom": (0.95, 0.99, 0.0005, 0.5, 2.0)},
        disable_other_geoms=False,
      ),
    ),
  )
  goal_marker_cfg = EntityCfg(spec_fn=get_qwerty_cube_goal_marker_spec)
  cfg.scene.entities = {
    "robot": robot_cfg,
    "cube": cube_cfg,
    "goal_marker": goal_marker_cfg,
  }

  # Contact sensors for the grasp-quality rewards: per-fingertip pad-vs-cube and
  # palm-vs-cube contact.
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="tip_object_contact",
      primary=ContactMatch(mode="geom", pattern=(".*_pad_collision",), entity="robot"),
      secondary=ContactMatch(mode="body", pattern="cube", entity="cube"),
      fields=("found",),
    ),
    ContactSensorCfg(
      name="palm_object_contact",
      primary=ContactMatch(mode="body", pattern=PALM_BODY, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="cube", entity="cube"),
      fields=("found",),
    ),
    # Distal-finger grasp region: the last TWO phalanges per finger -- the
    # fingertip pad plus the next segment in (MP for the 4 fingers, PP for the
    # thumb, which has one fewer phalanx). Use this instead of the pad-only
    # `tip_object_contact` to let the cube nestle in the distal fingers rather
    # than being propped on the very fingertips.
    ContactSensorCfg(
      name="distal_finger_object",
      primary=ContactMatch(
        mode="geom",
        pattern=(".*_pad_collision", ".*_MP_fit", ".*thumb_PP_fit"),
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="cube", entity="cube"),
      fields=("found",),
    ),
    # Intra-hand self-contact (hand subtree vs itself; adjacent links are excluded
    # so this is the non-adjacent finger<->finger / finger<->palm contact). Force
    # history captures the per-substep peak. Read by the self_contact_force cost to
    # encourage a gentle grasp.
    ContactSensorCfg(
      name="finger_self_contact",
      primary=ContactMatch(mode="subtree", pattern=PALM_BODY, entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern=PALM_BODY, entity="robot"),
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
      history_length=4,
    ),
  )

  # Fill per-robot palm site into the terms that need it.
  cfg.observations["actor"].terms["cube_pos"].params["asset_cfg"].site_names = (
    PALM_SITE,
  )
  # Cage (drop termination + escape penalty + viz): palm body is the frame,
  # fingertips + wrist are the cage points, and the hand-base +x (palm normal) is
  # the open up-axis the cube can lift along without penalty.
  for cage_params in (
    cfg.terminations["cube_dropped"].params,
    cfg.rewards["cage_escape"].params,
  ):
    cage_params["asset_cfg"].body_names = (PALM_BODY,)
    cage_params["asset_cfg"].site_names = (*FINGERTIP_SITES, PALM_SITE)
    cage_params["up_axis"] = 0
    cage_params["up_margin"] = 0.04

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

  # Place the cube in the hand cradle (tracks the hand under reset pitch).
  reset_event = cfg.events["reset_hand_and_cube"]
  reset_event.params["cradle_offset_b"] = CRADLE_OFFSET_B

  # Fill the per-robot fingertip-pad geoms for grip-friction DR (sliding + torsional).
  for term in ("pad_friction_slide", "pad_friction_spin"):
    cfg.events[term].params["asset_cfg"].geom_names = (".*_pad_collision",)

  # Perturb the home grasp at reset: flexion and spread get separate ranges.
  for name, joints, joint_range in (
    ("reset_finger_flexion", FLEXION_JOINTS, FLEXION_RESET_RANGE),
    ("reset_finger_abduction", ABDUCTION_JOINTS, ABDUCTION_RESET_RANGE),
  ):
    cfg.events[name] = EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": joint_range,
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=joints),
      },
    )

  # Draw the cube position-noise region as a group-5 box (hidden until toggled on).
  noise = reset_event.params["position_noise"]
  cube_pose_range = {ax: (-noise, noise) for ax in ("x", "y", "z")}
  cfg.scene.spec_fn = _make_sampling_viz_spec_fn(CUBE_POS, cube_pose_range)

  cfg.viewer.body_name = PALM_BODY

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
