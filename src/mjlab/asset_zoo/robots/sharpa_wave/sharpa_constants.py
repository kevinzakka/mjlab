"""Sharpa Wave hand constants."""

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

_XML_DIR = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "sharpa_wave" / "xmls"

# Finger links collide via primitive fits (fast). The mesh variant swaps each fit
# for the link's collision mesh (tight to the visual, no spurious contacts) at the
# cost of mesh-mesh collision on GPU. Flip this to benchmark the two.
USE_MESH_COLLISIONS: bool = True

SHARPA_RIGHT_XML: Path = _XML_DIR / "right_hand.xml"
SHARPA_RIGHT_MESH_XML: Path = _XML_DIR / "right_hand_mesh_collision.xml"
assert SHARPA_RIGHT_XML.exists()
assert SHARPA_RIGHT_MESH_XML.exists()


def get_spec(use_mesh_collisions: bool = USE_MESH_COLLISIONS) -> mujoco.MjSpec:
  xml = SHARPA_RIGHT_MESH_XML if use_mesh_collisions else SHARPA_RIGHT_XML
  return mujoco.MjSpec.from_file(str(xml))


##
# Actuator config.
##


@dataclass(frozen=True)
class _JointClass:
  """Per joint-class passive/limit params, ported from the Menagerie joint classes."""

  armature: float
  frictionloss: float
  damping: float
  effort_limit: float


# _CMC = _JointClass(
#   armature=0.0032, frictionloss=0.132, damping=4.2e-05, effort_limit=3.3
# )
# _PCMC = _JointClass(
#   armature=0.00012, frictionloss=0.012, damping=4.2e-05, effort_limit=0.5285
# )
# _MCP = _JointClass(
#   armature=0.00265, frictionloss=0.07456, damping=2.38e-05, effort_limit=1.864
# )
# _PIP = _JointClass(
#   armature=0.0006, frictionloss=0.01276, damping=4.06e-06, effort_limit=0.638
# )
# _DIP = _JointClass(
#   armature=0.00042, frictionloss=0.00378738, damping=1.21e-06, effort_limit=0.189369
# )
_CMC = _JointClass(armature=0.0032, frictionloss=0.0, damping=0.0, effort_limit=3.3)
_PCMC = _JointClass(
  armature=0.00012, frictionloss=0.0, damping=0.0, effort_limit=0.5285
)
_MCP = _JointClass(armature=0.00265, frictionloss=0.0, damping=0.0, effort_limit=1.864)
_PIP = _JointClass(armature=0.0006, frictionloss=0.0, damping=0.0, effort_limit=0.638)
_DIP = _JointClass(
  armature=0.00042, frictionloss=0.0, damping=0.0, effort_limit=0.189369
)

# Per joint: (stiffness kp, damping kv, joint class).
_JOINTS: dict[str, tuple[float, float, _JointClass]] = {
  "right_thumb_CMC_FE": (6.95, 0.2844001777075252, _CMC),
  "right_thumb_CMC_AA": (13.2, 0.403408719431069, _CMC),
  "right_thumb_MCP_FE": (4.76, 0.20380858182545714, _MCP),
  "right_thumb_MCP_AA": (6.62, 0.2403441941312949, _MCP),
  "right_thumb_IP": (0.9, 0.04189004080632966, _PIP),
  "right_index_MCP_FE": (4.76, 0.2078418814338731, _MCP),
  "right_index_MCP_AA": (6.62, 0.24506922678230658, _MCP),
  "right_index_PIP": (0.9, 0.04231493938472393, _PIP),
  "right_index_DIP": (0.9, 0.03504057019251041, _DIP),
  "right_middle_MCP_FE": (4.76, 0.2078418814338731, _MCP),
  "right_middle_MCP_AA": (6.62, 0.24506922678230658, _MCP),
  "right_middle_PIP": (0.9, 0.04231493938472393, _PIP),
  "right_middle_DIP": (0.9, 0.03504057019251041, _DIP),
  "right_ring_MCP_FE": (4.76, 0.2078418814338731, _MCP),
  "right_ring_MCP_AA": (6.62, 0.24506922678230658, _MCP),
  "right_ring_PIP": (0.9, 0.04231493938472393, _PIP),
  "right_ring_DIP": (0.9, 0.03504057019251041, _DIP),
  "right_pinky_CMC": (1.38, 0.028719467883859495, _PCMC),
  "right_pinky_MCP_FE": (4.76, 0.2078418814338731, _MCP),
  "right_pinky_MCP_AA": (6.62, 0.24506922678230653, _MCP),
  "right_pinky_PIP": (0.9, 0.042314939384723936, _PIP),
  "right_pinky_DIP": (0.9, 0.03504057019251041, _DIP),
}

HAND_ACTUATORS = tuple(
  BuiltinPositionActuatorCfg(
    target_names_expr=(name,),
    stiffness=kp,
    damping=kv,
    effort_limit=jc.effort_limit,
    armature=jc.armature,
    frictionloss=jc.frictionloss,
    viscous_damping=jc.damping,
  )
  for name, (kp, kv, jc) in _JOINTS.items()
)

##
# Collision config.
##

SHARPA_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision", ".*_fit"),
  # Soft-finger contact for the fingertip pads: condim 4 adds torsional friction
  # (a twisting moment about the contact normal), which a condim-3 point contact
  # cannot provide. A real compliant pad presses into a finite patch that resists
  # the cube twisting in place; condim 4 is the standard soft-finger model and
  # stabilizes the grasp. The contact condim is the max of the two geoms, so this
  # makes every pad<->cube contact condim 4. Everything else stays condim 3.
  condim={".*_pad_collision": 4, ".*": 3},
  # Pad friction is (sliding, torsional). Torsional has units of length = the
  # contact-patch diameter; 0.004 m (~4 mm) suits the ~8 mm-radius pad capsules and
  # sits just below MuJoCo's 0.005 default. Both axes are overridden per-env by the
  # grip-friction DR at startup (see the reorient env cfg); these are the DR-off
  # nominals. Other colliders keep sliding-only friction (torsional inert at condim 3).
  friction={
    ".*_pad_collision": (1.0, 0.004),
    ".*": (0.5,),
  },
  # Stiffened fingertip pads. The pads are the soft half of the finger<->cube
  # contact (stiffness ~ 1/timeconst^2), so they dominate grip penetration. With
  # equal priority the contact uses the 0.5 blend of pad and cube params; the cube
  # is set to the SAME solref/solimp (see the cube CollisionCfg in the sharpa env
  # cfg) so the blend is deterministic. Tuned on the frozen ytflo6sj policy: this
  # cuts finger<->cube p95 penetration ~8x (5.8 -> 0.7 mm) with no NaNs and no speed
  # cost. timeconst 0.012 sits just above the 2*dt=0.01 floor (dt=0.005) for
  # headroom; solimp d1=0.99 makes the contact near-rigid at full penetration.
  solref={
    ".*_pad_collision": (0.012, 1.0),
    ".*": (0.02, 1.0),
  },
  solimp={
    ".*_pad_collision": (0.95, 0.99, 0.0005, 0.5, 2.0),
  },
)

##
# Initial state.
##

# Caged grasp home: cupped fingers that hold the cube off the palm for the in-hand
# reorientation task -- the hand's reset/reference pose. The task adds the palm-up
# base orientation. Run this module (``python sharpa_constants.py``) to visualize
# it. First-match-wins, so specific joints precede the general patterns.
CAGED_HOME = EntityCfg.InitialStateCfg(
  joint_pos={
    "right_thumb_CMC_AA": 0.25,
    "right_thumb_CMC_FE": -0.8,
    "right_thumb_IP": -0.8,
    ".*_MCP_FE": -0.66,
    ".*_PIP": -0.85,
    ".*_DIP": -0.7,
    ".*": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Final config.
##

ARTICULATION = EntityArticulationInfoCfg(
  actuators=HAND_ACTUATORS,
  soft_joint_pos_limit_factor=1.0,
)


def get_sharpa_right_cfg(
  use_mesh_collisions: bool = USE_MESH_COLLISIONS,
) -> EntityCfg:
  return EntityCfg(
    init_state=CAGED_HOME,
    spec_fn=partial(get_spec, use_mesh_collisions=use_mesh_collisions),
    collisions=(SHARPA_COLLISION,),
    articulation=ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_sharpa_right_cfg())
  model = robot.spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
  mujoco.mj_forward(model, data)
  viewer.launch(model, data)
