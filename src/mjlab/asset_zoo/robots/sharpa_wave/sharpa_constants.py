"""Sharpa Wave hand constants."""

from dataclasses import dataclass
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

SHARPA_RIGHT_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "sharpa_wave" / "xmls" / "right_hand.xml"
)
assert SHARPA_RIGHT_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(SHARPA_RIGHT_XML))


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
  condim={".*": 3},
  friction={
    ".*_pad_collision": (1.0,),
    ".*": (0.5,),
  },
  solref={
    ".*_pad_collision": (0.06, 0.9),
    ".*": (0.02, 1.0),
  },
)

##
# Initial state.
##

# Neutral, open hand. The task env cfg sets the palm-up base orientation and the
# grasp-ready home pose for cube reorientation.
HOME = EntityCfg.InitialStateCfg(
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

##
# Final config.
##

ARTICULATION = EntityArticulationInfoCfg(
  actuators=HAND_ACTUATORS,
  soft_joint_pos_limit_factor=1.0,
)


def get_sharpa_right_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME,
    spec_fn=get_spec,
    collisions=(SHARPA_COLLISION,),
    articulation=ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_sharpa_right_cfg())
  viewer.launch(robot.spec.compile())
