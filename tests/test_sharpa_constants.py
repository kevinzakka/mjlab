"""Tests for sharpa_constants.py.

These lock the kinematic-only vendoring + config port: the compiled model must carry
the actuator gains, armature, frictionloss, joint damping, effort limits, and fingertip
friction that the config (not the XML) now owns.
"""

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.sharpa_wave import sharpa_constants
from mjlab.entity import Entity


@pytest.fixture(scope="module")
def sharpa_entity() -> Entity:
  return Entity(sharpa_constants.get_sharpa_right_cfg())


@pytest.fixture(scope="module")
def sharpa_model(sharpa_entity: Entity) -> mujoco.MjModel:
  return sharpa_entity.spec.compile()


def test_vendored_xml_is_kinematic_only() -> None:
  """The vendored XML carries no actuators and no simulator options (config owns them)."""
  m = mujoco.MjModel.from_xml_path(str(sharpa_constants.SHARPA_RIGHT_XML))
  assert m.nu == 0
  assert m.dof_armature.sum() == 0.0
  assert m.dof_frictionloss.sum() == 0.0
  assert m.dof_damping.sum() == 0.0
  # Untouched <option> resolves to MuJoCo defaults.
  assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
  assert m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
  assert m.opt.impratio == 1.0


def test_entity_creation(sharpa_entity: Entity) -> None:
  assert sharpa_entity.num_actuators == 22
  assert sharpa_entity.num_joints == 22
  assert sharpa_entity.is_actuated
  assert sharpa_entity.is_fixed_base


def test_actuator_gains(sharpa_model: mujoco.MjModel) -> None:
  """Each joint's position actuator reproduces the Menagerie kp/kv and effort limit."""
  for name, (kp, kv, jc) in sharpa_constants._JOINTS.items():
    actuator = sharpa_model.actuator(name)
    np.testing.assert_allclose(actuator.gainprm[0], kp, rtol=1e-12)
    np.testing.assert_allclose(actuator.biasprm[1], -kp, rtol=1e-12)
    np.testing.assert_allclose(actuator.biasprm[2], -kv, rtol=1e-12)
    np.testing.assert_allclose(actuator.forcerange[0], -jc.effort_limit, rtol=1e-12)
    np.testing.assert_allclose(actuator.forcerange[1], jc.effort_limit, rtol=1e-12)


def test_joint_dynamics(sharpa_model: mujoco.MjModel) -> None:
  """Armature, frictionloss, and passive damping match the ported joint-class values."""
  for name, (_, _, jc) in sharpa_constants._JOINTS.items():
    joint = sharpa_model.joint(name)
    dof = joint.dofadr[0]
    np.testing.assert_allclose(sharpa_model.dof_armature[dof], jc.armature, rtol=1e-12)
    np.testing.assert_allclose(
      sharpa_model.dof_frictionloss[dof], jc.frictionloss, rtol=1e-12
    )
    np.testing.assert_allclose(sharpa_model.dof_damping[dof], jc.damping, rtol=1e-12)


def test_actuators_position_control_limits(sharpa_model: mujoco.MjModel) -> None:
  """Position actuators allow setpoints beyond joint limits but clamp force."""
  for i in range(sharpa_model.nu):
    assert sharpa_model.actuator_ctrllimited[i] == 0
    assert sharpa_model.actuator_forcelimited[i] == 1


def test_hand_friction(sharpa_model: mujoco.MjModel) -> None:
  """All hand geoms share a uniform sliding friction of 0.7. Fingertip pads
  retain their softer solref for elastomer compliance."""
  for finger in ("thumb", "index", "middle", "ring", "pinky"):
    pad = sharpa_model.geom(f"right_{finger}_pad_collision")
    assert pad.friction[0] == 0.7
    assert pad.condim == 3
    # Intrinsic elastomer compliance stays in the XML.
    np.testing.assert_allclose(pad.solref, (0.06, 0.9))
  for name in ("palm000_collision", "right_index_PP_fit"):
    geom = sharpa_model.geom(name)
    assert geom.friction[0] == 0.7
    assert geom.condim == 3


def test_robot_compiles() -> None:
  assert isinstance(
    Entity(sharpa_constants.get_sharpa_right_cfg()).compile(), mujoco.MjModel
  )


# --- Motor / joint-class parameter checks -----------------------------------------
#
# The Sharpa hand has five joint classes, each corresponding to one motor type. These
# canonical values are the ground truth from the Menagerie model (README: "armature,
# damping, frictionloss, and actuatorfrcrange values specific to each joint class,
# matched to the Sharpa controller"). They are written here INDEPENDENTLY of
# sharpa_constants so a typo or mis-assignment in the config is caught.

# class -> (armature, frictionloss, damping, effort_limit).
_CLASS_MOTOR = {
  "CMC": (0.0032, 0.132, 4.2e-05, 3.3),
  "PCMC": (0.00012, 0.012, 4.2e-05, 0.5285),
  "MCP": (0.00265, 0.07456, 2.38e-05, 1.864),
  "PIP": (0.0006, 0.01276, 4.06e-06, 0.638),
  "DIP": (0.00042, 0.00378738, 1.21e-06, 0.189369),
}


def _expected_class(joint_name: str) -> str:
  """Anatomical joint -> motor class, derived from the name (not from the config)."""
  if joint_name in ("right_thumb_CMC_FE", "right_thumb_CMC_AA"):
    return "CMC"
  if joint_name == "right_pinky_CMC":
    return "PCMC"
  if joint_name.endswith(("_MCP_FE", "_MCP_AA")):
    return "MCP"
  if joint_name.endswith("_PIP") or joint_name == "right_thumb_IP":
    return "PIP"
  if joint_name.endswith("_DIP"):
    return "DIP"
  raise AssertionError(f"Unmapped joint: {joint_name}")


def test_every_joint_matches_its_motor_class(sharpa_model: mujoco.MjModel) -> None:
  """armature / frictionloss / joint-damping / forcerange match the joint's motor."""
  seen_classes = set()
  for j in range(sharpa_model.njnt):
    name = sharpa_model.joint(j).name
    cls = _expected_class(name)
    seen_classes.add(cls)
    armature, frictionloss, damping, effort = _CLASS_MOTOR[cls]
    dof = sharpa_model.joint(name).dofadr[0]
    np.testing.assert_allclose(sharpa_model.dof_armature[dof], armature, rtol=1e-9)
    np.testing.assert_allclose(
      sharpa_model.dof_frictionloss[dof], frictionloss, rtol=1e-9
    )
    np.testing.assert_allclose(sharpa_model.dof_damping[dof], damping, rtol=1e-9)
    actuator = sharpa_model.actuator(name)
    np.testing.assert_allclose(actuator.forcerange, (-effort, effort), rtol=1e-9)
    assert sharpa_model.actuator_forcelimited[actuator.id] == 1
  # All five motor classes are actually present.
  assert seen_classes == set(_CLASS_MOTOR)


def test_motor_sizes_decrease_distally() -> None:
  """Proximal motors are stronger/heavier than distal ones (CMC > MCP > PIP > DIP)."""
  order = ["CMC", "MCP", "PIP", "DIP"]
  armatures = [_CLASS_MOTOR[c][0] for c in order]
  efforts = [_CLASS_MOTOR[c][3] for c in order]
  assert armatures == sorted(armatures, reverse=True)
  assert efforts == sorted(efforts, reverse=True)
  for cls in _CLASS_MOTOR:
    arm, fric, damp, eff = _CLASS_MOTOR[cls]
    assert arm > 0 and fric > 0 and damp > 0 and eff > 0


def test_actuators_numerically_stable_at_sim_timestep(
  sharpa_model: mujoco.MjModel,
) -> None:
  """Each position actuator is integrable at the task timestep.

  The actuator natural frequency from its own reflected inertia is
  omega = sqrt(kp / armature). For explicit integration of the stiffness term the
  step must satisfy omega * dt << 2. We require a comfortable margin (omega * dt < 1)
  at the task's 0.005 s timestep, so the joints are NOT the source of blow-ups. Note
  this is conservative: the joint's link inertia adds to armature, lowering omega
  further.
  """
  timestep = 0.005  # SimulationCfg.mujoco.timestep for the reorient task.
  worst = 0.0
  for i in range(sharpa_model.nu):
    kp = sharpa_model.actuator_gainprm[i, 0]
    dof = sharpa_model.joint(sharpa_model.actuator(i).name).dofadr[0]
    armature = sharpa_model.dof_armature[dof]
    omega = (kp / armature) ** 0.5
    worst = max(worst, omega * timestep)
  assert worst < 1.0, f"stiffest actuator has omega*dt={worst:.3f} (>=1)"
