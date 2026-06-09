"""Tests for sharpa_constants.py."""

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


def test_fingertip_pad_friction(sharpa_model: mujoco.MjModel) -> None:
  """Fingertip elastomer pads are grippy (sliding friction 1.0); other hand geoms 0.5.

  Pad solref is stiffened to (0.012, 1.0) in sharpa_constants so the pads stop
  being the soft half of the finger-cube contact; the matching cube-side override
  lives in the reorient env cfg. The pads are condim 4 (soft-finger contact:
  sliding + torsional friction); the torsional coefficient has units of length
  (the contact-patch diameter). Both axes are randomized per-env in the env cfg.
  """
  for finger in ("thumb", "index", "middle", "ring", "pinky"):
    pad = sharpa_model.geom(f"right_{finger}_pad_collision")
    assert pad.condim == 4
    assert pad.friction[0] == 1.0  # sliding
    assert pad.friction[1] == 0.004  # torsional (patch diameter, metres)
    np.testing.assert_allclose(pad.solref, (0.012, 1.0))
  for name in ("palm000_collision", "right_index_PP_fit"):
    geom = sharpa_model.geom(name)
    assert geom.friction[0] == 0.5
    assert geom.condim == 3


def test_robot_compiles() -> None:
  assert isinstance(
    Entity(sharpa_constants.get_sharpa_right_cfg()).compile(), mujoco.MjModel
  )


# class -> (armature, frictionloss, damping, effort_limit).
_CLASS_MOTOR = {
  "CMC": (0.0032, 0.0, 0.0, 3.3),
  "PCMC": (0.00012, 0.0, 0.0, 0.5285),
  "MCP": (0.00265, 0.0, 0.0, 1.864),
  "PIP": (0.0006, 0.0, 0.0, 0.638),
  "DIP": (0.00042, 0.0, 0.0, 0.189369),
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
    arm, _, _, eff = _CLASS_MOTOR[cls]
    assert arm > 0 and eff > 0
