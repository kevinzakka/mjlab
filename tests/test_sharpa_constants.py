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


def test_fingertip_pad_friction(sharpa_model: mujoco.MjModel) -> None:
  """Fingertip elastomer pads are grippy (sliding friction 1.0); other hand geoms 0.5."""
  for finger in ("thumb", "index", "middle", "ring", "pinky"):
    pad = sharpa_model.geom(f"right_{finger}_pad_collision")
    assert pad.friction[0] == 1.0
    assert pad.condim == 3
    # Intrinsic elastomer compliance stays in the XML.
    np.testing.assert_allclose(pad.solref, (0.06, 0.9))
  for name in ("palm000_collision", "right_index_PP_fit"):
    geom = sharpa_model.geom(name)
    assert geom.friction[0] == 0.5
    assert geom.condim == 3


def test_robot_compiles() -> None:
  assert isinstance(
    Entity(sharpa_constants.get_sharpa_right_cfg()).compile(), mujoco.MjModel
  )
