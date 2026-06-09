from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.reorient.mdp.commands import ReorientationCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _alive_gate(
  env: ManagerBasedRlEnv,
  object_name: str | None,
  min_height: float,
) -> torch.Tensor | None:
  """Return a per-env multiplicative ``cube_held`` mask in {0., 1.}, or None.

  When ``object_name`` is provided, returns 1.0 where the object's root-link
  z-position is above ``min_height`` and 0.0 elsewhere. This is used to
  multiplicatively gate positive task rewards so they zero out when the cube
  is lost, which makes early termination implicitly costly (the policy loses
  all future positive reward) without introducing a negative drop_penalty
  cliff in the value function. See the "principled formulation" discussion:
  multiplicative gating preserves the always-non-negative reward structure
  and avoids suicide-incentive local minima.
  """
  if object_name is None:
    return None
  obj: Entity = env.scene[object_name]
  return (obj.data.root_link_pos_w[:, 2] > min_height).float()


def cube_orientation_tolerance(
  env: ManagerBasedRlEnv,
  command_name: str,
  bound: float = 0.1,
  margin: float = 3.141592653589793,
  value_at_margin: float = 0.1,
  gate_object_name: str | None = None,
  gate_min_height: float = 0.1,
) -> torch.Tensor:
  """Tolerance kernel: 1.0 inside ``[0, bound]``, linear decay outside.

  Reward is exactly 1.0 for any orientation error in the "good enough" band
  ``[0, bound]``, so the policy isn't pulled to chase err -> 0 once it is
  comfortably close. Outside the bound, reward decays linearly to
  ``value_at_margin`` at error = ``bound + margin``, then is clamped at that
  floor. Matches the dm_control / mujoco_playground ``tolerance`` shape with
  ``sigmoid="linear"``.

  If ``gate_object_name`` is set, the reward is multiplicatively gated by
  whether that object's root z is above ``gate_min_height``. Use this to
  zero out positive task reward when the cube is dropped, so termination
  is implicitly costly without an explicit negative drop_penalty.

  The tolerance kernel is *flat* inside the bound (no incentive to over-chase
  precision, which can destabilize a held grasp) and provides a *constant*
  gradient outside (PPO has the same shaping signal whether the cube is at
  err=0.5 or err=2.5).
  """
  command = cast(ReorientationCommand, env.command_manager.get_term(command_name))
  err = command.orientation_error
  # Distance outside the upper bound (err is always >= 0).
  d = (err - bound).clamp_min(0.0)
  decay = 1.0 + (value_at_margin - 1.0) * (d / margin).clamp(0.0, 1.0)
  reward = torch.where(err <= bound, torch.ones_like(err), decay)
  gate = _alive_gate(env, gate_object_name, gate_min_height)
  return reward if gate is None else reward * gate


def sustained_hold(
  env: ManagerBasedRlEnv,
  command_name: str,
  saturation_steps: float = 300.0,
  gate_object_name: str | None = None,
  gate_min_height: float = 0.1,
) -> torch.Tensor:
  """Dense, monotonic hold reward in [0, 1].

  Grows with the cumulative number of in-threshold steps this episode. That count
  only increases -- it pauses (never resets) when the goal advances -- so the
  reward is always up-or-flat and never dips at a goal switch, which would
  otherwise teach the policy that completing a hold is followed by a reward
  collapse. Saturates at 1 after ``saturation_steps`` total in-threshold steps.
  Replaces the sparse success bonus with a dense "reach and keep the pose" signal.

  If ``gate_object_name`` is set, the reward is multiplicatively gated by whether
  that object's root z is above ``gate_min_height`` (the ``cube_held`` gate).
  """
  command = cast(ReorientationCommand, env.command_manager.get_term(command_name))
  reward = (command.cumulative_hold / saturation_steps).clamp(max=1.0)
  gate = _alive_gate(env, gate_object_name, gate_min_height)
  return reward if gate is None else reward * gate


class NormalizedJointTorquePenalty(ManagerTermBase):
  """Effort penalty as the sum of squared torque fractions: ``sum((tau/tau_max)^2)``.

  Each joint is normalized by its own effort limit (which spans ~17x across this
  hand, CMC 3.3 N*m vs DIP 0.19 N*m), so the penalty is a dimensionless "fraction
  of capacity used" and a small distal joint at 90% of its limit costs the same as
  a big proximal one -- unlike a raw ``sum(tau^2)``, which the large joints
  dominate. ``tau_max`` is read once from the compiled model's actuator force range.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._asset: Entity = env.scene[asset_cfg.name]
    self._act_ids = asset_cfg.actuator_ids
    # actuator_forcerange[:, 1] is +effort_limit per actuator. The robot is the
    # only actuated entity, so the entity-local actuator ids line up with the host
    # model's global actuator order.
    forcerange = torch.as_tensor(
      env.sim.mj_model.actuator_forcerange[:, 1],
      device=env.device,
      dtype=torch.float32,
    )
    tau_max = (
      forcerange if isinstance(self._act_ids, slice) else forcerange[self._act_ids]
    )
    self._tau_max = tau_max.clamp_min(1e-3)

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del env, kwargs  # asset/limits resolved at init.
    tau = self._asset.data.actuator_force[:, self._act_ids]
    return torch.sum(torch.square(tau / self._tau_max), dim=-1)


def fingertip_object_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Fraction of fingertips in contact with the object, in ``[0, 1]``.

  Encourages a fingertip grasp. Reads a contact sensor whose primaries are the
  fingertip pad geoms and whose secondary is the object.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  return (found > 0).float().mean(dim=-1)


def cube_off_palm(
  env: ManagerBasedRlEnv,
  tip_sensor_name: str,
  palm_sensor_name: str,
) -> torch.Tensor:
  """1.0 when the cube is held by the fingertips and clear of the palm, else 0.0.

  Discourages the cube resting in the palm cup (a distal/fingertip grasp). Reads
  a fingertip-object and a palm-object contact sensor.
  """
  tip = env.scene[tip_sensor_name].data.found
  palm = env.scene[palm_sensor_name].data.found
  assert tip is not None and palm is not None
  tip_contact = (tip > 0).any(dim=-1)
  palm_contact = (palm > 0).any(dim=-1)
  return (tip_contact & ~palm_contact).float()
