from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.manipulation.mdp.commands import (
  LiftingCommand,
  MultiCubeLiftingCommand,
  ReorientationCommand,
)
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def staged_position_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  reaching_std: float,
  bringing_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Curriculum reward that gates lifting bonus on reaching progress.

  Returns reaching * (1 + bringing), where both terms are Gaussian kernels
  over position error. Ensures learning signal for approach before lift.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_pos_w = obj.data.root_link_pos_w
  reach_error = torch.sum(torch.square(ee_pos_w - obj_pos_w), dim=-1)
  reaching = torch.exp(-reach_error / reaching_std**2)
  position_error = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
  bringing = torch.exp(-position_error / bringing_std**2)
  return reaching * (1.0 + bringing)


def bring_object_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  std: float,
) -> torch.Tensor:
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))
  position_error = torch.sum(
    torch.square(command.target_pos - obj.data.root_link_pos_w), dim=-1
  )
  return torch.exp(-position_error / std**2)


def multi_cube_staged_position_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  reaching_std: float,
  bringing_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Staged reward for the target cube selected by MultiCubeLiftingCommand."""
  robot: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, MultiCubeLiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a MultiCubeLiftingCommand, got {type(command)}"
    )
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_pos_w = command.target_object_pos()
  reach_error = torch.sum(torch.square(ee_pos_w - obj_pos_w), dim=-1)
  reaching = torch.exp(-reach_error / reaching_std**2)
  position_error = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
  bringing = torch.exp(-position_error / bringing_std**2)
  return reaching * (1.0 + bringing)


def multi_cube_bring_object_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Gaussian reward for bringing the selected target cube to goal."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, MultiCubeLiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a MultiCubeLiftingCommand, got {type(command)}"
    )
  obj_pos_w = command.target_object_pos()
  position_error = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
  return torch.exp(-position_error / std**2)


def cube_orientation_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Exponential kernel on the cube-to-goal orientation error (radians).

  Uses ``exp(-err / std)`` (exponential in the angle, not Gaussian) so the reward
  stays meaningfully nonzero across the full [0, pi] range and provides a gradient
  from any starting error, which a peaked Gaussian does not. Reads the error cached by
  the command in _update_metrics, so it is consistent with the success detection even
  on the step a new goal is sampled.
  """
  command = cast(ReorientationCommand, env.command_manager.get_term(command_name))
  return torch.exp(-command.orientation_error / std)


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

  Compared to ``cube_orientation_tracking`` (exp kernel): the tolerance kernel
  is *flat* inside the bound (no incentive to over-chase precision, which can
  destabilize a held grasp) and provides a *constant* gradient outside (PPO
  has the same shaping signal whether the cube is at err=0.5 or err=2.5).
  """
  command = cast(ReorientationCommand, env.command_manager.get_term(command_name))
  err = command.orientation_error
  # Distance outside the upper bound (err is always >= 0).
  d = (err - bound).clamp_min(0.0)
  decay = 1.0 + (value_at_margin - 1.0) * (d / margin).clamp(0.0, 1.0)
  reward = torch.where(err <= bound, torch.ones_like(err), decay)
  gate = _alive_gate(env, gate_object_name, gate_min_height)
  return reward if gate is None else reward * gate


def cube_orientation_success_bonus(
  env: ManagerBasedRlEnv,
  command_name: str,
  gate_object_name: str | None = None,
  gate_min_height: float = 0.1,
) -> torch.Tensor:
  """Sparse bonus on each step the cube is within the goal threshold.

  If ``gate_object_name`` is set, the bonus is zeroed out when the cube's
  root z drops below ``gate_min_height`` (multiplicative ``cube_held`` gate).
  """
  command = cast(ReorientationCommand, env.command_manager.get_term(command_name))
  gate = _alive_gate(env, gate_object_name, gate_min_height)
  return command.at_goal if gate is None else command.at_goal * gate


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


def cube_rotation_toward_goal(
  env: ManagerBasedRlEnv,
  command_name: str,
  gate_object_name: str | None = None,
  gate_min_height: float = 0.1,
) -> torch.Tensor:
  """Reward for cube angular velocity projected onto the direction that closes
  the goal orientation error.

  Shaping signal that pays positive reward when the cube is *being rotated*
  toward the goal, independent of whether the goal has been reached. Without
  this, the tolerance-kernel reward only pays for *being* near the goal --
  there's no gradient for the *act* of rotating, so PPO settles into a local
  optimum of "hold the cube stably anywhere and wiggle fingers cosmetically"
  rather than discovering regrasps. With this term, any rotation in the
  correct direction earns positive reward, giving direct gradient signal that
  active reorientation is valuable.

  Computation:
    needed_axis_angle = log(goal_quat * cube_quat^-1)    [direction of needed rotation]
    needed_dir        = needed_axis_angle / |needed_axis_angle|
    alignment         = max(0, cube_omega . needed_dir)  [in rad/s, clamped to non-neg]

  Clamped to non-negative so wrong-direction motion is "free" (no penalty)
  rather than punished -- exploration of motions in any direction should be
  cost-free at the reward layer; the cube_held gate is the only cost of
  failing.

  If ``gate_object_name`` is set, multiplicatively gated by cube_held.
  """
  command = cast(ReorientationCommand, env.command_manager.get_term(command_name))
  cube = command.object  # already an Entity ref cached by the command
  # Rotation FROM cube TO goal: goal_quat * cube_quat^-1. Its log gives the
  # axis-angle vector pointing in the needed direction with magnitude=angle.
  quat_diff = quat_mul(command.goal_quat, quat_conjugate(cube.data.root_link_quat_w))
  needed = axis_angle_from_quat(quat_diff)
  needed_norm = needed.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  needed_dir = needed / needed_norm
  omega = cube.data.root_link_ang_vel_w
  alignment = (omega * needed_dir).sum(dim=-1).clamp_min(0.0)
  gate = _alive_gate(env, gate_object_name, gate_min_height)
  return alignment if gate is None else alignment * gate


def joint_pos_deviation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Sum of squared joint deviations from the entity's default joint pose.

  Used with a negative weight as a pose regularizer that pulls the hand back
  toward its default ``init_state.joint_pos``. Unlike the ``mdp.posture`` exp
  kernel, this penalty grows without bound -- which is exactly what's needed
  to combat degenerate-pose collapse (flat fingers, single-finger grasps),
  where the exp kernel saturates and provides no gradient back to home.
  """
  asset: Entity = env.scene[asset_cfg.name]
  err = (asset.data.joint_pos - asset.data.default_joint_pos)[:, asset_cfg.joint_ids]
  return torch.sum(err**2, dim=-1)


def cube_orientation_hold_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
  exponent: float = 3.0,
) -> torch.Tensor:
  """Back-loaded ramp ``(hold_counter / success_hold_steps) ** exponent``.

  Pays an increasing fraction of 1.0 for each consecutive in-threshold step,
  rewarding the policy for *staying* in the window rather than just touching
  it. The default ``exponent=3.0`` concentrates the reward in the last 1-2
  steps of the streak: with hold_steps=5, marginals are
  [0.008, 0.056, 0.152, 0.296, 0.488] -- the final step is worth ~0.5 of the
  total, roughly equal to the sum of all earlier steps. This back-loading is
  intentional: a linear (exponent=1) ramp is gamed by policies that bounce
  the cube just past the threshold at counter=N-1 and re-enter, harvesting
  near-maximal partial credit without ever completing a hold. With exponent
  >= 3, the marginal value of the final step dominates, so deliberately
  resetting the counter is strictly more costly than continuing.

  Bounded to [0, 1] and resets to 0 the step after a hold completes
  (``_sample_goal`` clears ``hold_counter``) or the moment the cube leaves
  the threshold.
  """
  command = cast(ReorientationCommand, env.command_manager.get_term(command_name))
  denom = max(command.cfg.success_hold_steps, 1)
  progress = (command.hold_counter.float() / denom).clamp(0.0, 1.0)
  return progress.pow(exponent)


def cube_stay_near_palm(
  env: ManagerBasedRlEnv,
  object_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Gaussian kernel keeping the cube near a hand site (discourages dropping)."""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ref_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  dist_sq = torch.sum(torch.square(obj.data.root_link_pos_w - ref_pos_w), dim=-1)
  return torch.exp(-dist_sq / std**2)


def joint_velocity_hinge_penalty(
  env: ManagerBasedRlEnv,
  max_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Quadratic hinge penalty on joint velocities exceeding a symmetric limit.

  Penalizes only the amount by which |v| exceeds max_vel. Returns a negative
  penalty, shaped as the negative squared L2 norm of the excess velocities.
  """
  robot: Entity = env.scene[asset_cfg.name]
  joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  excess = (joint_vel.abs() - max_vel).clamp_min(0.0)
  return (excess**2).sum(dim=-1)


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
