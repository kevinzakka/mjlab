from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation.mdp.commands import (
  LiftingCommand,
  MultiCubeLiftingCommand,
  ReorientationCommand,
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


def cube_orientation_tolerance(
  env: ManagerBasedRlEnv,
  command_name: str,
  bound: float = 0.1,
  margin: float = 3.141592653589793,
  value_at_margin: float = 0.1,
) -> torch.Tensor:
  """Tolerance kernel: 1.0 inside ``[0, bound]``, linear decay outside.

  Reward is exactly 1.0 for any orientation error in the "good enough" band
  ``[0, bound]``, so the policy isn't pulled to chase err -> 0 once it is
  comfortably close. Outside the bound, reward decays linearly to
  ``value_at_margin`` at error = ``bound + margin``, then is clamped at that
  floor. This matches the dm_control / mujoco_playground ``tolerance`` shape
  with ``sigmoid="linear"`` and is the standard dense signal for in-hand
  reorientation on LEAP / Allegro tasks.

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
  return torch.where(err <= bound, torch.ones_like(err), decay)


def cube_orientation_success_bonus(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Sparse bonus on each step the cube is within the goal threshold."""
  command = cast(ReorientationCommand, env.command_manager.get_term(command_name))
  return command.at_goal


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
