from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.reorient.mdp.commands import ReorientationCommand
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_conjugate,
  quat_inv,
  quat_mul,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _reorient_command(
  env: ManagerBasedRlEnv, command_name: str
) -> ReorientationCommand:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ReorientationCommand):
    raise TypeError(
      f"Command '{command_name}' must be a ReorientationCommand, got {type(command)}"
    )
  return command


def ee_to_object_distance(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Distance vector from end effector to object in base frame."""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_pos_w = obj.data.root_link_pos_w
  distance_vec_w = obj_pos_w - ee_pos_w
  return quat_apply_inverse(robot.data.root_link_quat_w, distance_vec_w)


def _quat_to_6d(quat: torch.Tensor) -> torch.Tensor:
  """First two columns of the rotation matrix as a continuous 6D rotation rep.

  Input quaternion shape (..., 4); output shape (..., 6).
  """
  mat = matrix_from_quat(quat)  # (..., 3, 3)
  return mat[..., :, :2].reshape(*quat.shape[:-1], 6)


def object_orientation_6d(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Cube orientation relative to the hand base frame, as a 6D rotation rep."""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  rel_quat = quat_mul(quat_inv(robot.data.root_link_quat_w), obj.data.root_link_quat_w)
  return _quat_to_6d(rel_quat)


def object_to_goal_orientation_6d(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Rotation from cube to goal, in the hand base frame, as a 6D rotation rep.

  This is the primary "how much to rotate" signal. The error rotation's magnitude is
  frame independent, but we express it in the hand base frame (same basis as
  ``object_orientation_6d``) so all orientation observations share one frame and stay
  consistent if the hand mounting orientation ever changes.
  """
  command = _reorient_command(env, command_name)
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  base_inv = quat_inv(robot.data.root_link_quat_w)
  goal_in_base = quat_mul(base_inv, command.goal_quat)
  cube_in_base = quat_mul(base_inv, obj.data.root_link_quat_w)
  rel_quat = quat_mul(goal_in_base, quat_conjugate(cube_in_base))
  return _quat_to_6d(rel_quat)


def goal_hold_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Normalized hold progress: ``hold_counter / success_hold_steps`` in [0, 1].

  Tells the policy how close it is to completing the current hold or, during
  the SUCCESS_WINDOW dwell, how stably it is staying in the threshold. 0 means
  it just lost the hold (or fresh goal); 1 means at or past the hold target.
  """
  command = _reorient_command(env, command_name)
  denom = max(command.cfg.success_hold_steps, 1)
  return (command.hold_counter.float() / denom).clamp(0.0, 1.0).unsqueeze(-1)


def goal_window_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Normalized success-window progress: ``window_timer / goal_switch_delay`` in [0, 1].

  Counts up only after a hold completes (the success window) and resets to 0 when
  the goal advances. Paired with :func:`goal_hold_progress` this exposes the full
  success state machine, so the sparse success bonus and the goal-switch timing are
  observable (intended as a critic-only privileged term).
  """
  command = _reorient_command(env, command_name)
  denom = max(command.cfg.goal_switch_delay, 1)
  return (command.window_timer.float() / denom).clamp(0.0, 1.0).unsqueeze(-1)


def object_lin_vel_b(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Cube linear velocity expressed in the hand base frame."""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  return quat_apply_inverse(robot.data.root_link_quat_w, obj.data.root_link_lin_vel_w)


def object_ang_vel_b(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Cube angular velocity expressed in the hand base frame."""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  return quat_apply_inverse(robot.data.root_link_quat_w, obj.data.root_link_ang_vel_w)


def _points_to_base_frame(robot: Entity, points_w: torch.Tensor) -> torch.Tensor:
  """Rotate world-frame points (B, n, 3) into the hand base frame."""
  b, n, _ = points_w.shape
  base_quat = robot.data.root_link_quat_w.unsqueeze(1).expand(b, n, 4).reshape(b * n, 4)
  return quat_apply_inverse(base_quat, points_w.reshape(b * n, 3)).reshape(b, n, 3)


def fingertip_to_object(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Fingertip positions relative to the object center, in the hand base frame.

  Flattened to (B, 3 * num_sites). Tells the policy where each contact point sits
  on/around the object, which is the key signal for finger gaiting and re-grasping.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  tip_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids]  # (B, n, 3)
  rel_w = tip_pos_w - obj.data.root_link_pos_w.unsqueeze(1)
  return _points_to_base_frame(robot, rel_w).reshape(rel_w.shape[0], -1)


def fingertip_to_palm(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Fingertip positions relative to the hand base, in the hand base frame.

  Flattened to (B, 3 * num_sites). An explicit Cartesian encoding of hand shape
  (forward kinematics) that the policy would otherwise have to infer from joint angles.
  """
  robot: Entity = env.scene[asset_cfg.name]
  tip_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids]  # (B, n, 3)
  rel_w = tip_pos_w - robot.data.root_link_pos_w.unsqueeze(1)
  return _points_to_base_frame(robot, rel_w).reshape(rel_w.shape[0], -1)


# Privileged extrinsics (critic-only): the per-env domain-randomized physics. The actor
# can't observe these on the real robot, but feeding them to the value function makes its
# estimates physics-aware (lower-variance advantages), and they are the latent an RMA
# teacher would encode. Bodies/geoms are selected by name via SceneEntityCfg and mapped
# to the model through the entity's own index table -- no hardcoded indices. Returned
# raw; obs normalization handles the differing scales.

_CUBE_BODY = SceneEntityCfg("cube", body_names=(".*",))
_CUBE_GEOM = SceneEntityCfg("cube", geom_names=("cube_geom",))


def object_mass(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _CUBE_BODY
) -> torch.Tensor:
  """Per-env object body mass (kg). Shape (B, n_bodies)."""
  obj: Entity = env.scene[asset_cfg.name]
  return env.sim.model.body_mass[:, obj.indexing.body_ids[asset_cfg.body_ids]]


def object_size(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _CUBE_GEOM
) -> torch.Tensor:
  """Per-env object geom half-extent (m); isotropic, so axis 0 is the full size."""
  obj: Entity = env.scene[asset_cfg.name]
  return env.sim.model.geom_size[:, obj.indexing.geom_ids[asset_cfg.geom_ids], 0]


def object_friction(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _CUBE_GEOM
) -> torch.Tensor:
  """Per-env object sliding + torsional friction. Shape (B, n_geoms * 2)."""
  obj: Entity = env.scene[asset_cfg.name]
  geom_ids = obj.indexing.geom_ids[asset_cfg.geom_ids]
  return env.sim.model.geom_friction[:, geom_ids, :2].reshape(env.num_envs, -1)


def object_com(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _CUBE_BODY
) -> torch.Tensor:
  """Per-env object center-of-mass position in the body frame (m). Shape (B, 3)."""
  obj: Entity = env.scene[asset_cfg.name]
  body_ids = obj.indexing.body_ids[asset_cfg.body_ids]
  return env.sim.model.body_ipos[:, body_ids].reshape(env.num_envs, -1)


def object_external_wrench(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _CUBE_BODY
) -> torch.Tensor:
  """External force+torque currently applied to the object (the active impulse).

  Time-varying, unlike the other extrinsics, so the critic can account for a
  perturbation in flight instead of seeing it as value noise. Shape (B, 6)."""
  obj: Entity = env.scene[asset_cfg.name]
  body_ids = obj.indexing.body_ids[asset_cfg.body_ids]
  return env.sim.data.xfrc_applied[:, body_ids].reshape(env.num_envs, -1)
