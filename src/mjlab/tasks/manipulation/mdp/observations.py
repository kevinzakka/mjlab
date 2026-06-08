from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor
from mjlab.tasks.manipulation.mdp.commands import (
  LiftingCommand,
  MultiCubeLiftingCommand,
  ReorientationCommand,
)
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_inv,
  quat_mul,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


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
  base_quat_w = robot.data.root_link_quat_w
  distance_vec_b = quat_apply(quat_inv(base_quat_w), distance_vec_w)
  return distance_vec_b


def object_to_goal_distance(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Distance vector from object to goal in base frame."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, LiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a LiftingCommand, got {type(command)}"
    )
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  obj_pos_w = obj.data.root_link_pos_w
  goal_pos_w = command.target_pos
  distance_vec_w = goal_pos_w - obj_pos_w
  base_quat_w = robot.data.root_link_quat_w
  distance_vec_b = quat_apply(quat_inv(base_quat_w), distance_vec_w)
  return distance_vec_b


def ee_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """EE linear velocity in EE frame."""
  robot: Entity = env.scene[asset_cfg.name]
  ee_vel_w = robot.data.site_vel_w[:, asset_cfg.site_ids].squeeze(1)  # (B, 6)
  ee_vel_linear_w = ee_vel_w[:, :3]
  ee_quat_w = robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
  ee_vel_linear_ee = quat_apply(quat_inv(ee_quat_w), ee_vel_linear_w)
  return ee_vel_linear_ee


def target_position(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Target position in EE frame."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, (LiftingCommand, MultiCubeLiftingCommand)):
    raise TypeError(
      f"Command '{command_name}' must be a LiftingCommand or "
      f"MultiCubeLiftingCommand, got {type(command)}"
    )
  robot: Entity = env.scene[asset_cfg.name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  ee_quat_w = robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
  target_pos_w = command.target_pos
  target_pos_rel_w = target_pos_w - ee_pos_w
  target_pos_ee = quat_apply(quat_inv(ee_quat_w), target_pos_rel_w)
  return target_pos_ee


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
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ReorientationCommand):
    raise TypeError(
      f"Command '{command_name}' must be a ReorientationCommand, got {type(command)}"
    )
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
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ReorientationCommand):
    raise TypeError(
      f"Command '{command_name}' must be a ReorientationCommand, got {type(command)}"
    )
  denom = max(command.cfg.success_hold_steps, 1)
  return (command.hold_counter.float() / denom).clamp(0.0, 1.0).unsqueeze(-1)


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


def camera_rgb(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """RGB observation in CNN-compatible format (B, C, H, W)."""
  sensor: CameraSensor = env.scene[sensor_name]
  rgb_data = sensor.data.rgb  # (B, H, W, 3)
  assert rgb_data is not None, f"Camera '{sensor_name}' has no RGB data"
  rgb_data = rgb_data.permute(0, 3, 1, 2)  # (B, 3, H, W)
  return rgb_data.float() / 255.0


def camera_depth(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  cutoff_distance: float,
  min_depth: float = 0.01,
) -> torch.Tensor:
  """Depth observation in CNN-compatible format (B, 1, H, W)."""
  sensor: CameraSensor = env.scene[sensor_name]
  depth_data = sensor.data.depth  # (B, H, W, 1)
  assert depth_data is not None, f"Camera '{sensor_name}' has no depth data"
  depth_data = depth_data.permute(0, 3, 1, 2)  # (B, 1, H, W)
  depth_data_clipped = torch.clamp(depth_data, min=min_depth, max=cutoff_distance)
  return torch.clamp(depth_data_clipped / cutoff_distance, 0.0, 1.0)


def camera_segmentation(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Per-pixel typed segmentation in (B, 2, H, W) format."""
  sensor: CameraSensor = env.scene[sensor_name]
  seg_data = sensor.data.segmentation  # (B, H, W, 2)
  assert seg_data is not None, f"Camera '{sensor_name}' has no segmentation data"
  return seg_data.permute(0, 3, 1, 2)  # (B, 2, H, W)


def camera_target_cube_mask(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
) -> torch.Tensor:
  """Binary mask of the target cube selected by a MultiCubeLiftingCommand.

  Output shape: (B, 1, H, W) float32.
  """
  sensor: CameraSensor = env.scene[sensor_name]
  seg_data = sensor.data.segmentation  # (B, H, W, 2)
  assert seg_data is not None, f"Camera '{sensor_name}' has no segmentation data"
  obj_ids = seg_data[..., 0]  # (B, H, W)
  obj_types = seg_data[..., 1]  # (B, H, W)

  command = env.command_manager.get_term(command_name)
  assert isinstance(command, MultiCubeLiftingCommand)
  target_ids = command.target_geom_ids  # (B, K)

  # Only geom hits should participate in the target mask.
  is_geom = obj_types == int(mujoco.mjtObj.mjOBJ_GEOM)
  mask = (obj_ids.unsqueeze(-1) == target_ids.unsqueeze(1).unsqueeze(1)).any(-1)
  mask = mask & is_geom
  return mask.float().unsqueeze(1)  # (B, 1, H, W)
