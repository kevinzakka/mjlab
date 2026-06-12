"""Reset events for the in-hand reorientation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_euler_xyz,
  quat_mul,
  random_orientation,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ("reset_hand_and_object",)

_DEFAULT_HAND_CFG = SceneEntityCfg("robot")
_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")


def reset_hand_and_object(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  cradle_offset_b: tuple[float, float, float],
  hand_pitch_range: tuple[float, float] = (0.0, 0.0),
  hand_roll_range: tuple[float, float] = (0.0, 0.0),
  hand_yaw_range: tuple[float, float] = (0.0, 0.0),
  position_noise: float = 0.0,
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> None:
  """Tilt the hand and nestle the object in its cradle, in one atomic reset.

  The cradle is rigid to the hand, so the object is placed in the hand-base body
  frame (``cradle_offset_b``) *after* the tilt is sampled. This keeps the object
  cradled at every hand tilt rather than at a fixed world point. The hand is tilted by
  a sampled roll/pitch/yaw about its base axes -- equivalently, this randomizes the
  gravity direction in the hand frame (mounting slop of a fixed fixture). The object's
  orientation is sampled uniformly over SO(3).
  """
  env_ids = resolve_env_ids(env, env_ids)
  n = len(env_ids)
  device = env.device

  hand: Entity = env.scene[hand_cfg.name]
  obj: Entity = env.scene[object_cfg.name]
  origins = env.scene.env_origins[env_ids]

  # Hand: random roll/pitch/yaw tilt applied in the base body frame.
  base = hand.data.default_root_state[env_ids]
  base_pos = base[:, 0:3] + origins
  roll = sample_uniform(hand_roll_range[0], hand_roll_range[1], n, device)
  pitch = sample_uniform(hand_pitch_range[0], hand_pitch_range[1], n, device)
  yaw = sample_uniform(hand_yaw_range[0], hand_yaw_range[1], n, device)
  base_quat = quat_mul(base[:, 3:7], quat_from_euler_xyz(roll, pitch, yaw))
  hand.write_mocap_pose_to_sim(torch.cat([base_pos, base_quat], dim=-1), env_ids)

  # Object: nestled in the tilted cradle, random orientation, zero velocity.
  offset = torch.tensor(cradle_offset_b, device=device).repeat(n, 1)
  obj_pos = base_pos + quat_apply(base_quat, offset)
  if position_noise > 0.0:
    obj_pos += sample_uniform(-position_noise, position_noise, (n, 3), device)
  obj_pose = torch.cat([obj_pos, random_orientation(n, device)], dim=-1)
  obj.write_root_link_pose_to_sim(obj_pose, env_ids)
  obj.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids)
