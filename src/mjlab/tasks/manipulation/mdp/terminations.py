from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def object_dropped(
  env: ManagerBasedRlEnv,
  object_name: str,
  max_distance: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """True when the cube has drifted too far from a hand site (i.e., dropped)."""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ref_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  distance = torch.norm(obj.data.root_link_pos_w - ref_pos_w, dim=-1)
  return distance > max_distance


def object_velocity_out_of_bounds(
  env: ManagerBasedRlEnv,
  object_name: str,
  max_lin_vel: float,
  max_ang_vel: float,
) -> torch.Tensor:
  """True when the object's speed exceeds a limit or is non-finite.

  The light, low-inertia cube can trigger constraint-solver blow-ups; terminating the
  env on a runaway velocity resets it to a finite state before the divergence reaches
  NaN and crashes training. The non-finite check is required because ``nan > limit`` is
  False, so a velocity that already went NaN would otherwise slip through.
  """
  obj: Entity = env.scene[object_name]
  lin = torch.norm(obj.data.root_link_lin_vel_w, dim=-1)
  ang = torch.norm(obj.data.root_link_ang_vel_w, dim=-1)
  out_of_bounds = (lin > max_lin_vel) | (ang > max_ang_vel)
  return out_of_bounds | ~torch.isfinite(lin) | ~torch.isfinite(ang)
