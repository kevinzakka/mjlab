"""Dynamic hand-cage drop detection and escape penalty for in-hand manipulation.

The cage is the axis-aligned bounding box of a set of hand points (fingertips +
wrist), recomputed every step in the palm frame, so it tracks the hand as the
fingers move. The box is padded by ``margin`` on every face, with an extra
``up_margin`` on the open/palm-normal face (``up_axis``) so the object can lift
off the palm during a reorientation without counting as escaped.

The object's L1 escape distance from this box drives both a debounced drop
termination (:func:`cage_drop`) and a shaped penalty (:class:`CageEscapePenalty`,
which also draws the live cage in the viewer).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg
  from mjlab.viewer.debug_visualizer import DebugVisualizer

__all__ = (
  "cage_escape_distance",
  "cage_bounds_in_palm",
  "cage_drop",
  "CageEscapePenalty",
)

_COUNTER_ATTR = "_cage_outside_counter"


def _palm_frame_points(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return (palm_pos_w, points_in_palm, palm_quat_w) for the cage.

  ``asset_cfg`` selects the palm frame (its first body) and the cage points
  (its sites: fingertips + wrist).
  """
  robot: Entity = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  if isinstance(body_ids, slice):
    raise ValueError("cage asset_cfg must name the palm body via body_names.")
  palm_pose = robot.data.body_link_pose_w[:, body_ids[0]]
  palm_pos, palm_quat = palm_pose[:, :3], palm_pose[:, 3:7]

  pts_w = robot.data.site_pos_w[:, asset_cfg.site_ids]  # (N, M, 3)
  n_pts = pts_w.shape[1]
  pts_palm = quat_apply_inverse(
    palm_quat.unsqueeze(1).expand(-1, n_pts, -1), pts_w - palm_pos.unsqueeze(1)
  )
  return palm_pos, pts_palm, palm_quat


def cage_bounds_in_palm(
  points_in_palm: torch.Tensor,
  margin: float,
  up_axis: int | None = None,
  up_margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """AABB (lo, hi) of the cage points, padded by ``margin``.

  The ``up_axis`` upper face is instead padded by ``up_margin`` (the open palm
  side, where the object may rise during a reorientation).
  """
  mn = points_in_palm.min(dim=1).values
  mx = points_in_palm.max(dim=1).values
  lo = mn - margin
  hi = mx + margin
  if up_axis is not None:
    hi = hi.clone()
    hi[:, up_axis] = mx[:, up_axis] + up_margin
  return lo, hi


def cage_escape_distance(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg,
  margin: float = 0.02,
  up_axis: int | None = None,
  up_margin: float = 0.04,
) -> torch.Tensor:
  """L1 distance the object is outside the palm-frame hand cage (0 if inside)."""
  obj: Entity = env.scene[object_name]
  palm_pos, pts_palm, palm_quat = _palm_frame_points(env, asset_cfg)
  lo, hi = cage_bounds_in_palm(pts_palm, margin, up_axis, up_margin)
  cube_palm = quat_apply_inverse(palm_quat, obj.data.root_link_pos_w - palm_pos)
  outside = (lo - cube_palm).clamp_min(0.0) + (cube_palm - hi).clamp_min(0.0)
  return outside.sum(dim=-1)


def cage_drop(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg,
  margin: float = 0.02,
  up_axis: int | None = None,
  up_margin: float = 0.04,
  max_outside_steps: int = 10,
  grace_steps: int = 0,
) -> torch.Tensor:
  """Terminate when the object stays outside the hand cage for N steps.

  The debounce counter increments each step the object is outside and resets to
  zero whenever it is inside. Reset is implicit: every episode reset nestles the
  object back in the cradle (inside the cage), so the counter zeroes on the next
  step without an explicit reset hook. During the first ``grace_steps`` of an
  episode the counter is held at zero, so the object can drop and settle into the
  cage (e.g. during an action warmup) without false-terminating.
  """
  escape = cage_escape_distance(env, object_name, asset_cfg, margin, up_axis, up_margin)
  outside = escape > 0.0
  if grace_steps > 0:
    outside &= env.episode_length_buf >= grace_steps

  counter = getattr(env, _COUNTER_ATTR, None)
  if counter is None or counter.shape[0] != env.num_envs:
    counter = torch.zeros(env.num_envs, device=env.device)
  counter = torch.where(outside, counter + 1.0, torch.zeros_like(counter))
  setattr(env, _COUNTER_ATTR, counter)

  return counter >= max_outside_steps


class CageEscapePenalty(ManagerTermBase):
  """Penalize (L1) escape distance from the hand cage, and draw the live cage.

  The penalty is plain distance-proportional (no escalation counter); the reward
  weight is applied by the term config. ``debug_vis`` renders the current cage as
  a box in the palm frame so it can be toggled in the viewer.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    p = cfg.params
    self._object_name: str = p["object_name"]
    self._asset_cfg: SceneEntityCfg = p["asset_cfg"]
    self._margin: float = p.get("margin", 0.02)
    self._up_axis: int | None = p.get("up_axis", None)
    self._up_margin: float = p.get("up_margin", 0.04)
    # Drives the viewer's per-reward debug-vis toggle; starts from draw_cage and
    # is flipped live by the GUI checkbox. Off by default (the cage box is ugly).
    self._debug_vis_enabled: bool = p.get("draw_cage", False)

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    return cage_escape_distance(
      env,
      self._object_name,
      self._asset_cfg,
      self._margin,
      self._up_axis,
      self._up_margin,
    )

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled:  # toggled by the viewer; off by default
      return
    env = self._env
    robot: Entity = env.scene[self._asset_cfg.name]
    body_ids = self._asset_cfg.body_ids
    if isinstance(body_ids, slice):
      return
    palm_pose = robot.data.body_link_pose_w[:, body_ids[0]]
    palm_pos, palm_quat = palm_pose[:, :3], palm_pose[:, 3:7]
    _, pts_palm, _ = _palm_frame_points(env, self._asset_cfg)
    lo, hi = cage_bounds_in_palm(pts_palm, self._margin, self._up_axis, self._up_margin)
    center_palm = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    palm_rot = matrix_from_quat(palm_quat)
    for i in visualizer.get_env_indices(env.num_envs):
      rot = palm_rot[i].detach().cpu().numpy().astype(np.float64)
      pos = palm_pos[i].detach().cpu().numpy().astype(np.float64)
      center = pos + rot @ center_palm[i].detach().cpu().numpy().astype(np.float64)
      visualizer.add_box(
        center=center,
        size=half[i].detach().cpu().numpy().astype(np.float64),
        mat=rot,
        color=(0.2, 0.8, 0.2, 0.25),
        label="cage",
      )
