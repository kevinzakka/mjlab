from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_box_plus,
  quat_error_magnitude,
  random_orientation,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class ReorientationCommand(CommandTerm):
  """Goal orientation for in-hand cube reorientation.

  Episode reset draws a uniform-SO(3) goal. A goal counts as reached once the cube
  stays within ``success_threshold`` for ``success_hold_steps`` consecutive steps;
  the goal is then held fixed for a further ``goal_switch_delay`` steps (the success
  window, so a success is a *parked* pose rather than a grazed one) before advancing.
  The next goal is either a fresh uniform-SO(3) draw or a bounded perturbation of the
  held one, per ``success_resample_full_so3``. The cube itself is reset by an event;
  this term only manages the goal and draws a translucent "ghost" cube above the hand
  at the goal orientation.
  """

  cfg: ReorientationCommandCfg

  def __init__(self, cfg: ReorientationCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.object: Entity = env.scene[cfg.entity_name]
    self.robot: Entity = env.scene[cfg.robot_name]
    self.marker: Entity | None = (
      env.scene[cfg.marker_name] if cfg.marker_name is not None else None
    )
    self._marker_offset = torch.tensor(cfg.viz.offset, device=self.device)
    # write_mocap_pose's "all envs" path (env_ids=None) collapses the mocap dim
    # and breaks the broadcast, so we always pass an explicit env-ids tensor.
    self._all_env_ids = torch.arange(self.num_envs, device=self.device)

    self.goal_quat = torch.zeros(self.num_envs, 4, device=self.device)
    self.goal_quat[:, 0] = 1.0
    # Cached each step in _update_metrics, before any resample-on-success.
    self.orientation_error = torch.zeros(self.num_envs, device=self.device)
    self.within_threshold = torch.zeros(self.num_envs, device=self.device)
    self.hold_counter = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )
    self.at_goal = torch.zeros(self.num_envs, device=self.device)
    self.success_count = torch.zeros(self.num_envs, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)
    # Success window: after a hold completes, keep the goal fixed for
    # goal_switch_delay more steps (the policy must park the pose, not graze it)
    # before advancing to the next goal.
    self.in_success_window = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.window_timer = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )
    # Cumulative in-threshold steps this episode. Only grows (resets on episode
    # reset, never on a goal switch), so a reward built from it is monotonic and
    # never dips when the goal advances.
    self.cumulative_hold = torch.zeros(self.num_envs, device=self.device)

    self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["within_threshold"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["success_count"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["in_success_window"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["window_timer"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["cumulative_hold"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.goal_quat

  def compute_success(self) -> torch.Tensor:
    return self.at_goal.bool()

  def _update_metrics(self) -> None:
    self.orientation_error = quat_error_magnitude(
      self.object.data.root_link_quat_w, self.goal_quat
    )
    within = self.orientation_error < self.cfg.success_threshold
    self.within_threshold = within.float()
    self.cumulative_hold = self.cumulative_hold + self.within_threshold

    # Approaching phase (not yet in the success window): hold within threshold to
    # complete a success. ``at_goal`` is a one-shot pulse on the completing step.
    approaching = ~self.in_success_window
    self.hold_counter = torch.where(
      approaching & within,
      self.hold_counter + 1,
      torch.where(approaching, torch.zeros_like(self.hold_counter), self.hold_counter),
    )
    just_succeeded = approaching & (self.hold_counter >= self.cfg.success_hold_steps)

    # Enter / advance the success window. Timer resets on a fresh success, then
    # counts up each step until goal_switch_delay (handled in _update_command).
    self.in_success_window = self.in_success_window | just_succeeded
    self.window_timer = torch.where(
      just_succeeded,
      torch.zeros_like(self.window_timer),
      torch.where(self.in_success_window, self.window_timer + 1, self.window_timer),
    )
    self.at_goal = just_succeeded.float()
    self.episode_success = torch.maximum(self.episode_success, self.at_goal)

    self.metrics["orientation_error"] = self.orientation_error
    self.metrics["within_threshold"] = self.within_threshold
    self.metrics["at_goal"] = self.at_goal
    self.metrics["success_count"] = self.success_count
    self.metrics["episode_success"] = self.episode_success
    self.metrics["in_success_window"] = self.in_success_window.float()
    self.metrics["window_timer"] = self.window_timer.float()
    self.metrics["cumulative_hold"] = self.cumulative_hold

  def _sample_goal(self, env_ids: torch.Tensor) -> None:
    self.goal_quat[env_ids] = random_orientation(len(env_ids), device=str(self.device))
    self.hold_counter[env_ids] = 0

  def _perturb_goal(self, env_ids: torch.Tensor) -> None:
    # Compose the current goal with a small random rotation: axis uniform on S^2,
    # angle uniform in [0, success_resample_max_angle]. Bounds the angular distance
    # between consecutive goals so chasing the next goal after a completed hold is
    # cheap (~few control steps) instead of paying a full random-SO(3) chase tax.
    n = len(env_ids)
    axis = torch.randn(n, 3, device=self.device)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    angle = torch.rand(n, device=self.device) * self.cfg.success_resample_max_angle
    delta = axis * angle.unsqueeze(-1)
    self.goal_quat[env_ids] = quat_box_plus(self.goal_quat[env_ids], delta)
    self.hold_counter[env_ids] = 0

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Episode reset (and timer fallback): fresh full-SO(3) goal, clear episode trackers.
    self.episode_success[env_ids] = 0.0
    self.success_count[env_ids] = 0.0
    self.cumulative_hold[env_ids] = 0.0
    self.in_success_window[env_ids] = False
    self.window_timer[env_ids] = 0
    self._sample_goal(env_ids)
    # Pose the marker at the freshly drawn goal now, so it is correct on the reset frame
    # rather than snapping into place one step later (it is otherwise only posed in
    # _update_command, which runs on step).
    self._pose_marker(env_ids)

  def _update_command(self) -> None:
    # Count each completed success once (at_goal is a one-shot pulse).
    success_ids = self.at_goal.nonzero().flatten()
    if len(success_ids) > 0:
      self.success_count[success_ids] += 1.0

    # Advance the goal only once the success window elapses, so a success means a
    # held pose rather than a grazed one. The next goal is either a fresh
    # uniform-SO(3) draw (a full reorientation each time) or a bounded perturbation
    # of the held goal, per success_resample_full_so3.
    switch = self.in_success_window & (self.window_timer >= self.cfg.goal_switch_delay)
    switch_ids = switch.nonzero().flatten()
    if len(switch_ids) > 0:
      if self.cfg.success_resample_full_so3:
        self._sample_goal(switch_ids)
      else:
        self._perturb_goal(switch_ids)
      self.in_success_window[switch_ids] = False
      self.window_timer[switch_ids] = 0

    # Pose the textured goal-marker cube above the hand at the goal orientation.
    self._pose_marker(self._all_env_ids)

  def _pose_marker(self, env_ids: torch.Tensor) -> None:
    """Pose the goal-marker mocap cube above the hand at the current goal orientation."""
    if self.marker is None:
      return
    pos = self.robot.data.root_link_pos_w[env_ids] + self._marker_offset
    pose = torch.cat([pos, self.goal_quat[env_ids]], dim=-1)
    self.marker.write_mocap_pose_to_sim(pose, env_ids=env_ids)


@dataclass(kw_only=True)
class ReorientationCommandCfg(CommandTermCfg):
  entity_name: str
  """Name of the cube entity to reorient."""
  robot_name: str = "robot"
  """Name of the hand entity (used to place the goal marker above the palm)."""
  marker_name: str | None = None
  """Name of the (fixed-base mocap) goal-marker entity to pose at the goal orientation.
  If None, no marker is posed."""
  success_threshold: float = 0.1
  """Orientation error (radians) below which a step counts as in-threshold."""
  success_hold_steps: int = 5
  """Consecutive in-threshold steps required before the goal counts as reached.
  Setting to 1 recovers the old single-step success criterion."""
  goal_switch_delay: int = 0
  """Steps to hold the achieved goal (the success window) before advancing to the
  next one. 0 advances immediately on hold completion; larger values require the
  policy to park the pose, rewarding stable holds over grazing the threshold."""
  success_resample_max_angle: float = math.pi / 4
  """Maximum angular distance (radians) between the just-achieved goal and the
  next one. Bounds the chase tax between consecutive goals. Ignored when
  ``success_resample_full_so3`` is True."""
  success_resample_full_so3: bool = False
  """If True, each goal switch draws a fresh uniform-SO(3) goal (a full
  reorientation every time) instead of a bounded perturbation of the held goal."""

  @dataclass
  class VizCfg:
    cube_half_extent: float = 0.03
    offset: tuple[float, float, float] = (0.0, 0.0, 0.15)
    color: tuple[float, float, float, float] = (0.2, 0.8, 0.2, 0.35)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> ReorientationCommand:
    return ReorientationCommand(self, env)
