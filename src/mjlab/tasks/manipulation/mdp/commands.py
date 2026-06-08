from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_box_plus,
  quat_error_magnitude,
  quat_from_euler_xyz,
  random_orientation,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class LiftingCommand(CommandTerm):
  cfg: LiftingCommandCfg

  def __init__(self, cfg: LiftingCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.object: Entity = env.scene[cfg.entity_name]
    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)

    self.metrics["object_height"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _update_metrics(self) -> None:
    object_pos_w = self.object.data.root_link_pos_w
    object_height = object_pos_w[:, 2]
    position_error = torch.norm(self.target_pos - object_pos_w, dim=-1)
    at_goal = (position_error < self.cfg.success_threshold).float()

    # Latch episode_success to 1 once goal is reached.
    self.episode_success = torch.maximum(self.episode_success, at_goal)

    self.metrics["object_height"] = object_height
    self.metrics["position_error"] = position_error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success

  def compute_success(self) -> torch.Tensor:
    position_error = self.metrics["position_error"]
    return position_error < self.cfg.success_threshold

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)

    # Reset episode success for resampled envs.
    self.episode_success[env_ids] = 0.0

    # Set target position based on difficulty mode.
    if self.cfg.difficulty == "fixed":
      target_pos = torch.tensor(
        [0.4, 0.0, 0.3], device=self.device, dtype=torch.float32
      ).expand(n, 3)
      self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]
    else:
      assert self.cfg.difficulty == "dynamic"
      r = self.cfg.target_position_range
      lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
      upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
      target_pos = sample_uniform(lower, upper, (n, 3), device=self.device)
      self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]

    # Reset object to new position.
    if self.cfg.object_pose_range is not None:
      r = self.cfg.object_pose_range
      lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
      upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
      pos = sample_uniform(lower, upper, (n, 3), device=self.device)
      pos = pos + self._env.scene.env_origins[env_ids]

      # Sample orientation (yaw only, keep upright).
      yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
      quat = quat_from_euler_xyz(
        torch.zeros(n, device=self.device),  # roll
        torch.zeros(n, device=self.device),  # pitch
        yaw,
      )
      pose = torch.cat([pos, quat], dim=-1)

      velocity = torch.zeros(n, 6, device=self.device)

      self.object.write_root_link_pose_to_sim(pose, env_ids=env_ids)
      self.object.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    for batch in env_indices:
      target_pos = self.target_pos[batch].cpu().numpy()
      visualizer.add_sphere(
        center=target_pos,
        radius=0.03,
        color=self.cfg.viz.target_color,
        label=f"target_position_{batch}",
      )


class MultiCubeLiftingCommand(CommandTerm):
  """Selects one of N cubes as the target at each reset."""

  cfg: MultiCubeLiftingCommandCfg

  def __init__(
    self,
    cfg: MultiCubeLiftingCommandCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg, env)

    self.cubes = [env.scene[name] for name in cfg.entity_names]
    self._num_cubes = len(self.cubes)

    geom_ids = [c.indexing.geom_ids for c in self.cubes]
    max_geoms = max(g.shape[0] for g in geom_ids)
    self._padded_geom_ids = torch.full(
      (self._num_cubes, max_geoms),
      -999,
      device=self.device,
      dtype=geom_ids[0].dtype,
    )
    for i, g in enumerate(geom_ids):
      self._padded_geom_ids[i, : g.shape[0]] = g

    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)
    self.target_selection = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    self._env_arange = torch.arange(self.num_envs, device=self.device)
    self._cached_target_obj_pos = torch.zeros(self.num_envs, 3, device=self.device)

    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  @property
  def target_geom_ids(self) -> torch.Tensor:
    """Geom IDs of the target cube per env. Shape: (B, K)."""
    return self._padded_geom_ids[self.target_selection]

  def target_object_pos(self) -> torch.Tensor:
    """Position of the target cube per env.

    Cached per step — updated in _update_metrics which runs before rewards.
    """
    return self._cached_target_obj_pos

  def _update_metrics(self) -> None:
    all_pos = torch.stack([c.data.root_link_pos_w for c in self.cubes])
    self._cached_target_obj_pos = all_pos[self.target_selection, self._env_arange]
    obj_pos = self._cached_target_obj_pos
    error = torch.norm(self.target_pos - obj_pos, dim=-1)
    at_goal = (error < self.cfg.success_threshold).float()
    self.episode_success = torch.maximum(self.episode_success, at_goal)
    self.metrics["position_error"] = error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success

  def compute_success(self) -> torch.Tensor:
    return self.metrics["position_error"] < self.cfg.success_threshold

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.episode_success[env_ids] = 0.0

    self.target_selection[env_ids] = torch.randint(
      0, self._num_cubes, (n,), device=self.device
    )

    if self.cfg.difficulty == "fixed":
      target = torch.tensor([0.4, 0.0, 0.3], device=self.device).expand(n, 3)
      self.target_pos[env_ids] = target + self._env.scene.env_origins[env_ids]
    else:
      r = self.cfg.target_position_range
      lo = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
      hi = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
      target = sample_uniform(lo, hi, (n, 3), device=self.device)
      self.target_pos[env_ids] = target + self._env.scene.env_origins[env_ids]

    r = self.cfg.object_pose_range
    lo = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
    hi = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
    for cube in self.cubes:
      pos = sample_uniform(lo, hi, (n, 3), device=self.device)
      pos = pos + self._env.scene.env_origins[env_ids]
      yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
      quat = quat_from_euler_xyz(
        torch.zeros(n, device=self.device),
        torch.zeros(n, device=self.device),
        yaw,
      )
      pose = torch.cat([pos, quat], dim=-1)
      velocity = torch.zeros(n, 6, device=self.device)
      cube.write_root_link_pose_to_sim(pose, env_ids=env_ids)
      cube.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    for batch in env_indices:
      target_pos = self.target_pos[batch].cpu().numpy()
      visualizer.add_sphere(
        center=target_pos,
        radius=0.03,
        color=(1.0, 0.5, 0.0, 0.3),
        label=f"target_position_{batch}",
      )
      cube_pos = self.target_object_pos()[batch].cpu().numpy()
      marker = cube_pos.copy()
      marker[2] += 0.04
      visualizer.add_sphere(
        center=marker,
        radius=0.01,
        color=(1.0, 0.0, 0.0, 1.0),
        label=f"target_cube_marker_{batch}",
      )


@dataclass(kw_only=True)
class MultiCubeLiftingCommandCfg(CommandTermCfg):
  entity_names: tuple[str, ...] = ()
  success_threshold: float = 0.05
  difficulty: Literal["fixed", "dynamic"] = "fixed"

  @dataclass
  class TargetPositionRangeCfg:
    x: tuple[float, float] = (0.3, 0.5)
    y: tuple[float, float] = (-0.2, 0.2)
    z: tuple[float, float] = (0.2, 0.4)

  target_position_range: TargetPositionRangeCfg = field(
    default_factory=TargetPositionRangeCfg
  )

  @dataclass
  class ObjectPoseRangeCfg:
    x: tuple[float, float] = (0.25, 0.40)
    y: tuple[float, float] = (-0.15, 0.15)
    z: tuple[float, float] = (0.02, 0.05)
    yaw: tuple[float, float] = (-math.pi, math.pi)

  object_pose_range: ObjectPoseRangeCfg = field(default_factory=ObjectPoseRangeCfg)

  def build(self, env: ManagerBasedRlEnv) -> MultiCubeLiftingCommand:
    return MultiCubeLiftingCommand(self, env)


class ReorientationCommand(CommandTerm):
  """Goal orientation for in-hand cube reorientation.

  Samples a uniformly random goal orientation (full SO(3)) at episode reset and
  resamples once the cube has been held within ``success_threshold`` for
  ``success_hold_steps`` consecutive steps. The cube itself is reset by an event;
  this term only manages the goal. A translucent "ghost" cube is drawn above the
  hand at the goal orientation.
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

    self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["within_threshold"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["success_count"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["in_success_window"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["window_timer"] = torch.zeros(self.num_envs, device=self.device)

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
    self.in_success_window[env_ids] = False
    self.window_timer[env_ids] = 0
    self._sample_goal(env_ids)

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
    if self.marker is not None:
      pos = self.robot.data.root_link_pos_w + self._marker_offset
      pose = torch.cat([pos, self.goal_quat], dim=-1)
      self.marker.write_mocap_pose_to_sim(pose, env_ids=self._all_env_ids)


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


@dataclass(kw_only=True)
class LiftingCommandCfg(CommandTermCfg):
  entity_name: str
  success_threshold: float = 0.05
  difficulty: Literal["fixed", "dynamic"] = "fixed"

  @dataclass
  class TargetPositionRangeCfg:
    """Configuration for target position sampling in dynamic mode."""

    x: tuple[float, float] = (0.3, 0.5)
    y: tuple[float, float] = (-0.2, 0.2)
    z: tuple[float, float] = (0.2, 0.4)

  # Only used in dynamic mode.
  target_position_range: TargetPositionRangeCfg = field(
    default_factory=TargetPositionRangeCfg
  )

  @dataclass
  class ObjectPoseRangeCfg:
    """Configuration for object pose sampling when resampling commands."""

    x: tuple[float, float] = (0.3, 0.35)
    y: tuple[float, float] = (-0.1, 0.1)
    z: tuple[float, float] = (0.02, 0.05)
    yaw: tuple[float, float] = (-math.pi, math.pi)

  object_pose_range: ObjectPoseRangeCfg | None = field(
    default_factory=ObjectPoseRangeCfg
  )

  @dataclass
  class VizCfg:
    target_color: tuple[float, float, float, float] = (1.0, 0.5, 0.0, 0.3)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> LiftingCommand:
    return LiftingCommand(self, env)
