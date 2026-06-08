"""Actions that control actuator transmissions (e.g., joints, tendons, sites)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class BaseActionCfg(ActionTermCfg):
  """Configuration for actions that control actuator transmissions."""

  transmission_type: TransmissionType = TransmissionType.JOINT
  """Type of transmission to control."""

  actuator_names: tuple[str, ...] | list[str]
  """Actuator names to control."""

  scale: float | dict[str, float] = 1.0
  """Action scale. Float or dict mapping actuator names to scales."""

  offset: float | dict[str, float] = 0.0
  """Action offset. Float or dict mapping actuator names to offsets."""

  preserve_order: bool = False
  """Whether to preserve the order of actuator names."""


class BaseAction(ActionTerm):
  """Apply actions to actuator transmissions with scale/offset processing.

  Supports controlling different transmission types (e.g., joints, tendons,
  sites) with configurable affine transformations applied to raw actions.
  """

  cfg: BaseActionCfg
  _entity: Entity

  def __init__(self, cfg: BaseActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    # Find targets based on transmission type.
    target_ids, target_names = self._find_targets(cfg)
    self._target_ids = torch.tensor(target_ids, device=self.device, dtype=torch.long)
    self._target_names = target_names

    self._num_targets = len(target_ids)
    self._action_dim = len(target_ids)

    self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)

    if isinstance(cfg.scale, (float, int)):
      self._scale = float(cfg.scale)
    elif isinstance(cfg.scale, dict):
      self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
      index_list, _, value_list = resolve_matching_names_values(
        cfg.scale, self._target_names
      )
      self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
    else:
      raise ValueError(
        f"Unsupported scale type: {type(cfg.scale)}. "
        f"Supported types are float and dict."
      )

    if isinstance(cfg.offset, (float, int)):
      self._offset = float(cfg.offset)
    elif isinstance(cfg.offset, dict):
      self._offset = torch.zeros_like(self._raw_actions)
      index_list, _, value_list = resolve_matching_names_values(
        cfg.offset, self._target_names
      )
      self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
    else:
      raise ValueError(
        f"Unsupported offset type: {type(cfg.offset)}. "
        f"Supported types are float and dict."
      )

    if cfg.clip is not None:
      self._clip = torch.tensor(
        [[-float("inf"), float("inf")]], device=self.device
      ).repeat(self.num_envs, self.action_dim, 1)
      index_list, _, value_list = resolve_matching_names_values(
        cfg.clip, self._target_names
      )
      self._clip[:, index_list] = torch.tensor(value_list, device=self.device)

  def _find_targets(self, cfg: BaseActionCfg) -> tuple[list[int], list[str]]:
    """Find target IDs and names based on transmission type.

    Args:
      cfg: Action configuration.

    Returns:
      Tuple of (target_ids, target_names).
    """
    if cfg.transmission_type == TransmissionType.JOINT:
      return self._entity.find_joints_by_actuator_names(cfg.actuator_names)
    elif cfg.transmission_type == TransmissionType.TENDON:
      return self._entity.find_tendons(
        cfg.actuator_names, preserve_order=cfg.preserve_order
      )
    elif cfg.transmission_type == TransmissionType.SITE:
      return self._entity.find_sites(
        cfg.actuator_names, preserve_order=cfg.preserve_order
      )
    else:
      raise ValueError(f"Unknown transmission type: {cfg.transmission_type}")

  # Properties.

  @property
  def scale(self) -> torch.Tensor | float:
    """Action scale."""
    return self._scale

  @property
  def offset(self) -> torch.Tensor | float:
    """Action offset."""
    return self._offset

  @property
  def raw_action(self) -> torch.Tensor:
    """Raw actions (before scale/offset)."""
    return self._raw_actions

  @property
  def action_dim(self) -> int:
    """Dimension of the action space."""
    return self._action_dim

  @property
  def target_ids(self) -> torch.Tensor:
    """Target IDs for the controlled transmission."""
    return self._target_ids

  @property
  def target_names(self) -> list[str]:
    """Target names for the controlled transmission."""
    return self._target_names

  def process_actions(self, actions: torch.Tensor):
    """Process raw actions by applying scale, offset, and optional clip."""
    self._raw_actions[:] = actions
    self._processed_actions = self._raw_actions * self._scale + self._offset
    if self.cfg.clip is not None:
      self._processed_actions = torch.clamp(
        self._processed_actions,
        min=self._clip[:, :, 0],
        max=self._clip[:, :, 1],
      )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset raw actions to zero for specified environments."""
    self._raw_actions[env_ids] = 0.0


##
# Joint actions.
##


@dataclass(kw_only=True)
class JointPositionActionCfg(BaseActionCfg):
  """Configuration for joint position control."""

  use_default_offset: bool = True

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> JointPositionAction:
    return JointPositionAction(self, env)


@dataclass(kw_only=True)
class JointVelocityActionCfg(BaseActionCfg):
  """Configuration for joint velocity control."""

  use_default_offset: bool = True

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> JointVelocityAction:
    return JointVelocityAction(self, env)


@dataclass(kw_only=True)
class JointEffortActionCfg(BaseActionCfg):
  """Configuration for joint effort (torque) control."""

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> JointEffortAction:
    return JointEffortAction(self, env)


class JointPositionAction(BaseAction):
  """Control joints via position targets."""

  def __init__(self, cfg: JointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    if cfg.use_default_offset:
      self._offset = self._entity.data.default_joint_pos[:, self._target_ids].clone()

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = self._processed_actions - encoder_bias
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


@dataclass(kw_only=True)
class RelativeJointPositionActionCfg(BaseActionCfg):
  """Configuration for joint position control relative to current positions.

  target = current_joint_pos + action * scale

  The ``offset`` field inherited from ``BaseActionCfg`` is not supported and
  must remain at its default of ``0.0``. The ``clip`` field is supported and
  clamps the delta (``action * scale``) before it is added to the current
  position.
  """

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT
    if self.offset != 0.0:
      raise ValueError(
        "RelativeJointPositionActionCfg does not support 'offset'. "
        "The target is current_pos + action * scale; a fixed offset has no meaning."
      )

  def build(self, env: ManagerBasedRlEnv) -> RelativeJointPositionAction:
    return RelativeJointPositionAction(self, env)


class RelativeJointPositionAction(BaseAction):
  """Control joints via position targets relative to current positions."""

  def apply_actions(self) -> None:
    current_pos = self._entity.data.joint_pos[:, self._target_ids]
    target = current_pos + self._processed_actions
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


@dataclass(kw_only=True)
class JointPositionOffsetEMAActionCfg(JointPositionActionCfg):
  """Joint position control anchored at the default joint pose, with EMA smoothing.

  Processing pipeline (per control step):

    raw_target = default_joint_pos + clip(action, ±1) * scale
    raw_target = clamp(raw_target, soft_lower_limit, soft_upper_limit)
    target     = ema_alpha * raw_target + (1 - ema_alpha) * prev_target
    if t < warmup_time_s: target = default_joint_pos

  The default pose acts as a stable anchor (each action is a bounded perturbation
  of a known-good configuration, not an integrator of prior commands), the EMA
  filters out high-frequency policy jitter, and the warmup hold gives the policy
  a quiet boot period before it starts driving the joints. Matches the action
  processing used in wuji-mjlab for in-hand manipulation.

  ``use_default_offset`` is forced True (the offset *is* the default pose) and
  must not be overridden.
  """

  ema_alpha: float = 0.5
  """EMA blend factor for target smoothing. 1.0 disables smoothing."""

  warmup_time_s: float = 0.0
  """Episode time during which the target is held at ``default_joint_pos``."""

  def __post_init__(self):
    super().__post_init__()
    self.use_default_offset = True

  def build(self, env: ManagerBasedRlEnv) -> JointPositionOffsetEMAAction:
    return JointPositionOffsetEMAAction(self, env)


class JointPositionOffsetEMAAction(JointPositionAction):
  """See ``JointPositionOffsetEMAActionCfg``."""

  def __init__(self, cfg: JointPositionOffsetEMAActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    # use_default_offset=True is enforced in cfg.__post_init__, so
    # JointPositionAction.__init__ replaces self._offset with the per-env
    # default-joint-pos tensor. Re-bind to a typed attribute to make the
    # tensor-ness explicit for downstream tensor ops (clone, fancy-indexing).
    assert isinstance(self._offset, torch.Tensor)
    self._offset: torch.Tensor = self._offset

    # Soft joint limits for hard clamping after the offset-and-scale.
    soft = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
    self._lower_limit = soft[..., 0]
    self._upper_limit = soft[..., 1]

    self._prev_target = self._offset.clone()
    self._ema_alpha = float(cfg.ema_alpha)
    self._warmup_steps = int(round(cfg.warmup_time_s / float(env.step_dt)))

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    clamped = torch.clamp(actions, -1.0, 1.0)
    raw_target = self._offset + clamped * self._scale
    raw_target = torch.clamp(raw_target, self._lower_limit, self._upper_limit)
    smoothed = (
      self._ema_alpha * raw_target + (1.0 - self._ema_alpha) * self._prev_target
    )
    if self._warmup_steps > 0:
      in_warmup = (self._env.episode_length_buf < self._warmup_steps).unsqueeze(-1)
      smoothed = torch.where(in_warmup, self._offset, smoothed)
    self._processed_actions = smoothed
    self._prev_target = smoothed.clone()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    # Reset EMA state so the first post-reset target starts from the default pose.
    if env_ids is None or isinstance(env_ids, slice):
      self._prev_target = self._offset.clone()
    else:
      self._prev_target[env_ids] = self._offset[env_ids]


class JointVelocityAction(BaseAction):
  """Control joints via velocity targets."""

  def __init__(self, cfg: JointVelocityActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    if cfg.use_default_offset:
      self._offset = self._entity.data.default_joint_vel[:, self._target_ids].clone()

  def apply_actions(self) -> None:
    self._entity.set_joint_velocity_target(
      self._processed_actions, joint_ids=self._target_ids
    )


class JointEffortAction(BaseAction):
  """Control joints via effort (torque) targets."""

  def __init__(self, cfg: JointEffortActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

  def apply_actions(self) -> None:
    self._entity.set_joint_effort_target(
      self._processed_actions, joint_ids=self._target_ids
    )


##
# Tendon actions.
##


@dataclass(kw_only=True)
class TendonLengthActionCfg(BaseActionCfg):
  """Configuration for tendon length control."""

  def __post_init__(self):
    self.transmission_type = TransmissionType.TENDON

  def build(self, env: ManagerBasedRlEnv) -> TendonLengthAction:
    return TendonLengthAction(self, env)


@dataclass(kw_only=True)
class TendonVelocityActionCfg(BaseActionCfg):
  """Configuration for tendon velocity control."""

  def __post_init__(self):
    self.transmission_type = TransmissionType.TENDON

  def build(self, env: ManagerBasedRlEnv) -> TendonVelocityAction:
    return TendonVelocityAction(self, env)


@dataclass(kw_only=True)
class TendonEffortActionCfg(BaseActionCfg):
  """Configuration for tendon effort control."""

  def __post_init__(self):
    self.transmission_type = TransmissionType.TENDON

  def build(self, env: ManagerBasedRlEnv) -> TendonEffortAction:
    return TendonEffortAction(self, env)


class TendonLengthAction(BaseAction):
  """Control tendons via length targets."""

  def __init__(self, cfg: TendonLengthActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

  def apply_actions(self) -> None:
    self._entity.set_tendon_len_target(
      self._processed_actions, tendon_ids=self._target_ids
    )


class TendonVelocityAction(BaseAction):
  """Control tendons via velocity targets."""

  def __init__(self, cfg: TendonVelocityActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

  def apply_actions(self) -> None:
    self._entity.set_tendon_vel_target(
      self._processed_actions, tendon_ids=self._target_ids
    )


class TendonEffortAction(BaseAction):
  """Control tendons via effort targets."""

  def __init__(self, cfg: TendonEffortActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

  def apply_actions(self) -> None:
    self._entity.set_tendon_effort_target(
      self._processed_actions, tendon_ids=self._target_ids
    )


##
# Site actions.
##


@dataclass(kw_only=True)
class SiteEffortActionCfg(BaseActionCfg):
  """Configuration for site effort control."""

  def __post_init__(self):
    self.transmission_type = TransmissionType.SITE

  def build(self, env: ManagerBasedRlEnv) -> SiteEffortAction:
    return SiteEffortAction(self, env)


class SiteEffortAction(BaseAction):
  """Control sites via effort targets."""

  def __init__(self, cfg: SiteEffortActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

  def apply_actions(self) -> None:
    self._entity.set_site_effort_target(
      self._processed_actions, site_ids=self._target_ids
    )
