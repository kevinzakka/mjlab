"""Curriculum terms for the in-hand reorientation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.tasks.reorient.mdp.commands import ReorientationCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ("GravityCurriculum",)


class GravityCurriculum(ManagerTermBase):
  """Population performance-gated gravity-magnitude curriculum.

  A single shared gravity ceiling for the whole population (not a per-env level). At each
  reset the resetting envs sample ``|g|`` uniformly in ``[floor, ceiling]`` along a fixed
  direction (default world ``-z``, i.e. full inverted) and write it per env -- a narrow
  band early, widening only as the ceiling rises. Keeping every env in one observable band
  matters because the policy does not observe gravity: a per-env curriculum that let envs
  diverge produced a wide unobservable spread that the policy and critic could not handle.

  The ceiling advances only when the population can do the *task* at the current gravity --
  not merely hold the cube. Each reset reads the just-finished episode's success (did the
  env complete a reorientation, ``command.episode_success``), tracks the population success
  fraction as an EMA, and every ``eval_interval`` environment steps steps the ceiling up by
  ``step`` if that fraction is at least ``advance_rate`` or down if it is at most
  ``retreat_rate`` (clamped to ``[floor, ceiling_max]``). Gating on success rather than on
  "didn't drop" keeps gravity low while reorientation develops, so the hard skill matures
  at reduced weight instead of gravity racing to full the moment the cube is merely held.

  Wired as a reset-mode event so it can declare the per-world ``opt.gravity`` field for
  expansion and write each env's gravity at reset. Logs ``Curriculum/gravity_ceiling`` and
  ``Curriculum/gravity_success_ema``.
  """

  model_fields = ("opt.gravity",)

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    p = cfg.params
    self._floor = float(p.get("floor", 1.0))
    self._ceiling_max = float(p.get("ceiling_max", 9.81))
    self._step = float(p.get("step", 0.5))
    self._advance_rate = float(p.get("advance_rate", 0.5))
    self._retreat_rate = float(p.get("retreat_rate", 0.2))
    self._eval_interval = int(p.get("eval_interval", 24 * 30))
    self._ema_alpha = float(p.get("ema_alpha", 0.1))
    self._command_name = str(p.get("command_name", "goal"))
    direction = p.get("direction", (0.0, 0.0, -1.0))
    d = torch.tensor(direction, device=self.device, dtype=torch.float32)
    self._direction = d / d.norm().clamp_min(1e-9)
    # Shared population state, persisted across resets.
    self._ceiling = self._floor
    self._success_ema = 0.0  # pessimistic: must demonstrate the task before advancing
    self._last_eval_step = 0

  def __call__(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **kwargs: object
  ) -> None:
    del kwargs  # Params are consumed in __init__.
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)

    # Population task-progress signal: fraction of just-finished episodes that completed a
    # reorientation. command.episode_success is not zeroed until later in the reset
    # sequence (the command manager resets after the reset events fire).
    command = env.command_manager.get_term(self._command_name)
    assert isinstance(command, ReorientationCommand)
    batch_rate = command.episode_success[env_ids].mean().item()
    self._success_ema = (
      self._ema_alpha * batch_rate + (1.0 - self._ema_alpha) * self._success_ema
    )

    # Rate-limited, performance-gated ceiling update.
    if env.common_step_counter - self._last_eval_step >= self._eval_interval:
      self._last_eval_step = env.common_step_counter
      if self._success_ema >= self._advance_rate:
        self._ceiling = min(self._ceiling + self._step, self._ceiling_max)
      elif self._success_ema <= self._retreat_rate:
        self._ceiling = max(self._ceiling - self._step, self._floor)

    # Sample each resetting env's gravity uniformly in the shared band and write it.
    n = len(env_ids)
    mag = (
      torch.rand(n, device=self.device) * (self._ceiling - self._floor) + self._floor
    )
    env.sim.model.opt.gravity[env_ids] = mag.unsqueeze(-1) * self._direction

    log = env.extras.get("log")
    if isinstance(log, dict):
      log["Curriculum/gravity_ceiling"] = torch.tensor(self._ceiling)
      log["Curriculum/gravity_success_ema"] = torch.tensor(self._success_ema)
