"""Domain randomization functions for simulation-option (``model.opt``) fields."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import requires_model_fields

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@requires_model_fields("opt.gravity")
def gravity_magnitude(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  mag_range: tuple[float, float] = (1.0, 9.81),
  direction: tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> None:
  """Randomize the gravity magnitude per environment along a fixed direction.

  Samples ``|g|`` uniformly in ``mag_range`` for each environment and points gravity at
  ``|g| * normalize(direction)``. The direction is held fixed (default world ``-z``), so
  only the strength of gravity varies across environments, not where it points.

  Requires per-world gravity, so ``opt.gravity`` is expanded to ``(num_envs, 3)`` (see
  :func:`requires_model_fields`); with a single environment the field stays shared.

  As a reset-mode event this gives a static per-env spread of gravity strengths; pairing
  it with a curriculum that advances ``mag_range[1]`` turns it into a difficulty schedule
  while keeping the lighter end of the range present.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = len(env_ids)
  device = env.device

  mag = torch.rand(n, device=device) * (mag_range[1] - mag_range[0]) + mag_range[0]
  d = torch.tensor(direction, device=device, dtype=torch.float32)
  d = d / d.norm().clamp_min(1e-9)
  gravity = mag.unsqueeze(-1) * d  # (n, 3)

  env.sim.model.opt.gravity[env_ids] = gravity


@requires_model_fields("opt.gravity")
def gravity_direction(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  tilt_range: tuple[float, float] = (0.0, math.pi),
  magnitude: float = 9.81,
  axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> None:
  """Randomize the gravity direction per environment by tilting it off world +z.

  Samples a tilt angle ``theta`` uniformly in ``tilt_range`` for each environment and
  points that env's gravity at ``magnitude * R(axis, theta) @ (0, 0, 1)``:

  * ``theta = 0`` -> gravity points along world **+z** (up). For an inverted (palm-down)
    hand this presses the cube up into the palm, i.e. the easy, palm-up-equivalent case.
  * ``theta = pi`` -> gravity points along world **-z** (down): ordinary gravity, the
    full inverted task where the cube must be gripped against its own weight.
  * intermediate angles tilt gravity through the horizontal.

  This requires per-world gravity, so the ``opt.gravity`` model field is expanded to
  ``(num_envs, 3)`` (see :func:`requires_model_fields`). With a single environment the
  field is left shared and every env reads the same value.

  Pair this reset-mode event with a curriculum that advances ``tilt_range[1]`` from ``0``
  to ``pi`` over training, so envs start with gravity stabilizing the grasp and gradually
  span the full range up to fully inverted (a per-env distribution of difficulty).
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = len(env_ids)
  device = env.device

  theta = torch.rand(n, device=device) * (tilt_range[1] - tilt_range[0]) + tilt_range[0]
  k = torch.tensor(axis, device=device, dtype=torch.float32)
  k = k / k.norm().clamp_min(1e-9)
  v = torch.tensor([0.0, 0.0, 1.0], device=device)

  # Rodrigues' rotation of v about unit axis k by theta (per env).
  cos = torch.cos(theta).unsqueeze(-1)  # (n, 1)
  sin = torch.sin(theta).unsqueeze(-1)
  k_cross_v = torch.linalg.cross(k.expand(n, 3), v.expand(n, 3))
  k_dot_v = (k * v).sum()  # scalar (k, v fixed)
  rotated = v * cos + k_cross_v * sin + k * (k_dot_v * (1.0 - cos))
  gravity = magnitude * rotated  # (n, 3)

  env.sim.model.opt.gravity[env_ids] = gravity
