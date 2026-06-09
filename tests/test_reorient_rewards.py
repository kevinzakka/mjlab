"""Tests for cube-reorient reward functions, particularly the shaping terms."""

from __future__ import annotations

from typing import Iterator

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.reorient.mdp.commands import ReorientationCommand
from mjlab.tasks.reorient.mdp.rewards import (
  cube_orientation_tolerance,
)
from mjlab.utils.lab_api.math import quat_from_angle_axis

TASK_ID = "Mjlab-Reorient-Cube-Sharpa"


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture(scope="module")
def env() -> Iterator[ManagerBasedRlEnv]:
  """Build the reorient env once for all reward tests."""
  cfg = load_env_cfg(TASK_ID)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  yield env
  env.close()


def _force_cube_state(
  env: ManagerBasedRlEnv,
  pos_z: list[float],
  quat: torch.Tensor,
  ang_vel: torch.Tensor,
) -> None:
  """Force the cube into a known root pose + angular velocity (writes to sim).

  ``quat`` shape (n, 4) wxyz, ``ang_vel`` shape (n, 3) in world frame, ``pos_z``
  the world-frame z height per env.
  """
  n = env.scene.num_envs
  device = env.device
  pose = torch.zeros(n, 7, device=device)
  pose[:, 2] = torch.tensor(pos_z, device=device)
  pose[:, 3:7] = quat.to(device)
  vel = torch.zeros(n, 6, device=device)
  vel[:, 3:6] = ang_vel.to(device)
  cube = env.scene["cube"]
  ids = torch.arange(n, device=device)
  cube.data.write_root_pose(pose, env_ids=ids)
  cube.data.write_root_velocity(vel, env_ids=ids)
  env.sim.forward()


# --- Gate behavior on the other task rewards --------------------------------


def test_tolerance_gate_zeros_below_height(env: ManagerBasedRlEnv) -> None:
  """cube_orientation_tolerance with gate zeroes out when cube is dropped."""
  cmd = env.command_manager.get_term("goal")
  assert isinstance(cmd, ReorientationCommand)
  # Force a specific (non-zero) orientation error so tolerance reward is positive
  # without the gate.
  cmd.goal_quat[:] = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=cmd.device
  )
  axis = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
  q_err = quat_from_angle_axis(torch.tensor([0.5, 0.5]), axis)
  _force_cube_state(env, pos_z=[0.14, 0.05], quat=q_err, ang_vel=torch.zeros(2, 3))
  r = cube_orientation_tolerance(
    env, "goal", gate_object_name="cube", gate_min_height=0.10
  )
  assert r[0].item() > 0  # cradled -> positive
  assert r[1].item() == 0.0  # dropped -> zero
