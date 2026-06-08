"""Tests for cube-reorient reward functions, particularly the shaping terms."""

from __future__ import annotations

import math
from typing import Iterator

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manipulation.mdp.commands import ReorientationCommand
from mjlab.tasks.manipulation.mdp.rewards import (
  cube_orientation_success_bonus,
  cube_orientation_tolerance,
  cube_rotation_toward_goal,
)
from mjlab.tasks.registry import load_env_cfg
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


# --- Rotation-toward-goal reward -------------------------------------------


def test_rotate_toward_goal_zero_when_static(env: ManagerBasedRlEnv) -> None:
  """Cube not rotating -> reward is 0."""
  cmd = env.command_manager.get_term("goal")
  assert isinstance(cmd, ReorientationCommand)
  # Cube at identity, goal at 90 deg around z (some non-zero error).
  axis = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
  cmd.goal_quat[:] = quat_from_angle_axis(
    torch.tensor([math.pi / 2, math.pi / 2]), axis.to(cmd.device)
  )
  q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
  zero_omega = torch.zeros(2, 3)
  _force_cube_state(env, pos_z=[0.14, 0.14], quat=q_id, ang_vel=zero_omega)
  r = cube_rotation_toward_goal(env, "goal")
  assert torch.allclose(r, torch.zeros_like(r), atol=1e-5)


def test_rotate_toward_goal_positive_when_aligned(env: ManagerBasedRlEnv) -> None:
  """Cube rotating toward the goal -> positive reward equal to |omega|."""
  cmd = env.command_manager.get_term("goal")
  assert isinstance(cmd, ReorientationCommand)
  device = cmd.device
  # Goal is +90 deg around z; cube at identity. Needed rotation is +z * (pi/2).
  axis = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device=device)
  cmd.goal_quat[:] = quat_from_angle_axis(
    torch.tensor([math.pi / 2, math.pi / 2], device=device), axis
  )
  q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
  # Spin cube around +z at 1.0 rad/s (perfectly aligned with needed direction).
  omega = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
  _force_cube_state(env, pos_z=[0.14, 0.14], quat=q_id, ang_vel=omega)
  r = cube_rotation_toward_goal(env, "goal")
  # Alignment should equal |omega| since direction is perfect.
  assert torch.allclose(r, torch.tensor([1.0, 2.0], device=device), atol=1e-4)


def test_rotate_toward_goal_zero_when_anti_aligned(env: ManagerBasedRlEnv) -> None:
  """Cube rotating AWAY from the goal -> reward clamps to 0 (no penalty)."""
  cmd = env.command_manager.get_term("goal")
  assert isinstance(cmd, ReorientationCommand)
  device = cmd.device
  axis = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device=device)
  cmd.goal_quat[:] = quat_from_angle_axis(
    torch.tensor([math.pi / 2, math.pi / 2], device=device), axis
  )
  q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
  # Spin -z (opposite of needed +z).
  omega = torch.tensor([[0.0, 0.0, -1.5], [0.0, 0.0, -3.0]])
  _force_cube_state(env, pos_z=[0.14, 0.14], quat=q_id, ang_vel=omega)
  r = cube_rotation_toward_goal(env, "goal")
  assert torch.allclose(r, torch.zeros_like(r), atol=1e-5)


def test_rotate_toward_goal_gate_zeros_below_height(env: ManagerBasedRlEnv) -> None:
  """When cube is below gate_min_height, reward is zero even if aligned."""
  cmd = env.command_manager.get_term("goal")
  assert isinstance(cmd, ReorientationCommand)
  device = cmd.device
  axis = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device=device)
  cmd.goal_quat[:] = quat_from_angle_axis(
    torch.tensor([math.pi / 2, math.pi / 2], device=device), axis
  )
  q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
  omega = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
  # Env 0 cradled, env 1 dropped below gate.
  _force_cube_state(env, pos_z=[0.14, 0.05], quat=q_id, ang_vel=omega)
  r = cube_rotation_toward_goal(
    env, "goal", gate_object_name="cube", gate_min_height=0.10
  )
  assert r[0].item() > 0  # cradled, aligned -> positive
  assert r[1].item() == 0.0  # dropped, gated to zero


def test_rotate_toward_goal_handles_zero_error(env: ManagerBasedRlEnv) -> None:
  """When cube is already at goal (needed rotation = 0), reward is well-defined."""
  cmd = env.command_manager.get_term("goal")
  assert isinstance(cmd, ReorientationCommand)
  # cube == goal -> axis_angle is the zero vector, direction is degenerate.
  cmd.goal_quat[:] = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=cmd.device
  )
  q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
  # Some non-zero omega.
  omega = torch.tensor([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]])
  _force_cube_state(env, pos_z=[0.14, 0.14], quat=q_id, ang_vel=omega)
  r = cube_rotation_toward_goal(env, "goal")
  assert torch.isfinite(r).all(), f"reward not finite at zero-error: {r}"


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


def test_success_bonus_gate_zeros_below_height(env: ManagerBasedRlEnv) -> None:
  """cube_orientation_success_bonus is gated by cube_held."""
  cmd = env.command_manager.get_term("goal")
  assert isinstance(cmd, ReorientationCommand)
  cmd.at_goal[:] = torch.tensor([1.0, 1.0], device=cmd.device)  # both succeeded
  q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
  _force_cube_state(env, pos_z=[0.14, 0.05], quat=q_id, ang_vel=torch.zeros(2, 3))
  r = cube_orientation_success_bonus(
    env, "goal", gate_object_name="cube", gate_min_height=0.10
  )
  assert r[0].item() == 1.0  # cradled success -> bonus
  assert r[1].item() == 0.0  # dropped success -> gated to zero
