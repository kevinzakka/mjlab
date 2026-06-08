"""State-machine tests for ``ReorientationCommand``.

Drives the command's internal state directly (bypassing the cube-quat -> error
computation in ``_update_metrics``) so the APPROACHING -> SUCCESS_WINDOW ->
new-goal transitions can be exercised step-by-step with controlled inputs.
"""

from __future__ import annotations

from typing import Iterator

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manipulation.mdp import ReorientationCommandCfg
from mjlab.tasks.manipulation.mdp.commands import ReorientationCommand
from mjlab.tasks.registry import load_env_cfg

TASK_ID = "Mjlab-Reorient-Cube-Sharpa"


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture(scope="module")
def env_and_command() -> Iterator[tuple[ManagerBasedRlEnv, ReorientationCommand]]:
  """Build the reorient env once and hand back (env, command) for all tests.

  Module-scoped because env construction is expensive (compiles the whole
  warp-mujoco graph). Tests don't mutate env state -- they only poke the
  command's tensors -- so it's safe to share.
  """
  cfg = load_env_cfg(TASK_ID)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  cmd = env.command_manager.get_term("goal")
  assert isinstance(cmd, ReorientationCommand)
  yield env, cmd
  env.close()


@pytest.fixture
def command(env_and_command) -> ReorientationCommand:
  """Reset the command's state before each test for independence."""
  _, cmd = env_and_command
  reset_all = torch.arange(cmd.num_envs, device=cmd.device)
  cmd._resample_command(reset_all)
  # Counters cleared (in addition to whatever _resample_command resets).
  cmd.hold_counter[:] = 0
  cmd.dwell_counter[:] = 0
  cmd.in_dwell[:] = False
  cmd.at_goal[:] = 0.0
  cmd.within_threshold[:] = 0.0
  cmd.orientation_error[:] = 0.0
  cmd.success_count[:] = 0.0
  cmd.episode_success[:] = 0.0
  return cmd


# --- State-machine driver ---------------------------------------------------


def _step(cmd: ReorientationCommand, within: list[bool]) -> None:
  """Drive one control step with a controlled within-threshold signal per env.

  Mirrors the state-update math inside ``_update_metrics`` but skips the
  cube-pose -> orientation-error computation so the test can directly inject
  the within-threshold signal it wants.
  """
  n = cmd.num_envs
  assert len(within) == n
  cmd.within_threshold = torch.tensor([float(v) for v in within], device=cmd.device)
  cmd.hold_counter = torch.where(
    cmd.within_threshold.bool(),
    cmd.hold_counter + 1,
    torch.zeros_like(cmd.hold_counter),
  )
  just_completed = cmd.hold_counter >= cmd.cfg.success_hold_steps
  cmd.at_goal = (just_completed & ~cmd.in_dwell).float()
  cmd._update_command()


# --- Phase transition tests -------------------------------------------------


def test_initial_state(command: ReorientationCommand) -> None:
  """Fresh state: not in dwell, all counters zero, goal is valid."""
  assert not command.in_dwell.any()
  assert (command.hold_counter == 0).all()
  assert (command.dwell_counter == 0).all()
  assert (command.success_count == 0).all()
  # goal_quat is a valid unit quaternion per env.
  norms = command.goal_quat.norm(dim=-1)
  assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_hold_counter_resets_on_exit(command: ReorientationCommand) -> None:
  """A within streak interrupted resets hold_counter to 0."""
  N = command.cfg.success_hold_steps
  assert N >= 2
  _step(command, within=[True, True])
  assert command.hold_counter.tolist() == [1, 1]
  _step(command, within=[True, False])
  assert command.hold_counter.tolist() == [2, 0]
  _step(command, within=[False, True])
  assert command.hold_counter.tolist() == [0, 1]


def test_hold_completion_pulses_at_goal_once(command: ReorientationCommand) -> None:
  """at_goal fires exactly on the step hold_counter reaches success_hold_steps."""
  N = command.cfg.success_hold_steps
  # Drive env 0 to completion; leave env 1 always-out.
  for _ in range(N - 1):
    _step(command, within=[True, False])
    assert command.at_goal[0].item() == 0.0
  # On the N-th in-window step, env 0 completes the hold.
  _step(command, within=[True, False])
  assert command.at_goal[0].item() == 1.0
  assert command.at_goal[1].item() == 0.0
  assert command.in_dwell[0].item()
  assert not command.in_dwell[1].item()
  # success_count increments by exactly 1.
  assert command.success_count[0].item() == 1.0
  assert command.success_count[1].item() == 0.0


def test_at_goal_does_not_re_pulse_during_dwell(command: ReorientationCommand) -> None:
  """In SUCCESS_WINDOW with cube still within threshold, at_goal stays 0."""
  N = command.cfg.success_hold_steps
  # Reach hold completion on env 0.
  for _ in range(N):
    _step(command, within=[True, True])
  assert command.at_goal[0].item() == 1.0
  assert command.in_dwell[0].item()
  # Continue with cube within threshold for several steps -- at_goal stays 0.
  for _ in range(5):
    _step(command, within=[True, True])
    assert command.at_goal[0].item() == 0.0
  assert command.success_count[0].item() == 1.0  # still just one success


def test_dwell_advances_and_exits(command: ReorientationCommand) -> None:
  """dwell_counter increments while in_dwell, exits on dwell_steps, samples new goal."""
  N = command.cfg.success_hold_steps
  D = command.cfg.success_dwell_steps
  # Complete a hold on env 0. After the Nth call: in_dwell=True, dwell_counter=1
  # (the entry call itself counts as the first dwell step).
  for _ in range(N):
    _step(command, within=[True, True])
  assert command.in_dwell[0].item()
  assert command.dwell_counter[0].item() == 1
  goal_before = command.goal_quat[0].clone()

  # The dwell phase lasts D total _update_command calls (entry + D-1 more).
  # The next D-2 calls leave us still in dwell with counter incrementing.
  for i in range(D - 2):
    _step(command, within=[False, False])
    assert command.in_dwell[0].item(), f"left dwell early at step {i}"
    assert command.dwell_counter[0].item() == i + 2  # 2, 3, ..., D-1
  # One more call brings counter from D-1 to D, which triggers exit:
  # new goal sampled, counters reset, in_dwell False.
  _step(command, within=[False, False])
  assert not command.in_dwell[0].item()
  assert command.hold_counter[0].item() == 0
  assert command.dwell_counter[0].item() == 0
  goal_after = command.goal_quat[0]
  assert not torch.allclose(goal_before, goal_after), (
    "Goal should have been re-sampled on dwell exit."
  )


def test_full_cycle_chains_successes(command: ReorientationCommand) -> None:
  """Drive two full cycles on env 0; success_count should be 2."""
  N = command.cfg.success_hold_steps
  D = command.cfg.success_dwell_steps
  for _ in range(2):
    # Reach + hold.
    for _ in range(N):
      _step(command, within=[True, False])
    # Complete the dwell (cube state during dwell doesn't matter for the counter).
    for _ in range(D):
      _step(command, within=[True, False])
  assert command.success_count[0].item() == 2.0
  assert command.success_count[1].item() == 0.0
  assert not command.in_dwell[0].item()  # back to APPROACHING after second cycle


def test_per_env_independence(command: ReorientationCommand) -> None:
  """Two envs progress independently through the state machine."""
  N = command.cfg.success_hold_steps
  # Step 1..N-1: only env 0 accumulates hold.
  for _ in range(N - 1):
    _step(command, within=[True, False])
  assert command.hold_counter[0].item() == N - 1
  assert command.hold_counter[1].item() == 0

  # Step N: env 0 completes the hold (enters dwell); env 1 unchanged.
  _step(command, within=[True, False])
  assert command.in_dwell[0].item() and not command.in_dwell[1].item()

  # During env 0's dwell, env 1 starts its own hold.
  for _ in range(N):
    _step(command, within=[False, True])
  assert command.in_dwell[0].item()  # still in dwell
  assert command.in_dwell[1].item()  # env 1 just entered dwell
  assert command.success_count.tolist() == [1.0, 1.0]


def test_resample_command_clears_episode_state(
  command: ReorientationCommand,
) -> None:
  """Episode reset zeros per-episode trackers and samples a fresh goal."""
  N = command.cfg.success_hold_steps
  # Build up some state on env 0.
  for _ in range(N + 3):
    _step(command, within=[True, False])
  assert command.success_count[0].item() == 1.0
  assert command.in_dwell[0].item()
  goal_before = command.goal_quat[0].clone()

  # Reset env 0 only.
  command._resample_command(torch.tensor([0], device=command.device))
  assert command.success_count[0].item() == 0.0
  assert command.episode_success[0].item() == 0.0
  assert command.hold_counter[0].item() == 0
  assert command.dwell_counter[0].item() == 0
  assert not command.in_dwell[0].item()
  goal_after = command.goal_quat[0]
  # New random orientation almost certainly differs from the previous one.
  assert not torch.allclose(goal_before, goal_after, atol=1e-3)


def test_compute_success_matches_at_goal(command: ReorientationCommand) -> None:
  """compute_success() returns at_goal.bool() exactly."""
  N = command.cfg.success_hold_steps
  for _ in range(N):
    _step(command, within=[True, True])
  assert torch.equal(command.compute_success(), command.at_goal.bool())


def test_config_has_required_fields() -> None:
  """The cfg dataclass exposes the new dwell knob and not the removed velocity ones."""
  cfg = ReorientationCommandCfg(
    entity_name="cube",
    resampling_time_range=(1.0e6, 1.0e6),
  )
  assert hasattr(cfg, "success_hold_steps")
  assert hasattr(cfg, "success_dwell_steps")
  assert not hasattr(cfg, "goal_ang_vel_max")
  assert not hasattr(cfg, "goal_ang_vel_decay")
