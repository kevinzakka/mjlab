"""Tests for the Sharpa in-hand cube reorientation task.

Covers the task config, the goal command's success/hold/window state machine, the
goal-relative observations, and the reward terms. The robot asset itself is tested
separately in ``test_sharpa_constants.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.reorient.mdp import ReorientationCommandCfg
from mjlab.tasks.reorient.mdp.commands import ReorientationCommand
from mjlab.tasks.reorient.mdp.observations import object_to_goal_orientation_6d
from mjlab.tasks.reorient.mdp.rewards import (
  NormalizedJointTorquePenalty,
  cube_orientation_tolerance,
  sustained_hold,
)
from mjlab.utils.lab_api.math import quat_box_plus, quat_from_angle_axis, quat_mul

TASK_ID = "Mjlab-Reorient-Cube-Sharpa"
PALM_SITE = "wrist_site"


# --- Config (no env build) --------------------------------------------------


def test_task_registered() -> None:
  assert TASK_ID in list_tasks()


def test_sim_options() -> None:
  """Simulator options (owned by SimulationCfg) match the contact-rich setup.

  Elliptic cone + impratio 10: the soft-finger grasp needs a hard friction
  constraint to suppress tangential creep, and impratio is only principled on the
  elliptic cone.
  """
  opt = load_env_cfg(TASK_ID).sim.mujoco
  assert opt.timestep == 0.005
  assert opt.integrator == "implicitfast"
  assert opt.cone == "elliptic"
  assert opt.impratio == 10


def test_reorientation_command() -> None:
  cfg = load_env_cfg(TASK_ID)
  assert isinstance(cfg.commands.get("goal"), ReorientationCommandCfg)


def test_palm_site_filled() -> None:
  """The per-robot palm site is wired into the cube-relative terms."""
  cfg = load_env_cfg(TASK_ID)
  assert cfg.observations["actor"].terms["cube_pos"].params["asset_cfg"].site_names == (
    PALM_SITE,
  )


def test_grip_friction_dr_is_startup() -> None:
  """Grip-friction DR randomizes sliding + torsional friction once per env at build.

  Friction is a material property, so it is sampled at ``startup`` (not per
  ``reset``). The pad and cube each get a sliding (axis 0) and a torsional (axis 1)
  term; all non-DR events stay ``reset``.
  """
  cfg = load_env_cfg(TASK_ID)
  dr_terms = {
    "cube_friction_slide",
    "cube_friction_spin",
    "pad_friction_slide",
    "pad_friction_spin",
  }
  assert dr_terms <= set(cfg.events)
  assert all(cfg.events[t].mode == "startup" for t in dr_terms)
  # All DR is startup (material/physical params sampled once per env); the rest reset.
  startup_dr = dr_terms | {"cube_mass", "cube_com", "cube_size", "encoder_bias"}
  assert all(
    term.mode == "reset" for name, term in cfg.events.items() if name not in startup_dr
  )


def test_play_disables_corruption() -> None:
  cfg = load_env_cfg(TASK_ID, play=True)
  assert cfg.observations["actor"].enable_corruption is False


def test_perception_dr_actor_corrupted_critic_clean() -> None:
  """The cube-pose obs carry noise + a per-env pose "blip" on the actor; the privileged
  critic is clean (enable_corruption=False strips it). No delay and no history are
  modeled (a realistic constant lag and any history would be a later add).
  """
  from mjlab.utils.noise import OutlierNoiseCfg

  cfg = load_env_cfg(TASK_ID)  # training cfg (corruption on)
  actor = cfg.observations["actor"]
  critic = cfg.observations["critic"]
  assert actor.enable_corruption is True
  assert critic.enable_corruption is False

  # The orientation terms carry the per-env pose "blip" on the actor.
  assert isinstance(actor.terms["cube_ori"].noise, OutlierNoiseCfg)
  assert isinstance(actor.terms["cube_to_goal_ori"].noise, OutlierNoiseCfg)

  # No delay, no history (kept simple until a realistic, hardware-measured lag).
  for term in ("cube_pos", "cube_ori", "cube_to_goal_ori"):
    assert actor.terms[term].delay_max_lag == 0
  assert actor.history_length in (None, 0)


def test_dynamics_dr_and_encoder_bias() -> None:
  """The physical-param DR (cube mass/CoM/size + joint encoder bias) is wired as startup
  events, and the encoder bias is handled at the obs level: the actor reads the biased
  encoder (biased=True) while the privileged critic reads the true position.
  """
  cfg = load_env_cfg(TASK_ID)
  for name in ("cube_mass", "cube_com", "cube_size", "encoder_bias"):
    assert name in cfg.events, f"missing DR event: {name}"
    assert cfg.events[name].mode == "startup"
  # cube_size must be isotropic (one factor on all axes -> stays a cube).
  assert cfg.events["cube_size"].params["isotropic"] is True
  # Encoder bias: actor sees the biased reading, critic sees the true position.
  assert cfg.observations["actor"].terms["joint_pos"].params["biased"] is True
  assert cfg.observations["critic"].terms["joint_pos"].params["biased"] is False


# --- Shared env fixture + helpers -------------------------------------------


@pytest.fixture(scope="module")
def env():
  """Build the reorient env once for all runtime tests. Each test resets first."""
  cfg = load_env_cfg(TASK_ID)
  cfg.scene.num_envs = 4
  e = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  e.reset()
  yield e
  e.close()


def _zero_action(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=env.device
  )


def _settle(env: ManagerBasedRlEnv) -> None:
  """Reset and step past the action warmup so the cube is stable in the grasp."""
  env.reset()
  warmup = int(getattr(env.action_manager.get_term("joint_pos"), "_warmup_steps", 0))
  for _ in range(warmup + 5):
    env.step(_zero_action(env))


def _force_cube_state(
  env: ManagerBasedRlEnv,
  pos_z: list[float],
  quat: torch.Tensor,
  ang_vel: torch.Tensor,
) -> None:
  """Force the cube into a known root pose + angular velocity (writes to sim).

  ``quat`` (n, 4) wxyz, ``ang_vel`` (n, 3) world frame, ``pos_z`` the world z per env.
  """
  n = env.num_envs
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


# --- Env smoke --------------------------------------------------------------


@pytest.mark.slow
def test_env_steps_without_nans(env) -> None:
  obs, _ = env.reset()
  assert obs["actor"].shape == (env.num_envs, 139)
  for _ in range(5):
    obs, rew, _, _, _ = env.step(torch.randn_like(_zero_action(env)))
    assert torch.isfinite(rew).all()
    for group in obs.values():
      assert torch.isfinite(group).all()


# --- Command state machine --------------------------------------------------


@pytest.mark.slow
def test_command_state_machine(env) -> None:
  """Drive the goal command through reach -> hold -> one success pulse -> window
  dwell -> goal switch, asserting each transition."""
  cmd = env.command_manager.get_term("goal")
  assert isinstance(cmd, ReorientationCommand)
  cube = env.scene["cube"]
  hold = cmd.cfg.success_hold_steps
  delay = cmd.cfg.goal_switch_delay

  _settle(env)
  # Clean "approaching" state.
  cmd.hold_counter[:] = 0
  cmd.success_count[:] = 0
  cmd.in_success_window[:] = False
  cmd.window_timer[:] = 0
  cmd.at_goal[:] = 0.0

  # Phase 1: hold within threshold (goal == cube) for `hold` steps. The hold counter
  # climbs one per step and exactly one at_goal pulse fires on the completing step.
  for i in range(hold):
    cmd.goal_quat[:] = cube.data.root_link_quat_w
    env.step(_zero_action(env))
    assert (cmd.within_threshold == 1).all(), f"cube left threshold at step {i}"
    assert (cmd.hold_counter == i + 1).all()
    if i < hold - 1:
      assert (cmd.at_goal == 0).all(), f"premature success at step {i}"
  assert (cmd.at_goal == 1).all(), "success did not fire after the hold"
  assert (cmd.success_count == 1).all()
  assert cmd.in_success_window.all()

  # Phase 2: the success window holds the goal fixed (no new success) for `delay`
  # steps, then advances to a new goal and clears the window.
  goal_before = cmd.goal_quat.clone()
  for _ in range(delay):
    env.step(_zero_action(env))
    assert (cmd.success_count == 1).all(), "spurious success during the window"
  assert not torch.allclose(cmd.goal_quat, goal_before), "goal did not switch"
  assert not cmd.in_success_window.any(), "window did not clear after the switch"


# --- Observations -----------------------------------------------------------


def _sixd_rotation_angle(sixd: torch.Tensor) -> torch.Tensor:
  """Rotation angle implied by a 6D rotation rep (Gram-Schmidt to a matrix).

  ``_quat_to_6d`` flattens the first two matrix columns row-major, so the 6 values
  are interleaved ``[m00, m01, m10, m11, m20, m21]``; reshape (3, 2) recovers the
  columns.
  """
  m = sixd.reshape(*sixd.shape[:-1], 3, 2)
  a, b = m[..., 0], m[..., 1]
  c1 = a / a.norm(dim=-1, keepdim=True)
  b = b - (c1 * b).sum(-1, keepdim=True) * c1
  c2 = b / b.norm(dim=-1, keepdim=True)
  c3 = torch.cross(c1, c2, dim=-1)
  trace = c1[..., 0] + c2[..., 1] + c3[..., 2]
  return torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))


@pytest.mark.slow
def test_object_to_goal_6d(env) -> None:
  """The cube->goal 6D obs is identity at the goal and encodes the error angle."""
  env.reset()
  cmd = env.command_manager.get_term("goal")
  assert isinstance(cmd, ReorientationCommand)
  cube_quat = env.scene["cube"].data.root_link_quat_w

  # Cube == goal -> identity rotation. The 6D is the first two columns flattened
  # row-major, so identity is [m00, m01, m10, m11, m20, m21] = [1, 0, 0, 1, 0, 0].
  cmd.goal_quat[:] = cube_quat
  sixd = object_to_goal_orientation_6d(env, "cube", "goal")
  identity = torch.tensor([1.0, 0, 0, 1.0, 0, 0], device=sixd.device)
  assert torch.allclose(sixd, identity.expand_as(sixd), atol=1e-4)

  # Goal a known angle off the cube -> the 6D reconstructs to that angle (the error
  # magnitude is frame independent, so the base-frame expression preserves it).
  angle = math.pi / 2
  axis = torch.randn(env.num_envs, 3, device=sixd.device)
  axis = axis / axis.norm(dim=-1, keepdim=True)
  cmd.goal_quat[:] = quat_box_plus(cube_quat, axis * angle)
  sixd = object_to_goal_orientation_6d(env, "cube", "goal")
  recon = _sixd_rotation_angle(sixd)
  assert torch.allclose(recon, torch.full_like(recon, angle), atol=1e-3)


# --- Rewards ----------------------------------------------------------------


@pytest.mark.slow
def test_tolerance_gate_zeros_below_height(env) -> None:
  """cube_orientation_tolerance is zeroed by the cube-held gate when dropped."""
  cmd = env.command_manager.get_term("goal")
  assert isinstance(cmd, ReorientationCommand)
  n = env.num_envs
  cmd.goal_quat[:] = torch.tensor([1.0, 0, 0, 0], device=cmd.device).expand(n, 4)
  axis = torch.tensor([0.0, 0, 1.0], device=cmd.device).expand(n, 3)
  q_err = quat_from_angle_axis(torch.full((n,), 0.5, device=cmd.device), axis)
  # Env 0 cradled (above the gate), the rest dropped (below it).
  _force_cube_state(
    env, pos_z=[0.14] + [0.05] * (n - 1), quat=q_err, ang_vel=torch.zeros(n, 3)
  )
  r = cube_orientation_tolerance(
    env, "goal", gate_object_name="cube", gate_min_height=0.10
  )
  assert r[0].item() > 0  # cradled -> positive
  assert (r[1:] == 0.0).all()  # dropped -> zero


@pytest.mark.slow
def test_sustained_hold_monotonic_no_dip_and_reset(env) -> None:
  """cumulative_hold grows by exactly within_threshold, never dips (incl. at goal
  switches), and resets only on episode reset; sustained_hold stays in [0, 1]."""
  env.reset()
  cmd = env.command_manager.get_term("goal")
  assert isinstance(cmd, ReorientationCommand)
  cube = env.scene["cube"]

  # 1) Increment law over a random rollout: every non-reset step the counter advances
  #    by exactly within_threshold (>= 0), so it can never dip -- not even at a switch.
  prev_cum = cmd.cumulative_hold.clone()
  prev_len = env.episode_length_buf.clone()
  for _ in range(120):
    env.step(0.3 * torch.randn_like(_zero_action(env)))
    advanced = env.episode_length_buf > prev_len  # stepped within the same episode
    delta = cmd.cumulative_hold[advanced] - prev_cum[advanced]
    assert torch.allclose(delta, cmd.within_threshold[advanced], atol=1e-4)
    assert (delta >= -1e-6).all()
    prev_cum = cmd.cumulative_hold.clone()
    prev_len = env.episode_length_buf.clone()

  # 2) Force in-threshold (goal == cube): the counter must grow.
  _settle(env)
  for _ in range(8):
    cmd.goal_quat[:] = cube.data.root_link_quat_w
    env.step(_zero_action(env))
  assert (cmd.cumulative_hold > 0).all()

  # 3) Move the goal 180 deg away: out of threshold, so the counter PAUSES, not dips.
  flip = torch.tensor([0.0, 1.0, 0.0, 0.0], device=env.device).expand_as(cmd.goal_quat)
  before = cmd.cumulative_hold.clone()
  for _ in range(8):
    cmd.goal_quat[:] = quat_mul(flip, cube.data.root_link_quat_w)
    env.step(_zero_action(env))
  assert (cmd.cumulative_hold >= before - 1e-6).all()

  # 4) Episode reset zeroes it; the reward is bounded.
  env.reset()
  assert int(torch.count_nonzero(cmd.cumulative_hold)) == 0
  reward = sustained_hold(env, "goal", 250.0, "cube", 0.05)
  assert torch.isfinite(reward).all()
  assert float(reward.min()) >= 0.0 and float(reward.max()) <= 1.0


@pytest.mark.slow
def test_normalized_torque_penalty(env) -> None:
  """tau_max is the per-actuator effort limit (non-uniform), and the penalty equals
  sum((tau / tau_max)^2)."""
  env.reset()
  asset_cfg = SceneEntityCfg("robot")
  asset_cfg.resolve(env.scene)
  cfg = RewardTermCfg(
    func=NormalizedJointTorquePenalty, weight=-0.1, params={"asset_cfg": asset_cfg}
  )
  term = NormalizedJointTorquePenalty(cfg, env)

  forcerange = np.clip(env.sim.mj_model.actuator_forcerange[:, 1], 1e-3, None)
  assert np.allclose(term._tau_max.cpu().numpy(), forcerange)
  # Effort limits span a wide range on this hand, so normalizing actually matters.
  assert float(term._tau_max.max() / term._tau_max.min()) > 5.0

  for _ in range(5):
    env.step(0.3 * torch.randn_like(_zero_action(env)))
  value = term(env)
  tau = env.scene["robot"].data.actuator_force[:, term._act_ids]
  expected = torch.sum(torch.square(tau / term._tau_max), dim=-1)
  assert torch.allclose(value, expected, atol=1e-5)
  assert torch.isfinite(value).all() and (value >= 0).all()
