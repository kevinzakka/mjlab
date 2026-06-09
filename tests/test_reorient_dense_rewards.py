"""Tests for the dense monotonic hold reward and the normalized torque penalty."""

import numpy as np
import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.reorient import mdp as reorient_mdp
from mjlab.tasks.reorient.config.sharpa.env_cfgs import sharpa_reorient_cube_env_cfg
from mjlab.tasks.reorient.mdp.rewards import NormalizedJointTorquePenalty
from mjlab.utils.lab_api.math import quat_mul

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def env():
  cfg = sharpa_reorient_cube_env_cfg()
  cfg.scene.num_envs = 8
  e = ManagerBasedRlEnv(cfg, device=DEVICE)
  yield e
  e.close()


def _zero_action(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=DEVICE)


@pytest.mark.slow
def test_sustained_hold_monotonic_no_dip_and_reset(env):
  """cumulative_hold grows by exactly within_threshold, never dips (incl. at goal
  switches), and resets only on episode reset; sustained_hold stays in [0, 1]."""
  env.reset()
  cmd = env.command_manager.get_term("goal")
  cube = env.scene["cube"]

  # 1) Increment law over a random rollout: every non-reset step, the counter
  #    advances by exactly within_threshold (>= 0), so it can never dip -- not even
  #    when the goal advances.
  prev_cum = cmd.cumulative_hold.clone()
  prev_len = env.episode_length_buf.clone()
  for _ in range(120):
    act = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=DEVICE)
    env.step(0.3 * act)
    advanced = env.episode_length_buf > prev_len  # stepped within the same episode
    delta = cmd.cumulative_hold[advanced] - prev_cum[advanced]
    assert torch.allclose(delta, cmd.within_threshold[advanced], atol=1e-4)
    assert (delta >= -1e-6).all()
    prev_cum = cmd.cumulative_hold.clone()
    prev_len = env.episode_length_buf.clone()

  # 2) Force in-threshold (goal == cube): the counter must grow.
  env.reset()
  for _ in range(25):  # settle past the action warmup
    env.step(_zero_action(env))
  for _ in range(8):
    cmd.goal_quat[:] = cube.data.root_link_quat_w
    env.step(_zero_action(env))
  grew = cmd.cumulative_hold.clone()
  assert (grew > 0).all()

  # 3) Move the goal far away (180 deg): out of threshold, so the counter must
  #    PAUSE -- not drop. This is the no-dip-at-goal-switch property.
  flip = torch.tensor([0.0, 1.0, 0.0, 0.0], device=DEVICE).expand_as(cmd.goal_quat)
  before = cmd.cumulative_hold.clone()
  for _ in range(8):
    cmd.goal_quat[:] = quat_mul(flip, cube.data.root_link_quat_w)
    env.step(_zero_action(env))
  assert (cmd.cumulative_hold >= before - 1e-6).all()

  # 4) Episode reset zeroes it; the reward is bounded.
  env.reset()
  assert int(torch.count_nonzero(cmd.cumulative_hold)) == 0
  reward = reorient_mdp.sustained_hold(env, "goal", 250.0, "cube", 0.05)
  assert torch.isfinite(reward).all()
  assert float(reward.min()) >= 0.0 and float(reward.max()) <= 1.0


@pytest.mark.slow
def test_normalized_torque_penalty(env):
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
  # Effort limits really do span a wide range on this hand, so normalizing matters.
  assert float(term._tau_max.max() / term._tau_max.min()) > 5.0

  for _ in range(5):
    act = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=DEVICE)
    env.step(0.3 * act)
  value = term(env)
  tau = env.scene["robot"].data.actuator_force[:, term._act_ids]
  expected = torch.sum(torch.square(tau / term._tau_max), dim=-1)
  assert torch.allclose(value, expected, atol=1e-5)
  assert torch.isfinite(value).all() and (value >= 0).all()
