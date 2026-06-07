"""Tests for the Sharpa in-hand cube reorientation task."""

import pytest
import torch

from mjlab.tasks.manipulation.mdp import ReorientationCommandCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

TASK_ID = "Mjlab-Reorient-Cube-Sharpa"
PALM_SITE = "wrist_site"


def test_task_registered() -> None:
  assert TASK_ID in list_tasks()


def test_sim_options() -> None:
  """Simulator options (owned by SimulationCfg) match the contact-rich setup."""
  cfg = load_env_cfg(TASK_ID)
  opt = cfg.sim.mujoco
  assert opt.timestep == 0.005
  assert opt.integrator == "implicitfast"
  assert opt.cone == "elliptic"
  assert opt.impratio == 1


def test_reorientation_command() -> None:
  cfg = load_env_cfg(TASK_ID)
  assert "goal" in cfg.commands
  assert isinstance(cfg.commands["goal"], ReorientationCommandCfg)


def test_palm_site_filled() -> None:
  """The per-robot palm site is wired into the cube-relative terms."""
  cfg = load_env_cfg(TASK_ID)
  assert cfg.observations["actor"].terms["cube_pos"].params["asset_cfg"].site_names == (
    PALM_SITE,
  )
  assert cfg.rewards["stay_near_palm"].params["asset_cfg"].site_names == (PALM_SITE,)
  assert cfg.terminations["cube_dropped"].params["asset_cfg"].site_names == (PALM_SITE,)


def test_no_domain_randomization() -> None:
  """No DR until the task trains well: only reset events are present."""
  cfg = load_env_cfg(TASK_ID)
  assert all(term.mode == "reset" for term in cfg.events.values())


def test_play_disables_corruption() -> None:
  cfg = load_env_cfg(TASK_ID, play=True)
  assert cfg.observations["actor"].enable_corruption is False


@pytest.mark.slow
def test_env_steps_without_nans() -> None:
  from mjlab.envs import ManagerBasedRlEnv

  cfg = load_env_cfg(TASK_ID)
  cfg.scene.num_envs = 4
  device = "cuda" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    obs, _ = env.reset()
    actor = obs["actor"]
    assert isinstance(actor, torch.Tensor) and actor.shape == (4, 87)
    for _ in range(5):
      act = torch.randn(
        env.num_envs, env.action_manager.total_action_dim, device=device
      )
      obs, rew, _, _, _ = env.step(act)
      assert torch.isfinite(rew).all()
      for group in obs.values():
        assert isinstance(group, torch.Tensor) and torch.isfinite(group).all()
  finally:
    env.close()
