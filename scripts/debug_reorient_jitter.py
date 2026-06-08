"""One-shot diagnostic: roll out a reorient policy and measure jitter / streaks.

Captures per-step action vector, joint vel, cube ang-vel, orientation_error,
hold_counter; analyses streak-length distribution, action churn, and the
relationship between "close to goal" and "wobbly". Single env, single rollout,
CPU is fine.

Usage:
  uv run python scripts/debug_reorient_jitter.py \\
    --checkpoint logs/.../model_2800.pt --steps 1500
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.manipulation.mdp.commands import ReorientationCommand
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def streak_lengths(within: np.ndarray) -> list[int]:
  """Lengths of consecutive in-threshold runs."""
  lengths: list[int] = []
  cur = 0
  for v in within:
    if v:
      cur += 1
    elif cur:
      lengths.append(cur)
      cur = 0
  if cur:
    lengths.append(cur)
  return lengths


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Mjlab-Reorient-Cube-Sharpa")
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--steps", type=int, default=1500)
  ap.add_argument("--output", default="/tmp/reorient_jitter.json")
  ap.add_argument("--device", default="cpu")
  args = ap.parse_args()

  env_cfg = load_env_cfg(args.task, play=True)  # infinite episode
  agent_cfg = load_rl_cfg(args.task)
  for g in env_cfg.observations.values():
    g.enable_corruption = False
  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  env_unwrapped = env
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(
    args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device
  )
  policy = runner.get_inference_policy(device=args.device)

  cmd = cast(ReorientationCommand, env_unwrapped.command_manager.get_term("goal"))
  robot = env_unwrapped.scene["robot"]
  cube = env_unwrapped.scene["cube"]

  # Buffers.
  ori_err: list[float] = []
  within: list[int] = []
  hold: list[int] = []
  goal_id: list[int] = []  # cumulative count of distinct goals seen
  actions_buf: list[np.ndarray] = []
  joint_vel_buf: list[np.ndarray] = []
  cube_ang_vel: list[float] = []

  obs = env.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]

  prev_goal = cmd.goal_quat.clone()
  goal_counter = 0

  for _ in range(args.steps):
    with torch.no_grad():
      a = policy(obs)
    obs, _, _, _ = env.step(a)

    actions_buf.append(a[0].cpu().numpy().copy())
    joint_vel_buf.append(robot.data.joint_vel[0].cpu().numpy().copy())
    cube_ang_vel.append(
      float(torch.linalg.norm(cube.data.root_link_ang_vel_w[0]).item())
    )
    ori_err.append(float(cmd.orientation_error[0].item()))
    within.append(int(cmd.within_threshold[0].item()))
    hold.append(int(cmd.hold_counter[0].item()))
    if not torch.allclose(prev_goal, cmd.goal_quat, atol=1e-6):
      goal_counter += 1
      prev_goal = cmd.goal_quat.clone()
    goal_id.append(goal_counter)

  env.close()

  actions = np.stack(actions_buf)  # [T, A]
  joint_vels = np.stack(joint_vel_buf)  # [T, J]
  ori_err_a = np.array(ori_err)
  within_a = np.array(within)
  hold_a = np.array(hold)
  cube_ang_vel_a = np.array(cube_ang_vel)
  T = actions.shape[0]

  # Streak distribution.
  streaks = streak_lengths(within_a)
  successes = int((hold_a == cmd.cfg.success_hold_steps).sum())
  total_goals = goal_counter

  # Jitter metrics on actions.
  action_norm = np.linalg.norm(actions, axis=1)  # per-step L2
  action_delta = np.linalg.norm(np.diff(actions, axis=0), axis=1)  # per-step change
  # FFT of one representative action dim (first one): power at high freq.
  rep = actions[:, 0] - actions[:, 0].mean()
  fft = np.fft.rfft(rep)
  power = np.abs(fft) ** 2
  hf_frac = float(power[len(power) // 2 :].sum() / max(power.sum(), 1e-12))

  joint_vel_norm = np.linalg.norm(joint_vels, axis=1)
  joint_vel_per_joint_p99 = np.quantile(np.abs(joint_vels), 0.99, axis=0)

  # Compare "near goal" vs "far from goal" action behavior.
  near = ori_err_a < 0.25  # near the goal
  far = ori_err_a > 0.5

  def stats(mask, x):
    if mask.sum() < 5:
      return None
    return {
      "n": int(mask.sum()),
      "mean": float(x[mask].mean()),
      "std": float(x[mask].std()),
      "p90": float(np.quantile(x[mask], 0.9)),
    }

  report = {
    "checkpoint": args.checkpoint,
    "rollout_steps": T,
    "summary": {
      "orientation_error": {
        "mean": float(ori_err_a.mean()),
        "p50": float(np.quantile(ori_err_a, 0.5)),
        "p10": float(np.quantile(ori_err_a, 0.1)),
      },
      "within_threshold_frac": float(within_a.mean()),
      "successes_in_rollout": successes,
      "total_goals_seen": total_goals,
      "streaks": {
        "count": len(streaks),
        "mean_len": float(np.mean(streaks)) if streaks else 0.0,
        "median_len": float(np.median(streaks)) if streaks else 0.0,
        "max_len": int(max(streaks)) if streaks else 0,
        "len_ge_5_count": int(sum(1 for s in streaks if s >= 5)),
        "histogram": {
          f"len_{k}": int(sum(1 for s in streaks if s == k)) for k in range(1, 11)
        },
        "long_tail_ge_10": int(sum(1 for s in streaks if s >= 10)),
      },
    },
    "actions": {
      "norm_mean": float(action_norm.mean()),
      "norm_p90": float(np.quantile(action_norm, 0.9)),
      "delta_norm_mean": float(action_delta.mean()),
      "delta_norm_p90": float(np.quantile(action_delta, 0.9)),
      "delta_norm_max": float(action_delta.max()),
      "high_freq_power_frac_dim0": hf_frac,
      "near_goal_delta_norm": stats(near[1:], action_delta),
      "far_from_goal_delta_norm": stats(far[1:], action_delta),
    },
    "joint_vel": {
      "norm_mean": float(joint_vel_norm.mean()),
      "norm_p90": float(np.quantile(joint_vel_norm, 0.9)),
      "norm_max": float(joint_vel_norm.max()),
      "abs_p99_per_joint_max": float(joint_vel_per_joint_p99.max()),
      "abs_p99_per_joint_mean": float(joint_vel_per_joint_p99.mean()),
      "near_goal_norm": stats(near, joint_vel_norm),
      "far_from_goal_norm": stats(far, joint_vel_norm),
    },
    "cube": {
      "ang_vel_norm_mean": float(cube_ang_vel_a.mean()),
      "ang_vel_norm_p90": float(np.quantile(cube_ang_vel_a, 0.9)),
      "ang_vel_norm_max": float(cube_ang_vel_a.max()),
      "near_goal_ang_vel": stats(near, cube_ang_vel_a),
      "far_from_goal_ang_vel": stats(far, cube_ang_vel_a),
    },
  }

  out = Path(args.output)
  out.parent.mkdir(parents=True, exist_ok=True)
  with open(out, "w") as f:
    json.dump(
      report
      | {
        "trajectory": {
          "ori_err": ori_err,
          "within": within,
          "hold": hold,
          "goal_id": goal_id,
          "cube_ang_vel": list(cube_ang_vel_a),
        },
      },
      f,
      indent=2,
    )

  # Pretty-print the summary.
  print(json.dumps(report, indent=2))
  print(f"\n[INFO] Full per-step trajectory saved to {out}")


if __name__ == "__main__":
  main()
