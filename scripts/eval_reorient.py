"""Deterministic eval harness for the Sharpa in-hand cube reorientation policy.

Two complementary modes:

* ``reach`` -- per-pose capability. Each env gets one fixed goal that is a
  controlled relative rotation (stratified angle, seeded axis) of its settled
  start, so difficulty *is* the rotation angle. No goal switching, no episode
  resets: every env runs the full time budget against its single goal. Reports
  success rate (held + touched), time-to-reach, errors, drop rate, and a
  success-vs-difficulty curve.
* ``chain`` -- sustained performance. The normal resampling command runs for a
  long fixed episode (seeded), measuring consecutive successes and the interval
  between them.

Both modes also report motion-quality stats (joint speed, action jerk, torque)
so smoothness can be hill-climbed numerically instead of eyeballed.

Determinism: a fixed seed fixes the starts and goals, observation corruption is
off (play cfg), and the policy acts on its mean (inference) output.

Example:
  uv run python scripts/eval_reorient.py --run gcbc_researchers/mjlab/6trb04od
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.manipulation.mdp import ReorientationCommand
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_box_plus, quat_error_magnitude
from mjlab.utils.os import get_wandb_checkpoint_path

TASK = "Mjlab-Reorient-Cube-Sharpa"
SUCCESS_THRESHOLD = 0.2  # rad; matches the training command.
DROP_HEIGHT = 0.05  # cube root z below this (in hand frame ~0.09) counts as dropped.


def _log(msg: str) -> None:
  """Progress to stderr so stdout stays a clean JSON dump for --out / piping."""
  print(msg, file=sys.stderr, flush=True)


def _build(env_cfg, run: str, device: str):
  """Build the env + wrapper and load the policy from a W&B run."""
  agent_cfg = load_rl_cfg(TASK)
  log_root = (Path("logs") / agent_cfg.experiment_name).resolve()
  _log(f"[build] fetching checkpoint for {run} ...")
  ckpt, _ = get_wandb_checkpoint_path(log_root, Path(run), None)
  _log(f"[build] compiling env ({env_cfg.scene.num_envs} envs, {device}) ...")
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
  runner = runner_cls(wenv, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  _log(f"[build] loaded {ckpt.name}")
  return env, wenv, policy, ckpt.name


def _warmup_steps(env: ManagerBasedRlEnv) -> int:
  action = env.action_manager.get_term("joint_pos")
  return int(getattr(action, "_warmup_steps", 0))


class _Quality:
  """Rolling collector for motion-quality stats."""

  def __init__(self):
    self.speed, self.jerk, self.torque = [], [], []
    self._a1 = self._a2 = None

  def update(self, env: ManagerBasedRlEnv, action: torch.Tensor):
    robot = env.scene["robot"]
    self.speed.append(robot.data.joint_vel.abs().mean().item())
    self.torque.append(robot.data.actuator_force.abs().mean().item())
    if self._a1 is not None and self._a2 is not None:
      self.jerk.append((action - 2 * self._a1 + self._a2).pow(2).sum(-1).mean().item())
    self._a2, self._a1 = self._a1, action

  def summary(self) -> dict:
    def ms(x):
      a = np.asarray(x)
      return {"mean": float(a.mean()), "p95": float(np.percentile(a, 95))}

    return {
      "joint_speed": ms(self.speed),
      "action_jerk": ms(self.jerk),
      "torque": ms(self.torque),
    }


def mode_reach(
  run: str,
  num_poses: int,
  budget_s: float,
  device: str,
  seed: int,
  threshold: float = SUCCESS_THRESHOLD,
):
  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = num_poses
  # Deterministic start: no reset pitch/position noise.
  reset = env_cfg.events["reset_hand_and_cube"]
  reset.params["hand_pitch_range"] = (-0.15, -0.15)
  reset.params["position_noise"] = 0.0
  # No episode resets during a trial; one fixed goal per env for the full budget.
  env_cfg.terminations = {}

  torch.manual_seed(seed)
  env, wenv, policy, ckpt = _build(env_cfg, run, device)
  cmd = env.command_manager.get_term("goal")
  assert isinstance(cmd, ReorientationCommand)
  cube = env.scene["cube"]
  dt = float(env.step_dt)
  warmup = _warmup_steps(env)
  budget = int(round(budget_s / dt))

  # Stratified relative-rotation goals: angle linearly spans [30, 180] deg, axis
  # is a seeded random unit vector. goal = box_plus(start, axis * angle).
  angles = torch.linspace(np.deg2rad(30.0), np.pi, num_poses, device=device)
  axes = torch.randn(num_poses, 3, device=device)
  axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-9)

  _log(
    f"[reach] {num_poses} poses | budget {budget_s}s ({budget} steps) | "
    f"threshold {threshold} rad | settling {warmup} steps ..."
  )
  obs, _ = wenv.reset()
  for _ in range(warmup):  # let the cube settle into the grasp.
    with torch.no_grad():
      obs, _, _, _ = wenv.step(policy(obs))
  # Lock in goals relative to each settled start; clear the success machinery.
  q_start = cube.data.root_link_quat_w.clone()
  cmd.goal_quat[:] = quat_box_plus(q_start, axes * angles.unsqueeze(-1))
  cmd.hold_counter[:] = 0
  cmd.success_count[:] = 0
  cmd.in_success_window[:] = False
  cmd.window_timer[:] = 0
  obs = wenv.get_observations()

  B = num_poses
  err = torch.zeros(budget, B, device=device)
  within = torch.zeros(budget, B, dtype=torch.bool, device=device)
  min_z = torch.full((B,), float("inf"), device=device)
  quality = _Quality()
  every = max(1, budget // 5)
  for t in range(budget):
    with torch.no_grad():
      action = policy(obs)
    obs, _, _, _ = wenv.step(action)
    e = quat_error_magnitude(cube.data.root_link_quat_w, cmd.goal_quat)
    err[t] = e
    within[t] = e < threshold
    min_z = torch.minimum(min_z, cube.data.root_link_pos_w[:, 2])
    if t % every == 0 or t == budget - 1:
      _log(
        f"[reach] step {t + 1}/{budget}  in-threshold {within[t].float().mean():.0%}"
      )
    quality.update(env, action)

  # Per-pose metrics.
  touched = within.any(0)
  # held = exists a run of `hold` consecutive in-threshold steps.
  hold = cmd.cfg.success_hold_steps
  kernel = within.float().t().unsqueeze(1)  # (B,1,budget)
  run_sum = torch.nn.functional.conv1d(
    kernel, torch.ones(1, 1, hold, device=device)
  ).squeeze(1)  # (B, budget-hold+1)
  held = (run_sum >= hold).any(-1)
  ttr = torch.where(
    touched, within.float().argmax(0).float(), torch.tensor(float("nan"))
  )
  dropped = min_z < DROP_HEIGHT
  best_err = err.min(0).values
  deg = torch.rad2deg(angles)

  def frac(x):
    return float(x.float().mean())

  result = {
    "mode": "reach",
    "run": run,
    "ckpt": ckpt,
    "num_poses": num_poses,
    "budget_s": budget_s,
    "threshold_rad": threshold,
    "seed": seed,
    "held_success_rate": frac(held),
    "touched_rate": frac(touched),
    "drop_rate": frac(dropped),
    "time_to_reach_s_median": float(np.nanmedian((ttr * dt).cpu().numpy())),
    "best_error_rad_mean": float(best_err.mean()),
    "difficulty_curve": [],
    "quality": quality.summary(),
  }
  # Success vs difficulty: bin by goal angle.
  bins = [(0, 90), (90, 135), (135, 181)]
  for lo, hi in bins:
    m = (deg >= lo) & (deg < hi)
    if m.any():
      result["difficulty_curve"].append(
        {
          "angle_deg": f"{lo}-{hi}",
          "n": int(m.sum()),
          "held": frac(held[m]),
          "touched": frac(touched[m]),
        }
      )
  return result


def mode_chain(run: str, episode_s: float, num_envs: int, device: str, seed: int):
  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = num_envs
  env_cfg.episode_length_s = episode_s
  env_cfg.terminations = {  # keep drop termination; drop the time_out so we measure one long episode.
    k: v for k, v in env_cfg.terminations.items() if k != "time_out"
  }
  torch.manual_seed(seed)
  env, wenv, policy, ckpt = _build(env_cfg, run, device)
  cmd = env.command_manager.get_term("goal")
  assert isinstance(cmd, ReorientationCommand)
  dt = float(env.step_dt)
  steps = int(round(episode_s / dt))

  _log(f"[chain] {num_envs} envs | episode {episode_s}s ({steps} steps) ...")
  obs, _ = wenv.reset()
  quality = _Quality()
  prev_succ = cmd.success_count.clone()
  # Per-env interval: step of each env's last success, and gaps between an env's
  # own successes (pooling across envs would just measure 24 staggered streams).
  last_succ_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
  intervals: list[int] = []
  within_frac = []
  every = max(1, steps // 5)
  for t in range(steps):
    with torch.no_grad():
      action = policy(obs)
    obs, _, _, _ = wenv.step(action)
    quality.update(env, action)
    within_frac.append(cmd.within_threshold.mean().item())
    succeeded = cmd.success_count > prev_succ
    for i in succeeded.nonzero().flatten().tolist():
      if last_succ_step[i] >= 0:
        intervals.append(t - int(last_succ_step[i]))
      last_succ_step[i] = t
    if t % every == 0 or t == steps - 1:
      _log(f"[chain] step {t + 1}/{steps}  succ/env {cmd.success_count.mean():.1f}")
    prev_succ = cmd.success_count.clone()

  succ = cmd.success_count
  intervals_arr = np.asarray(intervals)
  return {
    "mode": "chain",
    "run": run,
    "ckpt": ckpt,
    "num_envs": num_envs,
    "episode_s": episode_s,
    "seed": seed,
    "successes_per_episode_mean": float(succ.mean()),
    "successes_per_episode_median": float(succ.median()),
    "inter_success_interval_s_median": float(np.median(intervals_arr) * dt)
    if intervals_arr.size
    else None,
    "time_in_threshold_frac": float(np.mean(within_frac)),
    "quality": quality.summary(),
  }


def run_viz(run: str, num_envs: int, device: str, seed: int, viewer: str):
  """Watch the policy live (normal full-SO(3) chaining task) instead of metrics."""
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = num_envs
  torch.manual_seed(seed)
  env, wenv, policy, ckpt = _build(env_cfg, run, device)
  if viewer == "auto":
    # macOS has a GUI but no X11 DISPLAY, so prefer native there too.
    has_gui = sys.platform == "darwin" or bool(
      os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    viewer = "native" if has_gui else "viser"
  print(f"[viz] {ckpt} -> {viewer} viewer")
  if viewer == "native":
    NativeMujocoViewer(wenv, policy).run()
  else:
    ViserPlayViewer(wenv, policy).run()
  env.close()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", required=True, help="W&B run path, e.g. entity/project/id")
  ap.add_argument("--mode", choices=["reach", "chain", "both"], default="both")
  ap.add_argument("--num-poses", type=int, default=50)
  ap.add_argument("--budget-s", type=float, default=3.0)
  ap.add_argument(
    "--threshold",
    type=float,
    default=SUCCESS_THRESHOLD,
    help="Reach success threshold in rad (default matches training).",
  )
  ap.add_argument(
    "--hard",
    action="store_true",
    help="Discriminating reach: 0.1 rad threshold, 1 s budget.",
  )
  ap.add_argument("--episode-s", type=float, default=20.0)
  ap.add_argument("--chain-envs", type=int, default=32)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--out", type=str, default=None, help="Optional JSON output path.")
  ap.add_argument(
    "--viz",
    action="store_true",
    help="Launch the viewer with the policy instead of computing metrics.",
  )
  ap.add_argument("--viewer", choices=["auto", "native", "viser"], default="auto")
  ap.add_argument("--viz-envs", type=int, default=1)
  args = ap.parse_args()

  if args.viz:
    run_viz(args.run, args.viz_envs, args.device, args.seed, args.viewer)
    return

  budget_s, threshold = args.budget_s, args.threshold
  if args.hard:
    budget_s, threshold = 1.0, 0.1

  results = {}
  if args.mode in ("reach", "both"):
    results["reach"] = mode_reach(
      args.run, args.num_poses, budget_s, args.device, args.seed, threshold
    )
  if args.mode in ("chain", "both"):
    results["chain"] = mode_chain(
      args.run, args.episode_s, args.chain_envs, args.device, args.seed
    )

  print(json.dumps(results, indent=2))
  if args.out:
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
  main()
