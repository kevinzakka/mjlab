"""Evaluate a trained in-hand cube reorientation policy.

Loads a policy checkpoint (from a W&B run or a local .pt file), rolls it out
across many parallel envs, and reports per-episode and aggregate diagnostics
useful for debugging the resample-on-success loop:

  * What is the *minimum* orientation error each episode reached?
  * What fraction of episodes crossed the success threshold (default 0.1 rad),
    and what fraction would have crossed it under looser thresholds?
  * How many goals does the policy chain per episode (success_count)?
  * Why do episodes end (time-out vs cube-drop vs runaway velocity)?

The thresholded success curve (success at thresholds 0.05, 0.10, ..., 0.50)
is the key signal for "is the threshold too tight?" — if very few episodes
reach 0.10 rad but most reach 0.15 rad, the policy is converging but
under-resolved relative to the gate, not stuck on the wrong solution.

Example:
  uv run python -m mjlab.tasks.reorient.scripts.evaluate_reorient \\
    Mjlab-Reorient-Cube-Sharpa \\
    --wandb-run-path gcbc_researchers/mjlab/9o6qwl4o
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.reorient.mdp.commands import ReorientationCommand
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class EvaluateConfig:
  """Configuration for reorientation-policy evaluation."""

  wandb_run_path: str | None = None
  """W&B run path, e.g. 'entity/project/run_id'. Either this or
  ``checkpoint_file`` must be set."""

  wandb_checkpoint_name: str | None = None
  """Optional checkpoint filename within the W&B run (e.g. 'model_4000.pt').
  Defaults to the latest model checkpoint."""

  checkpoint_file: str | None = None
  """Local .pt checkpoint to load instead of pulling from W&B."""

  num_envs: int = 256
  """Number of parallel envs. Each env produces ``episodes_per_env``
  episodes, so total episodes = num_envs * episodes_per_env."""

  episodes_per_env: int = 1
  """How many episodes each env should complete before the rollout ends."""

  max_steps: int = 4000
  """Hard cap on total environment steps so a wedged policy can't hang."""

  thresholds: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
  """Orientation-error thresholds (radians) used in the success curve."""

  device: str | None = None
  """Device. Defaults to CUDA if available."""

  output_file: str | None = None
  """Optional path to dump full per-episode results + summary as JSON."""

  save_trajectory_envs: int = 0
  """How many envs' per-step error trajectories to save in ``output_file``.
  0 saves none. Useful for plotting in a notebook."""

  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs are written/cached."""


@dataclass
class _EpisodeRecord:
  env_idx: int
  episode_idx: int
  steps: int
  min_error: float
  final_error: float
  success_count: int
  first_success_step: int  # -1 if never
  termination: str  # one of: time_out, cube_dropped, cube_velocity, max_steps


def _resolve_checkpoint(cfg: EvaluateConfig, agent_cfg) -> Path:
  if cfg.checkpoint_file is not None:
    p = Path(cfg.checkpoint_file)
    if not p.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {p}")
    print(f"[INFO] Loading local checkpoint: {p}")
    return p
  if cfg.wandb_run_path is None:
    raise ValueError("Must provide either --wandb-run-path or --checkpoint-file.")
  log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
  resume_path, was_cached = get_wandb_checkpoint_path(
    log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
  )
  print(
    f"[INFO] Loading W&B checkpoint: {resume_path.name} "
    f"(run: {resume_path.parent.name}, {'cached' if was_cached else 'downloaded'})"
  )
  return resume_path


def _percentiles(x: torch.Tensor, qs: tuple[float, ...]) -> dict[str, float]:
  if x.numel() == 0:
    return {f"p{int(q * 100)}": float("nan") for q in qs}
  return {
    f"p{int(q * 100)}": x.quantile(torch.tensor(q, device=x.device)).item() for q in qs
  }


def run_evaluate(task_id: str, cfg: EvaluateConfig) -> dict:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=False)
  agent_cfg = load_rl_cfg(task_id)

  if "goal" not in env_cfg.commands:
    raise ValueError(
      f"Task {task_id} does not have a 'goal' command — is this a reorient task?"
    )

  # Disable obs corruption so we measure policy capability, not noise robustness.
  for group in env_cfg.observations.values():
    group.enable_corruption = False

  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env_unwrapped = env
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  resume_path = _resolve_checkpoint(cfg, agent_cfg)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  command = cast(ReorientationCommand, env_unwrapped.command_manager.get_term("goal"))

  term_mgr = env_unwrapped.termination_manager
  termination_names = [
    n
    for n in ("time_out", "cube_dropped", "cube_velocity")
    if n in term_mgr.active_terms
  ]

  num_envs = cfg.num_envs
  records: list[_EpisodeRecord] = []
  episodes_done = torch.zeros(num_envs, dtype=torch.int32, device=device)

  # Per-episode trackers (reset on each done).
  ep_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
  ep_min_error = torch.full(
    (num_envs,), float("inf"), dtype=torch.float32, device=device
  )
  ep_first_success = torch.full((num_envs,), -1, dtype=torch.int32, device=device)

  # Optional per-step trajectory log (env 0 .. save_trajectory_envs-1).
  trajectory_log: list[list[float]] = (
    [[] for _ in range(cfg.save_trajectory_envs)]
    if cfg.save_trajectory_envs > 0
    else []
  )

  obs = env.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]

  total_episodes = num_envs * cfg.episodes_per_env
  print(
    f"[INFO] Rolling out {num_envs} envs × {cfg.episodes_per_env} episodes "
    f"= {total_episodes} episodes (max_steps={cfg.max_steps})"
  )

  step = 0
  while episodes_done.min().item() < cfg.episodes_per_env and step < cfg.max_steps:
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env.step(actions)

    # Read pre-resample success/error cached by the command in _update_metrics.
    err = command.orientation_error
    at_goal = command.at_goal.bool()

    ep_steps += 1
    ep_min_error = torch.minimum(ep_min_error, err)
    # Latch first-success step (only fill envs that have not yet succeeded).
    fresh_success = at_goal & (ep_first_success < 0)
    ep_first_success = torch.where(
      fresh_success, ep_steps.to(ep_first_success.dtype), ep_first_success
    )

    for i, env_traj in enumerate(trajectory_log):
      env_traj.append(err[i].item())

    if dones.any():
      done_idx = dones.nonzero().flatten().tolist()
      # Snapshot per-term done flags before reset so we can attribute cause.
      term_flags = {n: term_mgr.get_term(n).clone() for n in termination_names}
      # And success_count for this episode (cleared on the env's next reset).
      success_count_now = command.success_count.clone()
      for ei in done_idx:
        # Determine cause: the first matching termination term wins, falling
        # back to 'unknown' if none matched (shouldn't happen, but defensive).
        cause = "unknown"
        for name in termination_names:
          if term_flags[name][ei].item():
            cause = name
            break
        records.append(
          _EpisodeRecord(
            env_idx=ei,
            episode_idx=int(episodes_done[ei].item()),
            steps=int(ep_steps[ei].item()),
            min_error=float(ep_min_error[ei].item()),
            final_error=float(err[ei].item()),
            success_count=int(success_count_now[ei].item()),
            first_success_step=int(ep_first_success[ei].item()),
            termination=cause,
          )
        )
        episodes_done[ei] += 1
        ep_steps[ei] = 0
        ep_min_error[ei] = float("inf")
        ep_first_success[ei] = -1
      completed = sum(r.success_count > 0 for r in records)
      print(
        f"[step {step:4d}] done={len(records):4d}/{total_episodes} "
        f"any-success={completed:4d} "
        f"min-err p50={torch.quantile(ep_min_error[ep_min_error.isfinite()], 0.5).item() if ep_min_error.isfinite().any() else float('nan'):.3f}"
      )
    step += 1

  if step >= cfg.max_steps and len(records) < total_episodes:
    print(
      f"[WARN] Hit max_steps={cfg.max_steps} with "
      f"{total_episodes - len(records)} episode(s) still running. "
      "Recording them as in-progress with termination='max_steps'."
    )
    for ei in range(num_envs):
      while int(episodes_done[ei].item()) < cfg.episodes_per_env:
        records.append(
          _EpisodeRecord(
            env_idx=ei,
            episode_idx=int(episodes_done[ei].item()),
            steps=int(ep_steps[ei].item()),
            min_error=float(ep_min_error[ei].item()),
            final_error=float(command.orientation_error[ei].item()),
            success_count=int(command.success_count[ei].item()),
            first_success_step=int(ep_first_success[ei].item()),
            termination="max_steps",
          )
        )
        episodes_done[ei] += 1
        # Only the first 'in-progress' record per env reflects partial state;
        # subsequent placeholders just pad to episodes_per_env.
        ep_steps[ei] = 0
        ep_min_error[ei] = float("inf")
        ep_first_success[ei] = -1

  # Aggregate.
  min_errors = torch.tensor([r.min_error for r in records], device=device)
  final_errors = torch.tensor([r.final_error for r in records], device=device)
  successes = torch.tensor([r.success_count for r in records], device=device)
  first_succ = torch.tensor(
    [r.first_success_step for r in records], dtype=torch.float32, device=device
  )

  success_curve = {
    f"min_err<={t:.2f}": (min_errors <= t).float().mean().item() for t in cfg.thresholds
  }
  term_breakdown = {
    n: sum(r.termination == n for r in records) / max(len(records), 1)
    for n in termination_names + ["max_steps", "unknown"]
  }
  succ_steps = first_succ[first_succ >= 0]
  summary = {
    "task_id": task_id,
    "checkpoint": str(resume_path),
    "num_episodes": len(records),
    "success_threshold_rad": command.cfg.success_threshold,
    "any_success_rate": float((successes > 0).float().mean().item()),
    "mean_success_count": float(successes.float().mean().item()),
    "p50_success_count": float(successes.float().quantile(0.5).item()),
    "p90_success_count": float(successes.float().quantile(0.9).item()),
    "min_error_rad": {
      "mean": float(min_errors.mean().item()),
      "min": float(min_errors.min().item()),
      "max": float(min_errors.max().item()),
      **_percentiles(min_errors, (0.1, 0.25, 0.5, 0.75, 0.9)),
    },
    "final_error_rad": {
      "mean": float(final_errors.mean().item()),
      **_percentiles(final_errors, (0.1, 0.5, 0.9)),
    },
    "success_curve_min_error": success_curve,
    "mean_steps_to_first_success": (
      float(succ_steps.mean().item()) if succ_steps.numel() > 0 else None
    ),
    "termination_breakdown": term_breakdown,
  }

  _print_summary(summary, cfg.thresholds)

  if cfg.output_file:
    out = {
      "summary": summary,
      "episodes": [asdict(r) for r in records],
      "trajectories": trajectory_log,
    }
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
      json.dump(out, f, indent=2)
    print(f"[INFO] Wrote {output_path}")

  env.close()
  return summary


def _print_summary(summary: dict, thresholds: tuple[float, ...]) -> None:
  pad = 36
  print("\n" + "=" * 70)
  print("Reorientation evaluation summary")
  print("=" * 70)
  print(f"{'task':<{pad}}{summary['task_id']}")
  print(f"{'checkpoint':<{pad}}{summary['checkpoint']}")
  print(f"{'episodes':<{pad}}{summary['num_episodes']}")
  print(
    f"{'success_threshold (rad / deg)':<{pad}}"
    f"{summary['success_threshold_rad']:.3f} / "
    f"{math.degrees(summary['success_threshold_rad']):.2f}"
  )
  print("-" * 70)
  print(
    f"{'any-success rate':<{pad}}{summary['any_success_rate']:.3f}  "
    "(fraction of episodes with ≥1 goal reached)"
  )
  print(
    f"{'mean goals/episode':<{pad}}{summary['mean_success_count']:.2f}  "
    f"(p50={summary['p50_success_count']:.0f}, p90={summary['p90_success_count']:.0f})"
  )
  steps_str = (
    f"{summary['mean_steps_to_first_success']:.1f}"
    if summary["mean_steps_to_first_success"] is not None
    else "n/a (no episode succeeded)"
  )
  print(f"{'mean steps to first success':<{pad}}{steps_str}")
  print("-" * 70)
  print("Min orientation error per episode (radians):")
  m = summary["min_error_rad"]
  print(
    f"  mean={m['mean']:.3f}  p10={m['p10']:.3f}  p25={m['p25']:.3f}  "
    f"p50={m['p50']:.3f}  p75={m['p75']:.3f}  p90={m['p90']:.3f}"
  )
  print("Success curve (fraction of episodes whose min error ≤ threshold):")
  for t in thresholds:
    frac = summary["success_curve_min_error"][f"min_err<={t:.2f}"]
    bar = "█" * int(round(frac * 40))
    print(f"  ≤ {t:.2f} rad ({math.degrees(t):5.1f}°)  {frac:5.3f}  {bar}")
  print("-" * 70)
  print("Termination breakdown:")
  for name, frac in summary["termination_breakdown"].items():
    if frac > 0:
      print(f"  {name:<24}{frac:.3f}")
  print("=" * 70)


def main() -> None:
  import mjlab  # noqa: F401  (exposes mjlab.TYRO_FLAGS below)
  import mjlab.tasks  # noqa: F401  (populates the task registry)

  reorient_tasks = [t for t in list_tasks() if "Reorient" in t]
  if not reorient_tasks:
    print("No reorientation tasks found in the registry.")
    sys.exit(1)

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(reorient_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    EvaluateConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_evaluate(chosen_task, args)


if __name__ == "__main__":
  main()
