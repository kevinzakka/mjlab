"""Deterministic batch eval of a Sharpa cube-reorientation policy on CPU MuJoCo.

Every trial starts from the *same* settled cube-on-palm state (derived once by
letting the cube settle in the cradle, then snapshotted) and is given *one* fixed
goal orientation. The trial succeeds if the cube reaches and holds the goal for
``success_hold_steps`` consecutive control steps; there are no chained
reorientations. The N goals are sampled from a fixed seed, so the whole eval is
reproducible. Trials are independent, so they run across a process pool.

Alongside success rate it reports the "red flag" diagnostics that predict transfer
trouble: actuator torque headroom, intra-hand self-contact force, contact
penetration, grasp force, and action rate.

Usage:
  uv run python -m mjlab.tasks.reorient.scripts.sim2sim_eval \
    --wandb-run-path entity/project/run_id --n-goals 100 --n-workers 8
"""

from __future__ import annotations

import dataclasses
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import tyro

import mjlab.tasks.reorient.scripts.sim2sim_core as core

# Per-worker singletons, populated by _init_worker in each process.
_W: dict = {}


def _init_worker(
  onnx_path,
  drop_height,
  start_qpos,
  start_qvel,
  max_steps,
  mesh_collision,
  inverted,
  gravity,
) -> None:
  import mujoco

  model, cfg = core.build_model(
    use_mesh_collisions=mesh_collision, inverted=inverted, gravity=gravity
  )
  policy = core.load_policy(Path(onnx_path), n_threads=1)
  index = core.build_index(model, policy)
  params = core.build_params(
    cfg, model, index, policy, drop_height=drop_height, inverted=inverted
  )
  _W.update(
    model=model,
    data=mujoco.MjData(model),
    policy=policy,
    index=index,
    params=params,
    start_qpos=start_qpos,
    start_qvel=start_qvel,
    max_steps=max_steps,
  )


def _run_goal(item: tuple[int, np.ndarray]) -> core.TrialResult:
  goal_index, goal_quat = item
  return core.rollout(
    _W["model"],
    _W["data"],
    _W["policy"],
    _W["index"],
    _W["params"],
    _W["start_qpos"],
    _W["start_qvel"],
    goal_quat,
    goal_index,
    _W["max_steps"],
  )


def _pct(values: np.ndarray, q: float) -> float:
  return float(np.percentile(values, q)) if len(values) else float("nan")


def _print_report(results: list[core.TrialResult], step_dt: float) -> None:
  n = len(results)
  succ = [r for r in results if r.success]
  dropped = sum(r.dropped for r in results)
  print("\n" + "=" * 64)
  print(f"Deterministic eval: {n} goals")
  print("=" * 64)
  print(f"  success rate : {len(succ) / n:6.1%}  ({len(succ)}/{n})")
  print(f"  drop rate    : {dropped / n:6.1%}  ({dropped}/{n})")
  if succ:
    t = np.array([r.steps_to_success for r in succ]) * step_dt
    print(
      f"  time-to-hold : median {np.median(t):.2f}s  mean {t.mean():.2f}s  "
      f"p90 {_pct(t, 90):.2f}s  (successes only)"
    )
  print(
    f"  final error  : median {np.median([r.final_error for r in results]):.3f} rad"
  )

  print("\n  red-flag diagnostics (per-trial max, aggregated over goals):")
  cols = [
    ("torque headroom |tau|/tau_max", [r.max_torque_frac for r in results], ""),
    ("self-contact force (N)", [r.max_self_force for r in results], ""),
    ("grasp force (N)", [r.max_grasp_force for r in results], ""),
    ("penetration (mm)", [r.max_penetration * 1e3 for r in results], ""),
    ("action rate", [r.max_action_rate for r in results], ""),
    ("min cube z (m)", [r.min_cube_z for r in results], ""),
  ]
  print(f"    {'metric':<34}{'median':>10}{'p95':>10}{'max':>10}")
  for name, vals, _ in cols:
    v = np.array(vals)
    print(f"    {name:<34}{np.median(v):>10.3f}{_pct(v, 95):>10.3f}{v.max():>10.3f}")

  # Joint violations: a joint driven into its hard stop is a hardware red flag. The sim
  # absorbs it via the limit constraint, so it should be ~0 (a sub-degree baseline is
  # just the caged home grasp, whose fingers rest right at their limits).
  n_pos = sum(r.max_pos_violation > np.radians(1.0) for r in results)
  print("\n  joint violations:")
  print(f"    {'metric':<34}{'median':>10}{'p95':>10}{'max':>10}")
  vcols = [
    ("pos past hard limit (deg)", [np.degrees(r.max_pos_violation) for r in results]),
    ("peak joint speed (rad/s)", [r.max_joint_speed for r in results]),
  ]
  for name, vals in vcols:
    v = np.array(vals)
    print(f"    {name:<34}{np.median(v):>10.3f}{_pct(v, 95):>10.3f}{v.max():>10.3f}")
  print(f"    trials driving a joint >1 deg past a stop: {n_pos}/{n}")
  print("=" * 64)


def _dump_csv(results: list[core.TrialResult], path: Path) -> None:
  import csv

  fields = [f.name for f in dataclasses.fields(core.TrialResult)]
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in results:
      w.writerow(dataclasses.asdict(r))
  print(f"[INFO] per-trial results -> {path}")


@dataclasses.dataclass
class Args:
  wandb_run_path: str | None = None
  """W&B run path 'entity/project/run_id' to pull the policy from (or use --onnx-path)."""
  onnx_name: str | None = None
  onnx_path: str | None = None
  """Local .onnx to use directly, bypassing wandb (e.g. a prior cached export)."""
  checkpoint_name: str | None = None
  cache_dir: str = "logs/sim2sim"
  n_goals: int = 100
  """Number of fixed goal orientations to evaluate."""
  seed: int = 0
  """Seed for the goal set (eval is deterministic given this)."""
  n_workers: int = 0
  """Process pool size. 0 = os.cpu_count() - 1; 1 = serial (easier to debug)."""
  max_seconds: float = 8.0
  """Per-trial time budget in sim seconds before counting a timeout."""
  max_settle_steps: int = 200
  """Cap on control steps to settle the cube into the cradle (exits early once at rest)."""
  mesh_collision: bool = False
  """Use the real finger collision meshes instead of the primitive capsule fits."""
  inverted: bool = False
  """Palm-down mount: the hand is bolted upside down and the cube hangs (must be gripped)."""
  gravity: float | None = None
  """Override |gravity| (m/s^2). Match an inverted policy's training gravity if its
  gravity curriculum hasn't reached 9.81 yet (else it drops at full weight)."""
  drop_height: float = 0.04
  csv: str | None = None
  """Optional path to write per-trial results as CSV."""


def main(args: Args) -> None:
  import os

  model, cfg = core.build_model(
    use_mesh_collisions=args.mesh_collision,
    inverted=args.inverted,
    gravity=args.gravity,
  )
  onnx_path = core.resolve_onnx(
    args.wandb_run_path,
    args.onnx_name,
    args.checkpoint_name,
    Path(args.cache_dir),
    onnx_path=args.onnx_path,
  )
  policy = core.load_policy(onnx_path)
  index = core.build_index(model, policy)
  core.verify(policy, model, index, expected_scale=core.cfg_action_scale(cfg))
  params = core.build_params(
    cfg, model, index, policy, drop_height=args.drop_height, inverted=args.inverted
  )

  step_dt = params.decimation * model.opt.timestep
  max_steps = int(round(args.max_seconds / step_dt))

  print("[INFO] deriving settled start state...")
  start_qpos, start_qvel = core.derive_settled_state(
    model, index, params, max_settle_steps=args.max_settle_steps
  )
  goals = core.sample_goals(args.n_goals, args.seed)
  items = list(enumerate(goals))

  n_workers = args.n_workers or max(1, (os.cpu_count() or 2) - 1)
  print(
    f"[INFO] running {args.n_goals} goals, up to {max_steps} steps each "
    f"({args.max_seconds:.0f}s), on {n_workers} worker(s)..."
  )

  initargs = (
    str(onnx_path),
    args.drop_height,
    start_qpos,
    start_qvel,
    max_steps,
    args.mesh_collision,
    args.inverted,
    args.gravity,
  )
  if n_workers == 1:
    _init_worker(*initargs)
    results = [_run_goal(it) for it in items]
  else:
    with ProcessPoolExecutor(
      max_workers=n_workers, initializer=_init_worker, initargs=initargs
    ) as ex:
      results = list(ex.map(_run_goal, items))

  results.sort(key=lambda r: r.goal_index)
  _print_report(results, step_dt)
  if args.csv is not None:
    _dump_csv(results, Path(args.csv))


if __name__ == "__main__":
  main(tyro.cli(Args))
