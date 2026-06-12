"""Realtime CPU-MuJoCo playback of a trained Sharpa cube-reorientation policy.

Runs the ONNX-exported policy against plain CPU ``mujoco`` (not ``mujoco_warp``) in
the native viewer, so the policy can be inspected in realtime on a laptop. The task
observation/action/goal logic is reimplemented in numpy (see ``sim2sim_core``).

Goal modes:
  * auto (default): a goal state machine samples a uniform-SO(3) goal and resamples
    on each successful hold, so the policy chains reorientations.
  * ``--manual``: you drive the goal yourself. Double-click the translucent goal cube
    (the ghost above the hand), then Ctrl + right-drag to rotate it; the policy tracks it.

Optional ``--rerun`` streams the red-flag diagnostics (goal error, torque headroom,
action rate, self-contact force, grasp force, penetration) as scalar time series.

Usage:
  uv run python -m mjlab.tasks.reorient.scripts.sim2sim_play \
    --wandb-run-path entity/project/run_id [--manual] [--rerun]
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import tyro

import mjlab.tasks.reorient.scripts.sim2sim_core as core


class _RerunLogger:
  """Streams scalar diagnostics to rerun (optional dependency)."""

  def __init__(self) -> None:
    try:
      import rerun as rr  # pyright: ignore[reportMissingImports]
    except ModuleNotFoundError as e:
      raise SystemExit(
        "rerun is not installed. Add it with: uv sync --group dev "
        "(it is in the dev group as 'rerun-sdk')."
      ) from e
    self.rr = rr
    rr.init("sim2sim_reorient", spawn=True)

  def log(self, step: int, metrics: dict[str, float]) -> None:
    self.rr.set_time("control_step", sequence=step)
    for key, value in metrics.items():
      self.rr.log(f"metrics/{key}", self.rr.Scalars(float(value)))


class Controller:
  """Per-control-step policy driver with substep decimation, for the native viewer.

  ``mjcb_control`` fires every physics substep; we recompute the action only on
  control-step boundaries (every ``decimation`` substeps) and hold ``ctrl`` between.
  """

  def __init__(
    self,
    model,
    policy: core.Policy,
    index: core.ModelIndex,
    params: core.TaskParams,
    *,
    manual: bool,
    rerun: _RerunLogger | None,
    seed: int,
    goal_spin_min: float = 0.0,
    goal_spin_max: float = 0.0,
  ) -> None:
    self.m = model
    self.policy = policy
    self.idx = index
    self.p = params
    self.manual = manual
    self.rerun = rerun
    self.rng = np.random.default_rng(seed)
    self.goal_spin = (goal_spin_min, goal_spin_max)  # rad/s, sampled at each reset

    self.substep = 0
    self.episode_step = 0
    self.last_action = np.zeros(policy.action_dim, np.float32)
    self.prev_action = np.zeros(policy.action_dim, np.float32)
    self.success_count = 0
    # Auto goal state machine (unused in manual mode).
    self.goal_quat = np.array([1.0, 0.0, 0.0, 0.0])
    self.hold = 0
    self.in_window = False
    self.window_timer = 0

  # --- reset (targeted writes; picked up on the next mj_step) ---
  def reset(self, data) -> None:
    idx = self.idx
    data.qvel[:] = 0.0
    data.qpos[idx.joint_qadr] = self.p.home_joint
    core.place_cube(data, idx, self.p, core.random_quat(self.rng))
    core.pin_hand(data, idx, self.p)
    data.ctrl[idx.ctrl_ids] = self.p.home_joint
    self.episode_step = 0
    self.last_action[:] = 0.0
    self.prev_action[:] = 0.0
    self.hold = 0
    self.in_window = False
    self.window_timer = 0
    if self.manual:
      # Ball-joint goal: keep the user's orientation, but (re)kick its spin.
      self._kick_goal_spin(data)
    else:
      # Mocap goal: a fresh teleported goal each reset.
      self.goal_quat = core.random_quat(self.rng)
      data.mocap_pos[idx.goal_mocapid] = self.p.hand_pos + self.p.ghost_offset
      data.mocap_quat[idx.goal_mocapid] = self.goal_quat

  def _kick_goal_spin(self, data) -> None:
    """Give the ball-joint goal a random-axis angular velocity sampled in [min, max]."""
    if self.idx.goal_dof_adr < 0 or self.goal_spin[1] <= 0.0:
      return
    speed = self.rng.uniform(*self.goal_spin)
    axis = self.rng.standard_normal(3)
    axis /= np.linalg.norm(axis) + 1e-9
    a = self.idx.goal_dof_adr
    data.qvel[a : a + 3] = axis * speed

  # --- mjcb_control ---
  def control_callback(self, model, data) -> None:
    del model
    if self.substep % self.p.decimation == 0:
      self._control_step(data)
    self.substep += 1

  def _control_step(self, data) -> None:
    idx = self.idx
    dropped = data.xpos[idx.cube_bid][2] < self.p.drop_height
    if dropped and self.episode_step >= self.p.drop_grace_steps:
      self.reset(data)
      data.ctrl[idx.ctrl_ids] = self.p.home_joint
      return  # kinematics stale after reset; act on the next boundary.

    if self.manual:
      # Goal is the ball-joint ghost's live orientation (spun and/or user-dragged).
      q = data.xquat[idx.goal_base_bid]
      self.goal_quat = q / (np.linalg.norm(q) + 1e-12)

    obs = core.assemble_obs(
      self.m,
      data,
      idx,
      self.p.home_joint,
      self.goal_quat,
      self.last_action,
      self.prev_action,
    )
    action = self.policy.act(obs)
    self.prev_action, self.last_action = self.last_action, action.astype(np.float32)

    if self.episode_step < self.p.warmup_steps:
      data.ctrl[idx.ctrl_ids] = self.p.home_joint
    else:
      data.ctrl[idx.ctrl_ids] = (
        data.qpos[idx.joint_qadr] + self.last_action * self.p.action_scale
      )

    metrics = core.compute_metrics(
      self.m, data, idx, self.goal_quat, self.last_action, self.prev_action
    )
    if not self.manual:
      # Mocap goal: advance the success state machine. The ball-joint goal needs no
      # bookkeeping (its position is anchored and its orientation evolves in physics).
      self._advance_goal(data, metrics["goal_error"])
    if self.rerun is not None:
      self.rerun.log(self.episode_step, metrics)
    self.episode_step += 1

  def _advance_goal(self, data, goal_error: float) -> None:
    """Auto goal state machine: hold within threshold, then resample (full SO(3))."""
    if not self.in_window:
      self.hold = self.hold + 1 if goal_error < self.p.success_threshold else 0
      if self.hold >= self.p.success_hold_steps:
        self.in_window = True
        self.window_timer = 0
        self.success_count += 1
        print(f"[success #{self.success_count}] err={goal_error:.3f} rad")
    else:
      self.window_timer += 1
      if self.window_timer >= 2 * self.p.success_hold_steps:
        self.goal_quat = core.random_quat(self.rng)
        self.in_window = False
        self.hold = 0
    idx = self.idx
    data.mocap_pos[idx.goal_mocapid] = self.p.hand_pos + self.p.ghost_offset
    data.mocap_quat[idx.goal_mocapid] = self.goal_quat


class EvalController:
  """Replays the deterministic eval in the viewer: each fixed goal, in sequence.

  Mirrors ``sim2sim_core.rollout`` semantics (same settled start, same seeded goals,
  reach-and-hold success) so what you watch matches the headless ``sim2sim_eval``
  numbers. Lingers on each outcome before advancing so successes are easy to see.
  """

  def __init__(
    self,
    model,
    policy: core.Policy,
    index: core.ModelIndex,
    params: core.TaskParams,
    *,
    goals: np.ndarray,
    start_qpos: np.ndarray,
    start_qvel: np.ndarray,
    max_steps: int,
    success_dwell: int,
    fail_dwell: int,
    rerun: _RerunLogger | None,
  ) -> None:
    self.m = model
    self.policy = policy
    self.idx = index
    self.p = params
    self.goals = goals
    self.start_qpos = start_qpos
    self.start_qvel = start_qvel
    self.max_steps = max_steps
    self.success_dwell = success_dwell
    self.fail_dwell = fail_dwell
    self.rerun = rerun

    self.substep = 0
    self.global_step = 0
    self.gi = 0
    self.results: list[tuple[bool, int, bool]] = []  # (success, steps, dropped)
    self.summarized = False
    self.last = np.zeros(policy.action_dim, np.float32)
    self.prev = np.zeros(policy.action_dim, np.float32)
    self.episode_step = 0
    self.hold = 0
    self.phase = "run"  # "run" | "dwell"
    self.dwell = 0

  def begin_trial(self, data) -> None:
    core.reset_to_start(data, self.idx, self.p, self.start_qpos, self.start_qvel)
    data.mocap_pos[self.idx.goal_mocapid] = self.p.hand_pos + self.p.ghost_offset
    data.mocap_quat[self.idx.goal_mocapid] = self.goals[self.gi]
    self.last[:] = 0.0
    self.prev[:] = 0.0
    self.episode_step = 0
    self.hold = 0
    self.phase = "run"

  def control_callback(self, model, data) -> None:
    del model
    if self.substep % self.p.decimation == 0:
      self._step(data)
    self.substep += 1

  def _apply_policy(self, data, goal: np.ndarray, allow_warmup: bool) -> dict:
    idx = self.idx
    obs = core.assemble_obs(
      self.m, data, idx, self.p.home_joint, goal, self.last, self.prev
    )
    self.prev, self.last = self.last, self.policy.act(obs).astype(np.float32)
    if allow_warmup and self.episode_step < self.p.warmup_steps:
      data.ctrl[idx.ctrl_ids] = self.p.home_joint
    else:
      data.ctrl[idx.ctrl_ids] = (
        data.qpos[idx.joint_qadr] + self.last * self.p.action_scale
      )
    metrics = core.compute_metrics(self.m, data, idx, goal, self.last, self.prev)
    if self.rerun is not None:
      self.rerun.log(self.global_step, metrics)
    self.global_step += 1
    return metrics

  def _step(self, data) -> None:
    goal = self.goals[self.gi]
    if self.phase == "dwell":
      self._apply_policy(data, goal, allow_warmup=False)  # keep holding the pose
      self.dwell -= 1
      if self.dwell <= 0:
        self.gi = (self.gi + 1) % len(self.goals)
        self.begin_trial(data)  # writes qpos; act on the next boundary
      return

    if (
      self.episode_step >= self.p.drop_grace_steps
      and data.xpos[self.idx.cube_bid][2] < self.p.drop_height
    ):
      self._finish(False, dropped=True)
      return
    m = self._apply_policy(data, goal, allow_warmup=True)
    if m["goal_error"] < self.p.success_threshold:
      self.hold += 1
      if self.hold >= self.p.success_hold_steps:
        self._finish(True, final_error=m["goal_error"])
        return
    else:
      self.hold = 0
    self.episode_step += 1
    if self.episode_step >= self.max_steps:
      self._finish(False, final_error=m["goal_error"])

  def _finish(
    self, success: bool, *, dropped: bool = False, final_error: float = 0.0
  ) -> None:
    self.results.append((success, self.episode_step if success else -1, dropped))
    label = "SUCCESS" if success else ("DROP" if dropped else "TIMEOUT")
    n_succ = sum(s for s, _, _ in self.results)
    print(
      f"[trial {self.gi + 1:>2}/{len(self.goals)}] {label:<7} "
      f"err={final_error:.3f} rad  |  {n_succ}/{len(self.results)} so far"
    )
    self.phase = "dwell"
    self.dwell = self.success_dwell if success else self.fail_dwell
    if len(self.results) == len(self.goals) and not self.summarized:
      self._summarize()
      self.summarized = True

  def _summarize(self) -> None:
    n = len(self.goals)
    n_succ = sum(s for s, _, _ in self.results)
    n_drop = sum(d for _, _, d in self.results)
    print(
      f"\n=== eval pass complete: {n_succ}/{n} success "
      f"({n_succ / n:.0%}), {n_drop} drops ===  (looping; Ctrl-C to stop)\n"
    )


@dataclasses.dataclass
class Args:
  wandb_run_path: str | None = None
  """W&B run path 'entity/project/run_id' to pull the policy from (or use --onnx-path)."""
  onnx_name: str | None = None
  """Specific .onnx filename in the run (default: newest)."""
  onnx_path: str | None = None
  """Local .onnx to use directly, bypassing wandb (e.g. a prior cached export)."""
  checkpoint_name: str | None = None
  """Specific .pt checkpoint to export from when the run has no .onnx (default: latest)."""
  cache_dir: str = "logs/sim2sim"
  """Directory to download / export the ONNX into."""
  manual: bool = False
  """Interactive ball-joint goal: drag it (Ctrl + right-drag) and/or spin it (--goal-spin-max)."""
  goal_damping: float = 2e-3
  """Ball-joint goal damping. Spin decay time constant ~= 1e-3 / this (so 2e-3 ~= 0.5 s)."""
  goal_spin_min: float = 0.0
  goal_spin_max: float = 0.0
  """Initial goal angular speed range (rad/s), sampled with a random axis each reset; 0 = static."""
  mesh_collision: bool = False
  """Use the real finger collision meshes instead of the primitive capsule fits."""
  inverted: bool = False
  """Palm-down mount: the hand is bolted upside down and the cube hangs (must be gripped)."""
  gravity: float | None = None
  """Override |gravity| (m/s^2). Match an inverted policy's training gravity if its
  gravity curriculum hasn't reached 9.81 yet (else it drops at full weight)."""
  eval: bool = False
  """Replay the deterministic eval (N fixed goals, one settled start) in the viewer."""
  n_goals: int = 10
  """Number of fixed goals to replay in --eval mode."""
  max_seconds: float = 8.0
  """Per-trial time budget (sim seconds) before a timeout, in --eval mode."""
  success_dwell: float = 1.5
  """Seconds to linger on a matched pose before the next goal (--eval)."""
  fail_dwell: float = 0.5
  """Seconds to linger after a drop/timeout before the next goal (--eval)."""
  rerun: bool = False
  """Stream red-flag diagnostics as scalar time series to rerun."""
  seed: int = 0
  no_viewer: bool = False
  """Run headless for a fixed number of control steps (smoke test)."""
  headless_steps: int = 2000
  drop_height: float = 0.04
  """Cube center z below this (m) triggers a reset."""


def main(args: Args) -> None:
  model, cfg = core.build_model(
    use_mesh_collisions=args.mesh_collision,
    goal_balljoint=args.manual,
    goal_damping=args.goal_damping,
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
  print(f"[INFO] ONNX obs_dim={policy.obs_dim} action_dim={policy.action_dim}")
  index = core.build_index(model, policy)
  core.verify(policy, model, index, expected_scale=core.cfg_action_scale(cfg))
  params = core.build_params(
    cfg, model, index, policy, drop_height=args.drop_height, inverted=args.inverted
  )
  rerun = _RerunLogger() if args.rerun else None

  step_dt = params.decimation * model.opt.timestep
  data = mujoco.MjData(model)

  if args.eval:
    print("[INFO] deriving settled start state...")
    start_qpos, start_qvel = core.derive_settled_state(model, index, params)
    goals = core.sample_goals(args.n_goals, args.seed)
    controller: Controller | EvalController = EvalController(
      model,
      policy,
      index,
      params,
      goals=goals,
      start_qpos=start_qpos,
      start_qvel=start_qvel,
      max_steps=int(round(args.max_seconds / step_dt)),
      success_dwell=int(round(args.success_dwell / step_dt)),
      fail_dwell=int(round(args.fail_dwell / step_dt)),
      rerun=rerun,
    )
    controller.begin_trial(data)
    print(
      f"[INFO] replaying {args.n_goals} deterministic eval goals "
      f"(seed {args.seed}); same start, one goal each."
    )
  else:
    controller = Controller(
      model,
      policy,
      index,
      params,
      manual=args.manual,
      rerun=rerun,
      seed=args.seed,
      goal_spin_min=args.goal_spin_min,
      goal_spin_max=args.goal_spin_max,
    )
    controller.reset(data)
    if args.manual:
      print(
        "[INFO] Goal is a damped ball joint. Ctrl + right-drag the translucent cube to "
        "rotate it; set --goal-spin-max to have it tumble and slow down on its own."
      )
    print(
      f"[INFO] control={1 / step_dt:.0f} Hz, warmup={params.warmup_steps} steps, "
      f"goal mode={'manual (ball joint)' if args.manual else 'auto'}"
    )
  mujoco.mj_forward(model, data)

  mujoco.set_mjcb_control(controller.control_callback)
  try:
    if args.no_viewer:
      for _ in range(args.headless_steps * params.decimation):
        mujoco.mj_step(model, data)
    else:
      mujoco.viewer.launch(model, data)
  finally:
    mujoco.set_mjcb_control(None)


if __name__ == "__main__":
  main(tyro.cli(Args))
