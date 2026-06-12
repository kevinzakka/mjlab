"""Shared core for the Sharpa cube-reorientation sim2sim harness.

CPU-MuJoCo reimplementation of the reorient task's observation, action, and
goal logic, driven by the ONNX-exported policy. Used by both the interactive
viewer (``sim2sim_play``) and the deterministic batch eval (``sim2sim_eval``).

Nothing that could drift is hardcoded: the compiled ``MjModel`` (geometry,
sites, gains, contact params, ``init_state`` keyframe) comes from the mjlab
scene compile; ordering/scale come from the ONNX metadata; task constants come
from the env cfg. Observation normalization lives inside the ONNX graph, so we
feed raw observations.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import Scene
from mjlab.tasks.reorient.config.sharpa.env_cfgs import (
  FINGERTIP_SITES,
  PALM_SITE,
  sharpa_reorient_cube_env_cfg,
)

# Actor observation term order. We assert the ONNX metadata matches this, so a
# config change that reorders/adds terms fails loudly instead of silently feeding
# the policy a scrambled observation.
EXPECTED_OBS_TERMS = (
  "joint_pos",
  "joint_vel",
  "cube_pos",
  "cube_ori",
  "cube_to_goal_ori",
  "cube_lin_vel",
  "cube_ang_vel",
  "fingertip_to_cube",
  "fingertip_to_palm",
  "actions",
  "prev_actions",
)


# --------------------------------------------------------------------------------------
# Quaternion helpers. Thin wrappers over MuJoCo's native ops (wxyz, Hamilton product),
# which match mjlab.utils.lab_api.math to ~1e-15 (cross-checked offline).
# --------------------------------------------------------------------------------------
def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  out = np.empty(4)
  mujoco.mju_mulQuat(out, a, b)
  return out


def quat_conj(q: np.ndarray) -> np.ndarray:
  out = np.empty(4)
  mujoco.mju_negQuat(out, q)  # conjugate (== inverse for unit quats)
  return out


# Unit quaternions only (xquat / goal quats are normalized), so inverse == conjugate.
quat_inv = quat_conj


def mat_from_quat(q: np.ndarray) -> np.ndarray:
  out = np.empty(9)
  mujoco.mju_quat2Mat(out, q)
  return out.reshape(3, 3)


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate vector(s) v by quaternion q. v: (3,) or (n, 3)."""
  return v @ mat_from_quat(q).T


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate vector(s) v by the inverse of q. v: (3,) or (n, 3)."""
  return v @ mat_from_quat(q)  # R^-1 @ v == v @ R for each row


def quat_to_6d(q: np.ndarray) -> np.ndarray:
  """First two columns of the rotation matrix, row-major: [m00,m01,m10,m11,m20,m21]."""
  return mat_from_quat(q)[:, :2].reshape(6)


def quat_error_magnitude(q1: np.ndarray, q2: np.ndarray) -> float:
  """Geodesic angle (radians) between two orientations."""
  diff = np.empty(3)
  mujoco.mju_subQuat(diff, q1, q2)  # q1 ⊖ q2 in the Lie algebra; |·| is the angle
  return float(np.linalg.norm(diff))


def random_quat(rng: np.random.Generator) -> np.ndarray:
  """Uniform random orientation (Shoemake), wxyz."""
  u1, u2, u3 = rng.random(3)
  return np.array(
    [
      np.sqrt(u1) * np.cos(2 * np.pi * u3),
      np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
      np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
      np.sqrt(u1) * np.sin(2 * np.pi * u3),
    ]
  )


def sample_goals(n: int, seed: int) -> np.ndarray:
  """N fixed uniform-SO(3) goal quaternions (deterministic for a given seed)."""
  rng = np.random.default_rng(seed)
  return np.stack([random_quat(rng) for _ in range(n)])


# --------------------------------------------------------------------------------------
# ONNX policy + metadata.
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class Policy:
  session: ort.InferenceSession
  input_name: str
  output_name: str
  obs_dim: int
  action_dim: int
  joint_names: list[str]
  observation_names: list[str]
  action_scale: np.ndarray  # (action_dim,)

  def act(self, obs: np.ndarray) -> np.ndarray:
    out = self.session.run(
      [self.output_name], {self.input_name: obs[None].astype(np.float32)}
    )
    return np.asarray(out[0])[0]


def _parse_csv(meta: dict[str, str], key: str) -> list[str]:
  if key not in meta:
    raise KeyError(
      f"ONNX metadata is missing '{key}'. Was this exported by "
      f"ManipulationOnPolicyRunner? Keys present: {sorted(meta)}"
    )
  return meta[key].split(",")


def load_policy(onnx_path: Path, *, n_threads: int | None = None) -> Policy:
  opts = ort.SessionOptions()  # ty: ignore[possibly-missing-attribute]
  if n_threads is not None:
    # Keep each worker single-threaded so a process pool doesn't oversubscribe cores.
    opts.intra_op_num_threads = n_threads
    opts.inter_op_num_threads = n_threads
  session = ort.InferenceSession(
    str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
  )
  meta = session.get_modelmeta().custom_metadata_map
  inp = session.get_inputs()[0]
  out = session.get_outputs()[0]

  joint_names = _parse_csv(meta, "joint_names")
  observation_names = _parse_csv(meta, "observation_names")
  scale_vals = [float(x) for x in _parse_csv(meta, "action_scale")]
  action_dim = int(out.shape[-1])
  if len(scale_vals) == 1:
    scale_vals = scale_vals * action_dim
  return Policy(
    session=session,
    input_name=inp.name,
    output_name=out.name,
    obs_dim=int(inp.shape[-1]),
    action_dim=action_dim,
    joint_names=joint_names,
    observation_names=observation_names,
    action_scale=np.asarray(scale_vals, dtype=np.float32),
  )


def _pick_file(files: list, name: str | None, kind: str):
  """Choose one W&B run file: by exact name, else the only one, else the newest."""
  if name is not None:
    matches = [f for f in files if f.name == name]
    if not matches:
      raise FileNotFoundError(
        f"'{name}' not found. Available: {[f.name for f in files]}"
      )
    return matches[0]
  if len(files) == 1:
    return files[0]
  chosen = max(files, key=lambda f: f.updated_at)
  print(f"[INFO] Multiple {kind} files; picking newest: {chosen.name}")
  return chosen


def resolve_onnx(
  run_path: str | None,
  onnx_name: str | None,
  checkpoint_name: str | None,
  cache_dir: Path,
  onnx_path: str | None = None,
) -> Path:
  """Return a metadata-tagged policy .onnx for the run.

  Precedence: an explicit local ``onnx_path`` (fully offline); else the run's uploaded
  ``.onnx`` if present; else export one on the fly from a ``.pt`` checkpoint (older
  reorient runs lack ONNX because of a now-fixed export bug).
  """
  if onnx_path is not None:
    p = Path(onnx_path)
    if not p.exists():
      raise FileNotFoundError(f"--onnx-path {p} does not exist.")
    print(f"[INFO] Using local policy: {p}")
    return p
  if run_path is None:
    raise ValueError("Provide --wandb-run-path or --onnx-path.")

  import wandb

  cache_dir.mkdir(parents=True, exist_ok=True)
  run = wandb.Api().run(run_path)
  onnx_files = [f for f in run.files() if f.name.endswith(".onnx")]
  if onnx_files:
    chosen = _pick_file(onnx_files, onnx_name, "onnx")
    chosen.download(root=str(cache_dir), replace=True)
    path = cache_dir / chosen.name
    print(f"[INFO] Downloaded policy: {path}")
    return path

  print("[INFO] No .onnx in run; exporting one on the fly from a .pt checkpoint...")
  return export_onnx_from_checkpoint(run, run_path, checkpoint_name, cache_dir)


def _resolve_checkpoint_file(run, checkpoint_name: str | None):
  """Pick a model_<N>.pt run file via the plain files() listing.

  We deliberately avoid mjlab's ``get_wandb_checkpoint_path``, which uses
  ``run.files(pattern="model_%.pt")`` -- that server-side LIKE query crashes inside
  wandb's pagination (``project.run`` comes back None) on some runs/wandb versions. The
  unfiltered ``run.files()`` is reliable.
  """
  pts = [f for f in run.files() if f.name.endswith(".pt") and "model_" in f.name]
  if not pts:
    raise FileNotFoundError(f"No model_*.pt checkpoints in run {run.id}.")
  if checkpoint_name is not None:
    chosen = next((f for f in pts if f.name == checkpoint_name), None)
    if chosen is None:
      raise FileNotFoundError(
        f"'{checkpoint_name}' not in run. Available: {sorted(f.name for f in pts)}"
      )
    return chosen

  def step_of(name: str) -> int:
    try:
      return int(name.split("model_")[1].split(".pt")[0])
    except (IndexError, ValueError):
      return -1

  return max(pts, key=lambda f: step_of(f.name))


def export_onnx_from_checkpoint(
  run, run_path: str, checkpoint_name: str | None, cache_dir: Path
) -> Path:
  """Build the env + runner on CPU, load a .pt checkpoint, and export a tagged ONNX.

  Mirrors ``ManipulationOnPolicyRunner.save()``'s export block, but on demand. Heavy
  imports (torch / env / warp) are local so the realtime path never pays for them. The
  resulting ONNX is cached under ``cache_dir`` and reused on later runs (so the wandb
  round-trip and env build only happen once per checkpoint).
  """
  ckpt = _resolve_checkpoint_file(run, checkpoint_name)
  onnx_path = cache_dir / f"{run.id}_{Path(ckpt.name).stem}.onnx"
  if onnx_path.exists():
    print(f"[INFO] Reusing cached export: {onnx_path}")
    return onnx_path

  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
  from mjlab.tasks.reorient.config.sharpa.rl_cfg import (
    sharpa_reorient_cube_ppo_runner_cfg,
  )
  from mjlab.tasks.reorient.rl import ManipulationOnPolicyRunner

  ckpt_dir = cache_dir / "checkpoints" / run.id
  pt_path = ckpt_dir / ckpt.name
  if pt_path.exists():
    print(f"[INFO] Checkpoint {ckpt.name} (cached); building env to export...")
  else:
    print(f"[INFO] Checkpoint {ckpt.name} (downloading); building env to export...")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt.download(root=str(ckpt_dir), replace=True)

  agent_cfg = sharpa_reorient_cube_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=sharpa_reorient_cube_env_cfg(play=True), device="cpu")
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = ManipulationOnPolicyRunner(wrapped, asdict(agent_cfg), device="cpu")
    runner.load(str(pt_path), load_cfg={"actor": True}, strict=True, map_location="cpu")
    runner.export_policy_to_onnx(str(cache_dir), onnx_path.name)
    attach_metadata_to_onnx(str(onnx_path), get_base_metadata(env, run_path))
  finally:
    env.close()
  print(f"[INFO] Exported ONNX: {onnx_path}")
  return onnx_path


# --------------------------------------------------------------------------------------
# Compiled model + index.
# --------------------------------------------------------------------------------------
# Rotor inertia for the ball-joint goal. Fixes the effective rotational inertia so the
# damping value behaves intuitively (decay time ~ armature / damping) regardless of the
# tiny ghost-geom mass, and stabilizes the integration.
_GOAL_ARMATURE = 1e-3


def _make_goal_balljoint(spec, damping: float, anchor_pos) -> None:
  """Turn the kinematic mocap goal marker into a damped ball-joint body.

  The auto-wrapped marker is a mocap body (no DOFs), so it can only be teleported, not
  spun or dragged physically. For an interactive/dynamic goal we give it a 3-DOF ball
  joint with damping, anchored at ``anchor_pos`` (the ghost world position for the
  mounting), with the box geom centered on the joint so gravity exerts no torque (it
  holds whatever orientation it is left at and only loses speed to damping). Adding the
  joint adds 4 qpos, so we extend the ``init_state`` keyframe with the joint's identity
  quaternion (appended last, since the goal marker is the last entity).
  """
  gb = spec.body("goal_marker/mocap_base")
  child = spec.body("goal_marker/goal")
  g = child.geoms[0]
  gb.mocap = False
  gb.pos = [float(x) for x in anchor_pos]
  gb.add_joint(
    type=mujoco.mjtJoint.mjJNT_BALL,
    damping=damping,
    armature=_GOAL_ARMATURE,
    name="goal_ball",
  )
  gb.add_geom(
    type=g.type,
    size=g.size,
    material=g.material,
    rgba=g.rgba,
    contype=0,
    conaffinity=0,
    group=g.group,
    density=1000.0,
    name="goal_grab",
  )
  spec.delete(g)
  key = spec.keys[0]
  key.qpos = list(key.qpos) + [1.0, 0.0, 0.0, 0.0]


def build_model(
  use_mesh_collisions: bool = False,
  goal_balljoint: bool = False,
  goal_damping: float = 2e-3,
  inverted: bool = False,
  gravity: float | None = None,
) -> tuple[mujoco.MjModel, ManagerBasedRlEnvCfg]:
  """Compile the CPU model from the same env cfg used for training, with sim opts.

  ``use_mesh_collisions`` swaps the primitive finger collision fits for the real
  link/elastomer meshes. ``inverted`` mounts the hand palm-down (cube hangs).
  ``goal_balljoint`` replaces the kinematic mocap goal with a damped 3-DOF ball joint
  (so the goal can be dragged and/or spun); ``goal_damping`` sets how fast it loses speed.
  ``gravity`` overrides |g| (m/s^2, points -z); use it to match the strength an inverted
  policy was trained at when it used a gravity curriculum that has not yet reached 9.81.
  """
  cfg = sharpa_reorient_cube_env_cfg(
    play=True, use_mesh_collisions=use_mesh_collisions, inverted=inverted
  )
  scene = Scene(cfg.scene, device="cpu")
  if goal_balljoint:
    # Anchor the ghost where the env poses it: hand root position + the (world) ghost
    # offset, both read from the cfg so the inverted side-offset is honored.
    from mjlab.tasks.reorient.mdp import ReorientationCommandCfg

    goal_cfg = cfg.commands["goal"]
    assert isinstance(goal_cfg, ReorientationCommandCfg)
    hand_pos = np.asarray(cfg.scene.entities["robot"].init_state.pos, dtype=float)
    anchor = hand_pos + np.asarray(goal_cfg.viz.offset, dtype=float)
    _make_goal_balljoint(scene.spec, goal_damping, anchor)
  model = scene.compile()
  cfg.sim.mujoco.apply(model)  # timestep, elliptic cone, impratio, solver iters.
  if gravity is not None:
    model.opt.gravity[:] = (0.0, 0.0, -gravity)
  return model, cfg


@dataclasses.dataclass
class ModelIndex:
  joint_qadr: np.ndarray  # (nu,) qpos address per actuated joint, policy order
  joint_vadr: np.ndarray  # (nu,) qvel address per actuated joint, policy order
  ctrl_ids: np.ndarray  # (nu,) actuator id per actuated joint, policy order
  base_bid: int  # hand base body (root_link frame)
  cube_bid: int
  cube_qadr: int  # cube free-joint qpos start
  cube_vadr: int  # cube free-joint qvel start (6 dofs: lin 3 + ang 3)
  wrist_sid: int
  fingertip_sids: np.ndarray  # (5,)
  hand_mocapid: int
  goal_mocapid: int  # -1 when the goal is a ball joint instead of a mocap
  goal_base_bid: int  # goal-marker root body (read its xquat as the goal orientation)
  goal_dof_adr: int  # goal ball-joint qvel start (3 dofs), or -1 for a mocap goal
  joint_lower: np.ndarray  # (nu,) per-joint lower limit (jnt_range), policy order
  joint_upper: np.ndarray  # (nu,) per-joint upper limit (jnt_range), policy order
  home_qpos: np.ndarray  # (nq,) from init_state keyframe
  hand_body_ids: frozenset  # robot subtree body ids (for self-contact classification)


def _name2id(model, objtype, name: str) -> int:
  i = mujoco.mj_name2id(model, objtype, name)
  if i < 0:
    raise KeyError(f"'{name}' (type {objtype}) not found in compiled model.")
  return i


def build_index(model, policy: Policy) -> ModelIndex:
  obj = mujoco.mjtObj
  # Actuated joints, in the policy's joint order (sourced from ONNX metadata).
  joint_qadr, joint_vadr, ctrl_ids, jlo, jhi = [], [], [], [], []
  for jname in policy.joint_names:
    jid = _name2id(model, obj.mjOBJ_JOINT, f"robot/{jname}")
    joint_qadr.append(model.jnt_qposadr[jid])
    joint_vadr.append(model.jnt_dofadr[jid])
    # Hard mechanical limit (inf if unlimited) for joint-violation diagnostics.
    lo, hi = model.jnt_range[jid] if model.jnt_limited[jid] else (-np.inf, np.inf)
    jlo.append(lo)
    jhi.append(hi)
    aid = _name2id(model, obj.mjOBJ_ACTUATOR, f"robot/{jname}")
    # Confirm the actuator actually drives this joint (trntype JOINT, trnid == jid).
    assert model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
    assert model.actuator_trnid[aid, 0] == jid, f"actuator/joint mismatch for {jname}"
    ctrl_ids.append(aid)

  base_bid = _name2id(model, obj.mjOBJ_BODY, "robot/mocap_base")
  hand_mocapid = int(model.body_mocapid[base_bid])
  assert hand_mocapid >= 0, "robot/mocap_base is not a mocap body"

  cube_jid = _name2id(model, obj.mjOBJ_JOINT, "cube/cube_joint")
  assert model.jnt_type[cube_jid] == mujoco.mjtJoint.mjJNT_FREE
  cube_bid = int(model.jnt_bodyid[cube_jid])
  cube_qadr = int(model.jnt_qposadr[cube_jid])
  cube_vadr = int(model.jnt_dofadr[cube_jid])

  goal_base_bid = _name2id(model, obj.mjOBJ_BODY, "goal_marker/mocap_base")
  goal_mocapid = int(model.body_mocapid[goal_base_bid])
  goal_dof_adr = -1
  for j in range(model.njnt):
    if int(model.jnt_bodyid[j]) == goal_base_bid and (
      model.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL
    ):
      goal_dof_adr = int(model.jnt_dofadr[j])
      break

  wrist_sid = _name2id(model, obj.mjOBJ_SITE, f"robot/{PALM_SITE}")
  fingertip_sids = np.array(
    [_name2id(model, obj.mjOBJ_SITE, f"robot/{s}") for s in FINGERTIP_SITES]
  )

  key_id = _name2id(model, obj.mjOBJ_KEY, "init_state")
  home_qpos = model.key_qpos[key_id].copy()

  hand_body_ids = frozenset(
    i
    for i in range(model.nbody)
    if (mujoco.mj_id2name(model, obj.mjOBJ_BODY, i) or "").startswith("robot/")
  )

  return ModelIndex(
    joint_qadr=np.array(joint_qadr),
    joint_vadr=np.array(joint_vadr),
    ctrl_ids=np.array(ctrl_ids),
    base_bid=base_bid,
    cube_bid=cube_bid,
    cube_qadr=cube_qadr,
    cube_vadr=cube_vadr,
    wrist_sid=wrist_sid,
    fingertip_sids=fingertip_sids,
    hand_mocapid=hand_mocapid,
    goal_mocapid=goal_mocapid,
    goal_base_bid=goal_base_bid,
    goal_dof_adr=goal_dof_adr,
    joint_lower=np.array(jlo),
    joint_upper=np.array(jhi),
    home_qpos=home_qpos,
    hand_body_ids=hand_body_ids,
  )


# --------------------------------------------------------------------------------------
# Task parameters (derived once from cfg + model + policy; picklable for workers).
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class TaskParams:
  decimation: int
  warmup_steps: int
  action_scale: np.ndarray  # (nu,)
  home_joint: np.ndarray  # (nu,)
  success_threshold: float
  success_hold_steps: int
  ghost_offset: np.ndarray  # (3,) world offset of the goal ghost from the hand root
  drop_height: float
  drop_grace_steps: int  # skip drop detection for this many steps after a reset
  hand_pos: np.ndarray  # (3,) hand mount position (palm-up vs inverted)
  hand_rot: np.ndarray  # (4,) hand mount orientation (wxyz)
  cradle_offset_b: np.ndarray  # (3,) cube cradle in the hand body frame (lifted)
  inverted: bool  # palm-down mounting: the cube hangs, no settling-by-rest


def cfg_action_scale(cfg) -> float:
  """The scalar action scale from the cfg (cross-checked against the ONNX metadata)."""
  from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg

  action_cfg = cfg.actions["joint_pos"]
  assert isinstance(action_cfg, RelativeJointPositionActionCfg)
  assert isinstance(action_cfg.scale, (int, float)), "expected a scalar action scale"
  return float(action_cfg.scale)


def build_params(
  cfg, model, index: ModelIndex, policy: Policy, drop_height: float, inverted: bool
):
  from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
  from mjlab.tasks.reorient.mdp import ReorientationCommandCfg

  action_cfg = cfg.actions["joint_pos"]
  goal_cfg = cfg.commands["goal"]
  assert isinstance(action_cfg, RelativeJointPositionActionCfg)
  assert isinstance(goal_cfg, ReorientationCommandCfg)
  step_dt = cfg.decimation * model.opt.timestep
  # Mount + cube placement, read from the cfg so it is correct for any variant
  # (palm-up or inverted) without hardcoding orientation-specific constants.
  robot_init = cfg.scene.entities["robot"].init_state
  cradle_b = cfg.events["reset_hand_and_cube"].params["cradle_offset_b"]
  grace = cfg.terminations["cube_dropped"].params.get("grace_steps", 0)
  return TaskParams(
    decimation=cfg.decimation,
    warmup_steps=int(round(action_cfg.warmup_time_s / step_dt)),
    action_scale=policy.action_scale,
    home_joint=index.home_qpos[index.joint_qadr].copy(),
    success_threshold=goal_cfg.success_threshold,
    success_hold_steps=goal_cfg.success_hold_steps,
    ghost_offset=np.asarray(goal_cfg.viz.offset),
    drop_height=drop_height,
    drop_grace_steps=int(grace),
    hand_pos=np.asarray(robot_init.pos, dtype=float),
    hand_rot=np.asarray(robot_init.rot, dtype=float),
    cradle_offset_b=np.asarray(cradle_b, dtype=float),
    inverted=inverted,
  )


# --------------------------------------------------------------------------------------
# Verification: fail loudly on any policy/reimplementation mismatch.
# --------------------------------------------------------------------------------------
def verify(policy: Policy, model, index: ModelIndex, expected_scale: float) -> None:
  errors = []
  if list(policy.observation_names) != list(EXPECTED_OBS_TERMS):
    errors.append(
      f"observation term order mismatch:\n  onnx: {policy.observation_names}\n"
      f"  expected: {list(EXPECTED_OBS_TERMS)}"
    )
  n_act = policy.action_dim
  n_tips = len(index.fingertip_sids)
  term_dims = {
    "joint_pos": n_act,
    "joint_vel": n_act,
    "cube_pos": 3,
    "cube_ori": 6,
    "cube_to_goal_ori": 6,
    "cube_lin_vel": 3,
    "cube_ang_vel": 3,
    "fingertip_to_cube": 3 * n_tips,
    "fingertip_to_palm": 3 * n_tips,
    "actions": n_act,
    "prev_actions": n_act,
  }
  expected_obs_dim = sum(term_dims[t] for t in EXPECTED_OBS_TERMS)
  if expected_obs_dim != policy.obs_dim:
    errors.append(
      f"obs dim mismatch: onnx expects {policy.obs_dim}, "
      f"reimplementation builds {expected_obs_dim}. Per-term: {term_dims}"
    )
  if policy.action_dim != model.nu:
    errors.append(f"action dim {policy.action_dim} != model.nu {model.nu}")
  if len(policy.joint_names) != model.nu:
    errors.append(f"{len(policy.joint_names)} joint names != model.nu {model.nu}")
  if not np.allclose(policy.action_scale, expected_scale, atol=1e-3):
    errors.append(f"action_scale {policy.action_scale} != cfg scale {expected_scale}")
  if errors:
    raise RuntimeError(
      "Policy/environment verification FAILED:\n- " + "\n- ".join(errors)
    )
  print(
    f"[OK] verified: obs_dim={policy.obs_dim}, action_dim={policy.action_dim}, "
    f"action_scale={policy.action_scale[0]:.3f}, {len(EXPECTED_OBS_TERMS)} obs terms"
  )


# --------------------------------------------------------------------------------------
# Observation assembly (reimplements the actor observation group, in order).
# --------------------------------------------------------------------------------------
def assemble_obs(
  model,
  data,
  index: ModelIndex,
  home_joint: np.ndarray,
  goal_quat: np.ndarray,
  last_action: np.ndarray,
  prev_action: np.ndarray,
) -> np.ndarray:
  base_quat = data.xquat[index.base_bid]
  base_pos = data.xpos[index.base_bid]
  cube_pos = data.xpos[index.cube_bid]
  cube_quat = data.xquat[index.cube_bid]
  wrist_pos = data.site_xpos[index.wrist_sid]
  tip_pos = data.site_xpos[index.fingertip_sids]  # (5, 3)

  # Cube velocity at the body origin in world frame (matches mjlab cvel-based vel).
  vel6 = np.zeros(6)
  mujoco.mj_objectVelocity(
    model, data, mujoco.mjtObj.mjOBJ_BODY, index.cube_bid, vel6, 0
  )
  cube_ang_w, cube_lin_w = vel6[0:3], vel6[3:6]

  base_inv = quat_inv(base_quat)
  goal_in_base = quat_mul(base_inv, goal_quat)
  cube_in_base = quat_mul(base_inv, cube_quat)
  return np.concatenate(
    [
      data.qpos[index.joint_qadr] - home_joint,  # joint_pos (rel)
      data.qvel[index.joint_vadr],  # joint_vel (rel; default vel = 0)
      quat_apply_inverse(base_quat, cube_pos - wrist_pos),  # cube_pos
      quat_to_6d(quat_mul(base_inv, cube_quat)),  # cube_ori
      quat_to_6d(quat_mul(goal_in_base, quat_conj(cube_in_base))),  # cube_to_goal_ori
      quat_apply_inverse(base_quat, cube_lin_w),  # cube_lin_vel
      quat_apply_inverse(base_quat, cube_ang_w),  # cube_ang_vel
      quat_apply_inverse(base_quat, tip_pos - cube_pos).reshape(
        -1
      ),  # fingertip_to_cube
      quat_apply_inverse(base_quat, tip_pos - base_pos).reshape(
        -1
      ),  # fingertip_to_palm
      last_action,
      prev_action,
    ]
  ).astype(np.float32)


# --------------------------------------------------------------------------------------
# Diagnostics ("red flags" that predict transfer trouble).
# --------------------------------------------------------------------------------------
def _contact_metrics(
  model, data, index: ModelIndex
) -> tuple[float, float, float, float]:
  """(max self-contact force, total self-contact force, grasp force, max penetration).

  Self-contact = intra-hand (both geoms in the robot subtree). Grasp = a hand geom
  against the cube. Penetration depth = -contact.dist (positive when overlapping).
  """
  self_max = self_sum = grasp_sum = max_pen = 0.0
  buf = np.zeros(6)
  for i in range(data.ncon):
    c = data.contact[i]
    max_pen = max(max_pen, -float(c.dist))
    mujoco.mj_contactForce(model, data, i, buf)
    fn = abs(float(buf[0]))  # normal force in the contact frame
    b1 = int(model.geom_bodyid[c.geom1])
    b2 = int(model.geom_bodyid[c.geom2])
    h1, h2 = b1 in index.hand_body_ids, b2 in index.hand_body_ids
    if index.cube_bid in (b1, b2) and (h1 or h2):
      grasp_sum += fn
    elif h1 and h2:
      self_sum += fn
      self_max = max(self_max, fn)
  return self_max, self_sum, grasp_sum, max_pen


def compute_metrics(
  model,
  data,
  index: ModelIndex,
  goal_quat: np.ndarray,
  action: np.ndarray,
  prev_action: np.ndarray,
) -> dict[str, float]:
  tau = data.actuator_force[index.ctrl_ids]
  tau_max = model.actuator_forcerange[index.ctrl_ids, 1]  # symmetric (-eff, eff)
  with np.errstate(divide="ignore", invalid="ignore"):
    tau_frac = np.where(tau_max > 0, np.abs(tau) / tau_max, 0.0)
  self_max, self_sum, grasp, max_pen = _contact_metrics(model, data, index)

  # Joint position-limit violation: how far any joint went past its hard mechanical
  # limit (rad). The sim absorbs this via the limit constraint; on hardware it means a
  # joint driven into a hard stop. (We don't flag commanded-target overruns: with the
  # compliant position gains the target is effectively a torque command, so commanding
  # past a limit is intentional and bounded.)
  qpos = data.qpos[index.joint_qadr]
  pos_viol = np.maximum(
    np.maximum(qpos - index.joint_upper, index.joint_lower - qpos), 0.0
  )
  return {
    "goal_error": quat_error_magnitude(data.xquat[index.cube_bid], goal_quat),
    "torque_frac_max": float(tau_frac.max()),
    "action_rate": float(np.linalg.norm(action - prev_action)),
    "self_force_max": self_max,
    "self_force_sum": self_sum,
    "grasp_force": grasp,
    "max_penetration": max_pen,
    "cube_z": float(data.xpos[index.cube_bid][2]),
    "pos_limit_violation": float(pos_viol.max()),
    "joint_speed_max": float(np.abs(data.qvel[index.joint_vadr]).max()),
  }


# --------------------------------------------------------------------------------------
# Reset / start-state helpers.
# --------------------------------------------------------------------------------------
def place_cube(
  data, index: ModelIndex, params: TaskParams, cube_quat: np.ndarray
) -> None:
  """Cradle the cube in the hand body frame (mount-agnostic) with the given orientation.

  Mirrors the training reset: ``hand_pos + R(hand_rot) @ cradle_offset_b`` (the cradle is
  body-frame and lifted along the palm normal), so the cube nestles in the same spot of
  the hand whether the hand is palm-up or inverted.
  """
  cube_pos = params.hand_pos + quat_apply(params.hand_rot, params.cradle_offset_b)
  data.qpos[index.cube_qadr : index.cube_qadr + 3] = cube_pos
  data.qpos[index.cube_qadr + 3 : index.cube_qadr + 7] = cube_quat


def pin_hand(data, index: ModelIndex, params: TaskParams) -> None:
  """Pin the hand mocap at its mount pose (never randomized)."""
  data.mocap_pos[index.hand_mocapid] = params.hand_pos
  data.mocap_quat[index.hand_mocapid] = params.hand_rot


def reset_to_start(
  data,
  index: ModelIndex,
  params: TaskParams,
  start_qpos: np.ndarray,
  start_qvel: np.ndarray,
) -> None:
  data.qpos[:] = start_qpos
  data.qvel[:] = start_qvel
  pin_hand(data, index, params)
  data.ctrl[index.ctrl_ids] = params.home_joint


def derive_settled_state(
  model,
  index: ModelIndex,
  params: TaskParams,
  max_settle_steps: int = 200,
  lin_tol: float = 2e-3,
  ang_tol: float = 2e-2,
  cube_start_quat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Derive the canonical start state: cube cradled in the home grasp, snapshotted.

  Palm-up: the cube rests in the cup, so we step (home grasp held) until it comes to
  rest and snapshot that. Inverted: a held home grasp can't hold the hanging cube (the
  policy has to actively grip from step 0, which is why that variant has no warmup), so
  we skip settling and snapshot the lifted-cradle placement directly. Either way the
  start is deterministic and computed once.
  """
  if cube_start_quat is None:
    cube_start_quat = np.array([1.0, 0.0, 0.0, 0.0])
  data = mujoco.MjData(model)
  data.qpos[index.joint_qadr] = params.home_joint
  place_cube(data, index, params, cube_start_quat)
  pin_hand(data, index, params)
  data.ctrl[index.ctrl_ids] = params.home_joint
  mujoco.mj_forward(model, data)

  if params.inverted:
    print("[INFO] inverted mount: skipping settle (policy grips from step 0)")
    return data.qpos.copy(), data.qvel.copy()

  vadr = index.cube_vadr
  at_rest = 0
  needed = 3  # consecutive control steps below tolerance
  for step in range(max_settle_steps):
    for _ in range(params.decimation):
      mujoco.mj_step(model, data)
    lin = float(np.linalg.norm(data.qvel[vadr : vadr + 3]))
    ang = float(np.linalg.norm(data.qvel[vadr + 3 : vadr + 6]))
    at_rest = at_rest + 1 if (lin < lin_tol and ang < ang_tol) else 0
    if at_rest >= needed:
      print(f"[INFO] cube settled after {step + 1} control steps")
      break
  else:
    print(f"[INFO] cube not fully at rest after {max_settle_steps} steps (using as-is)")
  return data.qpos.copy(), data.qvel.copy()


# --------------------------------------------------------------------------------------
# Deterministic single-goal rollout (used by the batch eval).
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class TrialResult:
  goal_index: int
  success: bool
  steps_to_success: int  # -1 if never
  dropped: bool
  n_steps: int
  final_error: float
  min_cube_z: float
  max_torque_frac: float
  max_self_force: float
  max_grasp_force: float
  max_penetration: float
  max_action_rate: float
  max_pos_violation: float  # rad any joint went past its hard limit
  max_joint_speed: float  # rad/s, peak joint speed


def rollout(
  model,
  data,
  policy: Policy,
  index: ModelIndex,
  params: TaskParams,
  start_qpos: np.ndarray,
  start_qvel: np.ndarray,
  goal_quat: np.ndarray,
  goal_index: int,
  max_steps: int,
) -> TrialResult:
  """One deterministic reach-and-hold trial from the settled start to a fixed goal."""
  reset_to_start(data, index, params, start_qpos, start_qvel)
  mujoco.mj_forward(model, data)
  last = np.zeros(policy.action_dim, np.float32)
  prev = np.zeros(policy.action_dim, np.float32)

  hold = 0
  steps_to_success = -1
  dropped = False
  agg = dict(
    min_cube_z=np.inf,
    max_torque_frac=0.0,
    max_self_force=0.0,
    max_grasp_force=0.0,
    max_penetration=0.0,
    max_action_rate=0.0,
    max_pos_violation=0.0,
    max_joint_speed=0.0,
  )
  m = {"goal_error": float(np.pi)}
  step = 0
  for step in range(max_steps):
    obs = assemble_obs(model, data, index, params.home_joint, goal_quat, last, prev)
    action = policy.act(obs)
    prev, last = last, action.astype(np.float32)

    if step < params.warmup_steps:
      data.ctrl[index.ctrl_ids] = params.home_joint
    else:
      data.ctrl[index.ctrl_ids] = (
        data.qpos[index.joint_qadr] + last * params.action_scale
      )

    m = compute_metrics(model, data, index, goal_quat, last, prev)
    agg["min_cube_z"] = min(agg["min_cube_z"], m["cube_z"])
    agg["max_torque_frac"] = max(agg["max_torque_frac"], m["torque_frac_max"])
    agg["max_self_force"] = max(agg["max_self_force"], m["self_force_max"])
    agg["max_grasp_force"] = max(agg["max_grasp_force"], m["grasp_force"])
    agg["max_penetration"] = max(agg["max_penetration"], m["max_penetration"])
    agg["max_action_rate"] = max(agg["max_action_rate"], m["action_rate"])
    agg["max_pos_violation"] = max(agg["max_pos_violation"], m["pos_limit_violation"])
    agg["max_joint_speed"] = max(agg["max_joint_speed"], m["joint_speed_max"])

    if m["goal_error"] < params.success_threshold:
      hold += 1
      if hold >= params.success_hold_steps and steps_to_success < 0:
        steps_to_success = step
        break  # reached and held -> success
    else:
      hold = 0

    if step >= params.drop_grace_steps and m["cube_z"] < params.drop_height:
      dropped = True
      break

    for _ in range(params.decimation):
      mujoco.mj_step(model, data)

  return TrialResult(
    goal_index=goal_index,
    success=steps_to_success >= 0,
    steps_to_success=steps_to_success,
    dropped=dropped,
    n_steps=step + 1,
    final_error=m["goal_error"],
    min_cube_z=agg["min_cube_z"],
    max_torque_frac=agg["max_torque_frac"],
    max_self_force=agg["max_self_force"],
    max_grasp_force=agg["max_grasp_force"],
    max_penetration=agg["max_penetration"],
    max_action_rate=agg["max_action_rate"],
    max_pos_violation=agg["max_pos_violation"],
    max_joint_speed=agg["max_joint_speed"],
  )
