"""Standalone intuition + sanity demo for the MuJoCo ``touch_grid`` sensor.

A throwaway exploration (not part of the harness) to build intuition for what a
fingertip ``touch_grid`` tactile sensor resolves, using a fixed capsule pad sized
like a Sharpa fingertip and the 45 mm qwerty cube. Four modes:

  # Drop the cube face/edge/corner-down, save a taxel figure to <repo>/touch_grid_drop.png
  uv run python scripts/touch_grid_drop_demo.py

  # Physics check: taxel normal-force sum == solver contact force == cube weight (m*g)
  uv run python scripts/touch_grid_drop_demo.py --check

  # Stream a slide/roll/press sweep to rerun: watch the hotspot migrate, sharpen on
  # an edge, and brighten under load (per-finger panels + 3D cube/pad + force scalars)
  uv run python scripts/touch_grid_drop_demo.py --rerun

  # Watch it live in the MuJoCo viewer: cube tumbles onto the pad, taxels light up
  #   (macOS: run via `uv run mjpython scripts/touch_grid_drop_demo.py --viz`)
  uv run python scripts/touch_grid_drop_demo.py --viz

The headline: a corner concentrates contact into one blazing taxel, an edge into
a short streak, a flat face into a low broad patch. MuJoCo emits only a handful
of contact points per geom pair, so this is intensity/concentration coding, not a
dense GelSight-style imprint -- but face/edge/corner are clearly distinguishable,
and the integrated normal force is physically correct (that is what --check proves).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np

# Geometry / contact params lifted from the real assets.
CUBE_HALF = 0.0225  # qwerty cube half-extent (45 mm cube)
PAD_R, PAD_HL = 0.009, 0.007  # fingertip pad capsule (radius, half-length)
PAD_Z = 0.05  # pad height above ground
STANDOFF = 0.014  # site pulled back below the pad so the pad subtends a clean cone

SOLREF = [0.012, 1.0]
SOLIMP = [0.95, 0.99, 0.0005, 0.5, 2.0]

FACE = [1.0, 0.0, 0.0, 0.0]
EDGE = [math.cos(math.pi / 8), math.sin(math.pi / 8), 0.0, 0.0]


def quat_align(a, b) -> list[float]:
  """Shortest-arc quaternion (wxyz) rotating unit vector a onto unit vector b."""
  a = np.array(a, float) / np.linalg.norm(a)
  b = np.array(b, float) / np.linalg.norm(b)
  q = np.array([1 + np.dot(a, b), *np.cross(a, b)])
  return (q / np.linalg.norm(q)).tolist()


CORNER = quat_align([1, 1, 1], [0, 0, -1])  # a vertex pointing straight down
POSES = {"face": FACE, "edge": EDGE, "corner": CORNER}


def build(
  cube_quat: list[float], drop_h: float, grid: int, fov: str, flat: bool = False
) -> tuple[mujoco.MjModel, mujoco.MjData]:
  spec = mujoco.MjSpec()
  spec.activate_plugin("mujoco.sensor.touch_grid")
  spec.option.timestep = 0.002  # gravity defaults to (0, 0, -9.81)

  # Fixed pad with an upward-looking touch_grid site (-z_site -> +z_world via a
  # 180 deg flip about x). Site sits below the pad so the contact patch subtends a
  # modest cone instead of blowing up in angle at zero range. ``flat`` swaps the
  # tiny fingertip capsule for a wide plate the cube can rest on -- used by --check,
  # where a stable resting contact is needed to compare against the cube's weight.
  pad = spec.worldbody.add_body(name="pad", pos=[0, 0, PAD_Z])
  if flat:
    pad.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=[0.04, 0.04, 0.005],
      solref=SOLREF,
      solimp=SOLIMP,
      condim=4,
      friction=[1.0, 0.004, 0.0001],
      rgba=[0.3, 0.6, 0.3, 1.0],
    )
    pad.add_site(name="s", pos=[0, 0, -0.03], quat=[0, 1, 0, 0])
  else:
    pad.add_geom(
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      size=[PAD_R, PAD_HL, 0.0],
      quat=[0.707107, 0, 0.707107, 0],  # capsule long axis along world x
      solref=SOLREF,
      solimp=SOLIMP,
      condim=4,
      friction=[1.0, 0.004, 0.0001],
      rgba=[0.3, 0.6, 0.3, 1.0],
    )
    pad.add_site(name="s", pos=[0, 0, -STANDOFF], quat=[0, 1, 0, 0])

  cube = spec.worldbody.add_body(
    name="cube", pos=[0, 0, PAD_Z + drop_h], quat=cube_quat
  )
  cube.add_freejoint()
  cube.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[CUBE_HALF, CUBE_HALF, CUBE_HALF],
    solref=SOLREF,
    solimp=SOLIMP,
    condim=4,
    friction=[1.0, 0.004, 0.0001],
    density=300.0,
    rgba=[0.7, 0.4, 0.2, 1.0],
  )

  pl = spec.add_plugin(name="t", plugin_name="mujoco.sensor.touch_grid", active=True)
  pl.config = {"nchannel": "3", "size": f"{grid} {grid}", "fov": fov, "gamma": "0"}
  sen = spec.add_sensor(
    type=mujoco.mjtSensor.mjSENS_PLUGIN,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="s",
    name="t",
  )
  sen.plugin = pl
  m = spec.compile()
  return m, mujoco.MjData(m)


def read_taxels(m, d, grid: int) -> np.ndarray:
  sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "t")
  a, n = m.sensor_adr[sid], m.sensor_dim[sid]
  return d.sensordata[a : a + n].reshape(3, grid, grid)


def contact_normal_sum(m, d) -> float:
  """Sum of the solver's per-contact normal forces (the physical ground truth)."""
  total = 0.0
  buf = np.zeros(6)
  for i in range(d.ncon):
    mujoco.mj_contactForce(m, d, i, buf)
    total += abs(float(buf[0]))  # buf[0] = normal force in the contact frame
  return total


def settle(m, d, steps: int = 4000, vel_tol: float = 1e-4, hold: int = 200) -> int:
  """Step until the cube is *sustainably* at rest; return the settling step count.

  Requires ``hold`` consecutive low-velocity steps so we don't mistake the
  velocity zero-crossing at peak bounce-compression (where the contact force spikes
  well above m*g) for true rest.
  """
  rest = 0
  for k in range(steps):
    mujoco.mj_step(m, d)
    rest = rest + 1 if (float(np.linalg.norm(d.qvel)) < vel_tol and d.ncon > 0) else 0
    if rest >= hold:
      return k
  return steps


def static_press(name: str, grid: int, fov: str, depth: float = 0.0015):
  """Cube pressed onto the capsule pad at a fixed penetration, evaluated statically.

  Positions the cube so its lowest feature (face/edge/corner) overlaps the pad top
  by ``depth``, then a single ``mj_forward`` solves the instantaneous contact -- no
  bounce, no transient -- so the taxel sum and the solver contact force are read at
  the exact same, settled instant.
  """
  m, d = build(POSES[name], drop_h=0.0, grid=grid, fov=fov)
  pad_top = PAD_Z + PAD_R
  reach = {"face": 1.0, "edge": math.sqrt(2), "corner": math.sqrt(3)}[name]
  d.qpos[0:3] = [0.0, 0.0, pad_top + CUBE_HALF * reach - depth]
  d.qpos[3:7] = POSES[name]
  mujoco.mj_forward(m, d)
  normal = read_taxels(m, d, grid)[0]
  return contact_normal_sum(m, d), float(normal.sum()), int(d.ncon), normal


# --------------------------------------------------------------------------------------
# Mode: physics check.
# --------------------------------------------------------------------------------------
def run_check(grid: int, fov: str) -> None:
  ok = True

  # Test 1: weight balance. Rest the cube flat on a wide plate; at static
  # equilibrium the plate must carry the whole cube, so the integrated taxel normal
  # force == the solver's contact normal force == m*g. (A 45 mm cube won't balance
  # on a 9 mm capsule, hence a plate here -- the identity is geometry-independent.)
  print("Test 1 -- weight balance (cube at rest on a flat plate):")
  print("  expect  sum(taxel normal)  ==  sum(solver contact normal)  ==  m*g\n")
  m, d = build(FACE, drop_h=0.01, grid=grid, fov="80 80", flat=True)
  mg = 9.81 * float(m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")])
  settled = settle(m, d)
  contact = contact_normal_sum(m, d)
  taxels = float(read_taxels(m, d, grid)[0].sum())
  e_sensor = abs(taxels - contact) / (contact + 1e-9)  # sensor vs solver
  e_weight = abs(contact - mg) / mg  # solver vs gravity
  print(f"  m*g                 = {mg:7.4f} N   (settled after {settled} steps)")
  print(f"  solver contact sum  = {contact:7.4f} N   ({e_weight:.1%} vs m*g)")
  print(f"  taxel normal sum    = {taxels:7.4f} N   ({e_sensor:.1%} vs solver)")
  t1 = e_sensor < 0.05 and e_weight < 0.05
  ok = ok and t1
  print(f"  -> {'PASS' if t1 else 'FAIL'}\n")

  # Test 2: the sensor reproduces the solver's contact force for face/edge/corner
  # contacts on the fingertip capsule, evaluated under a *static* press (fixed
  # penetration, one mj_forward) so we compare both at the same settled instant --
  # no impact transient. Also reports the peak-taxel concentration, which is the
  # intuition payoff: a corner pours the same-order force through far fewer taxels.
  print("Test 2 -- sensor == solver under a static press (capsule pad):")
  print(
    f"  {'pose':<12} {'solver (N)':>11} {'taxels (N)':>11} {'err':>7} "
    f"{'peak/taxel':>11} {'lit taxels':>11}"
  )
  for name in ("face", "edge", "corner"):
    contact, taxels, ncon, normal = static_press(name, grid, fov)
    del ncon
    err = abs(taxels - contact) / (contact + 1e-9)
    peak = float(normal.max())
    lit = int((normal > 0.01 * peak).sum()) if peak > 0 else 0
    flag = err < 0.05 and contact > 0
    ok = ok and flag
    print(
      f"  {name + '-down':<12} {contact:>11.3f} {taxels:>11.3f} {err:>6.1%} "
      f"{peak:>11.3f} {lit:>11d}  {'PASS' if flag else 'FAIL'}"
    )

  print(
    "\nThe taxel sum reproduces the solver's contact force to <5% (binning/rounding), "
    "and\nat rest equals m*g: real, conserved contact force, not a cosmetic heatmap. "
    "Note how\nthe corner drives a high peak through few taxels while the face spreads "
    "a low one." + ("" if ok else "\n[!] a check FAILED -- inspect above.")
  )


# --------------------------------------------------------------------------------------
# Mode: static figure of the three poses.
# --------------------------------------------------------------------------------------
def run_figure(grid: int, fov: str, out: Path, per_panel: bool) -> None:
  import matplotlib.pyplot as plt

  frames = []
  for name in ("face", "edge", "corner"):
    m, d = build(POSES[name], drop_h=0.03, grid=grid, fov=fov)
    best, best_total, best_ncon = np.zeros((3, grid, grid)), -1.0, 0
    for _ in range(1200):  # capture the peak-force frame during impact
      mujoco.mj_step(m, d)
      img = read_taxels(m, d, grid)
      if img[0].sum() > best_total:
        best_total, best, best_ncon = float(img[0].sum()), img.copy(), d.ncon
    normal, shear = best[0], np.linalg.norm(best[1:], axis=0)
    frames.append((f"{name}-down", normal, shear, best_ncon))
    print(
      f"  {name + '-down':<12} peak_normal={normal.max():6.3f} N  contacts={best_ncon}"
    )

  nmax = None if per_panel else (max(f[1].max() for f in frames) or 1.0)
  smax = None if per_panel else (max(f[2].max() for f in frames) or 1.0)
  fig, axes = plt.subplots(2, 3, figsize=(11, 7))
  im0 = im1 = None
  for col, (name, normal, shear, ncon) in enumerate(frames):
    im0 = axes[0, col].imshow(normal, cmap="inferno", vmin=0, vmax=nmax)
    axes[0, col].set_title(f"{name}: {normal.max():.2f} N, {ncon} pts", fontsize=10)
    im1 = axes[1, col].imshow(shear, cmap="viridis", vmin=0, vmax=smax)
    axes[1, col].set_title(f"shear {shear.max():.2f} N", fontsize=10)
    for ax in (axes[0, col], axes[1, col]):
      ax.set_xticks([])
      ax.set_yticks([])
  axes[0, 0].set_ylabel("normal", fontsize=11)
  axes[1, 0].set_ylabel("shear-mag", fontsize=11)
  assert im0 is not None and im1 is not None
  if not per_panel:
    fig.colorbar(im0, ax=axes[0, :].tolist(), label="N / taxel")
    fig.colorbar(im1, ax=axes[1, :].tolist(), label="N / taxel")
  scale = "per-panel autoscale" if per_panel else "shared scale (true intensity)"
  fig.suptitle(
    f"touch_grid: 45mm cube on an {2 * PAD_R * 1000:.0f}mm pad "
    f"({grid}x{grid} taxels, fov {fov}) -- {scale}",
    fontsize=12,
  )
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out, dpi=110, bbox_inches="tight")
  print(f"\nwrote {out}")


# --------------------------------------------------------------------------------------
# Mode: interactive viewer (cube tumbles onto the pad, taxels render live).
# --------------------------------------------------------------------------------------
class _Dropper:
  """mjcb_control hook: re-drops the cube with a random tumble each time it settles."""

  def __init__(self, m, d, grid: int) -> None:
    self.m, self.d, self.grid = m, d, grid
    self.rng = np.random.default_rng(0)
    self.settled = 0
    self.cooldown = 0
    self._respawn()

  def _respawn(self) -> None:
    d, m = self.d, self.m
    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    d.qpos[0:3] = [0.0, 0.0, PAD_Z + 0.05]
    q = self.rng.standard_normal(4)
    d.qpos[3:7] = q / np.linalg.norm(q)
    d.qvel[3:6] = self.rng.standard_normal(3) * 4.0  # random tumble
    self.settled = 0
    self.cooldown = 60
    mujoco.mj_forward(m, d)

  def cb(self, m, d) -> None:
    if self.cooldown > 0:
      self.cooldown -= 1
      return
    # Re-drop once the cube has been at rest, or if it rolled off onto the floor.
    if float(np.linalg.norm(d.qvel)) < 5e-3:
      self.settled += 1
    else:
      self.settled = 0
    if self.settled > 120 or d.qpos[2] < PAD_Z - 0.03:
      self._respawn()


def run_viz(grid: int, fov: str) -> None:
  import mujoco.viewer

  m, d = build(CORNER, drop_h=0.05, grid=grid, fov=fov)
  dropper = _Dropper(m, d, grid)
  mujoco.set_mjcb_control(dropper.cb)
  print(
    "Viewer: the cube tumbles onto the pad and the touch_grid taxels render as a "
    "colored grid on the pad (bright = high normal force). Ctrl-C / close to quit.\n"
    "Tip: enable Contact Force in the viewer's Visualization panel to compare arrows."
  )
  try:
    mujoco.viewer.launch(m, d)
  finally:
    mujoco.set_mjcb_control(None)


# --------------------------------------------------------------------------------------
# Mode: rerun stream (kinematically sweep the cube; watch the taxels respond live).
# --------------------------------------------------------------------------------------
def _set_pose(m, d, pos, quat_wxyz) -> None:
  """Teleport the cube and recompute contacts (no integration -- a prescribed motion)."""
  d.qpos[0:3] = pos
  d.qpos[3:7] = quat_wxyz
  d.qvel[:] = 0.0
  mujoco.mj_forward(m, d)


def _sweep_poses(grid_top: float, depth: float):
  """Yield (phase, pos, quat_wxyz) frames for slide -> roll -> press."""
  # Slide: a flat face dragged across the pad -- pure hotspot migration.
  for x in np.linspace(-0.018, 0.018, 60):
    z = grid_top + CUBE_HALF - depth
    yield "1-slide", [float(x), 0.0, z], FACE
  # Roll: pivot about x through 90 deg, z tracking the support height so the
  # penetration stays fixed -- face (flat) -> edge (45 deg, a sharp line) -> face.
  for th in np.linspace(0.0, math.pi / 2, 70):
    support = CUBE_HALF * (abs(math.cos(th)) + abs(math.sin(th)))
    z = grid_top + support - depth
    yield "2-roll", [0.0, 0.0, z], [math.cos(th / 2), math.sin(th / 2), 0.0, 0.0]
  # Press: flat face, ramp the penetration -- brightness/force scales with push.
  for dz in np.linspace(0.0, 0.003, 40):
    z = grid_top + CUBE_HALF - dz
    yield "3-press", [0.0, 0.0, z], FACE


def run_rerun(grid: int, fov: str) -> None:
  try:
    import rerun as rr  # pyright: ignore[reportMissingImports]
  except ModuleNotFoundError as e:
    raise SystemExit("rerun is not installed: uv sync --group dev") from e
  from matplotlib import colormaps

  inferno, viridis = colormaps["inferno"], colormaps["viridis"]

  def colorize(img2d, vmax, cmap):
    x = np.clip(img2d / (vmax + 1e-9), 0.0, 1.0)
    rgb = (cmap(x)[..., :3] * 255).astype(np.uint8)
    return np.kron(rgb, np.ones((14, 14, 1), np.uint8))  # upscale for visibility

  # Flat plate so the prescribed slide/roll/press stay in stable contact.
  m, d = build(FACE, drop_h=0.0, grid=grid, fov=fov, flat=True)
  plate_top = PAD_Z + 0.005
  rr.init("touch_grid_sweep", spawn=True)
  # Static geometry: the green pad plate, logged once.
  rr.log(
    "scene/pad",
    rr.Boxes3D(centers=[[0, 0, PAD_Z]], half_sizes=[[0.04, 0.04, 0.005]]),
    static=True,
  )
  print("Streaming to rerun (spawns the viewer). Scrub the timeline to replay.")

  # Fixed display scales so brightness is comparable frame-to-frame (a press should
  # visibly brighten, not auto-normalize back to full white every frame).
  n_scale, s_scale = 0.6, 0.3
  for k, (phase, pos, quat) in enumerate(_sweep_poses(plate_top, depth=0.0015)):
    _set_pose(m, d, pos, quat)
    img = read_taxels(m, d, grid)
    normal, shear = img[0], np.linalg.norm(img[1:], axis=0)

    rr.set_time("frame", sequence=k)
    rr.log("tactile/normal", rr.Image(colorize(normal, n_scale, inferno)))
    rr.log("tactile/shear", rr.Image(colorize(shear, s_scale, viridis)))
    rr.log("force/total_normal", rr.Scalars(float(normal.sum())))
    rr.log("force/peak_taxel", rr.Scalars(float(normal.max())))
    rr.log(
      "force/lit_taxels", rr.Scalars(int((normal > 0.05 * (normal.max() or 1)).sum()))
    )
    rr.log("phase", rr.TextLog(phase))
    # 3D context: the cube (wxyz -> xyzw) and the live contact points.
    qx = [quat[1], quat[2], quat[3], quat[0]]
    rr.log(
      "scene/cube",
      rr.Boxes3D(
        centers=[pos],
        half_sizes=[[CUBE_HALF] * 3],
        quaternions=[rr.Quaternion(xyzw=qx)],
        colors=[[180, 100, 50]],
      ),
    )
    pts = [d.contact[i].pos.tolist() for i in range(d.ncon)]
    rr.log("scene/contacts", rr.Points3D(pts, radii=0.002, colors=[[255, 60, 60]]))

  print("done. The normal panel shows the hotspot slide across, sharpen on the edge")
  print("at 45 deg, then brighten as the press deepens. Scrub 'frame' to explore.")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--check", action="store_true", help="run the physics sanity check")
  ap.add_argument("--viz", action="store_true", help="interactive viewer (see it fall)")
  ap.add_argument(
    "--rerun", action="store_true", help="stream a slide/roll/press sweep to rerun"
  )
  ap.add_argument("--grid", type=int, default=12, help="taxels per side")
  ap.add_argument(
    "--fov", type=str, default="70 60", help="horizontal/vertical fov deg"
  )
  ap.add_argument(
    "--per-panel-scale",
    action="store_true",
    help="autoscale each figure panel (reveals faint face/edge detail) instead of a "
    "shared scale (which shows the true ~20x corner-vs-face intensity jump)",
  )
  ap.add_argument(
    "--out",
    type=Path,
    default=Path(__file__).resolve().parent.parent / "touch_grid_drop.png",
    help="figure output path (default: <repo>/touch_grid_drop.png)",
  )
  args = ap.parse_args()

  if args.viz:
    run_viz(args.grid, args.fov)
  elif args.rerun:
    run_rerun(args.grid, args.fov)
  elif args.check:
    run_check(args.grid, args.fov)
  else:
    run_figure(args.grid, args.fov, args.out, args.per_panel_scale)


if __name__ == "__main__":
  main()
