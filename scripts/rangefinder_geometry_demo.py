"""Dense local-geometry sensing with MuJoCo's camera rangefinder (raycast array).

A sibling to ``touch_grid_drop_demo.py``, but using a fundamentally different sensor.
Where ``touch_grid`` bins the few rigid *contact points* (so a fingertip only ever
lights 1-2 taxels), an orthographic camera + ``mjSENS_RANGEFINDER`` casts a dense grid
of rays at the object and reports, per ray, the distance and the surface normal. That
gives a real depth-plus-normal image, so a corner shows up as a depth peak with three
distinct face normals and a flat face shows up as constant depth with one normal -- the
dense geometry signal the contact-based sensor cannot produce. No SDF required; this
samples geometry with rays.

  uv run python scripts/rangefinder_geometry_demo.py          # figure -> <repo>/rangefinder_geometry.png
  uv run python scripts/rangefinder_geometry_demo.py --check  # verify depth/normal vs analytic ground truth

This is a geometry probe, not a force sensor: it reports what the surface *is* (shape
and orientation), not how hard it is pressed.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

CUBE_HALF = 0.0225  # qwerty cube half-extent (45 mm)
GRID = 28  # rays per side
EXTENT = 0.06  # orthographic view size (m), ~ a fingertip-scale patch
CAM_Z = 0.12  # camera height above the cube center

_RAY = mujoco.mjtRayDataField
DATASPEC = (1 << int(_RAY.mjRAYDATA_DIST)) | (1 << int(_RAY.mjRAYDATA_NORMAL))


def quat_align(a, b) -> list[float]:
  """Shortest-arc quaternion (wxyz) rotating unit vector a onto unit vector b."""
  a = np.array(a, float) / np.linalg.norm(a)
  b = np.array(b, float) / np.linalg.norm(b)
  q = np.array([1 + np.dot(a, b), *np.cross(a, b)])
  return (q / np.linalg.norm(q)).tolist()


POSES = {
  "face": [1.0, 0.0, 0.0, 0.0],
  "edge": [math.cos(math.pi / 8), math.sin(math.pi / 8), 0.0, 0.0],
  "corner": quat_align([1, 1, 1], [0, 0, 1]),
}


def rangefinder(
  add_target: Callable[[mujoco.MjSpec], None],
  grid: int = GRID,
  extent: float = EXTENT,
  cam_z: float = CAM_Z,
) -> tuple[np.ndarray, np.ndarray]:
  """Cast a downward orthographic ray grid at a target; return (depth, normal).

  ``add_target`` adds the geometry to look at (in its own body, since the rangefinder
  excludes the camera's body). Returns depth ``(grid, grid)`` with -1 where a ray
  misses, and the per-ray surface normal ``(grid, grid, 3)``.
  """
  spec = mujoco.MjSpec()
  add_target(spec)
  cam = spec.worldbody.add_camera(name="rf", pos=[0, 0, cam_z])
  cam.proj = mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
  cam.resolution = np.array([grid, grid], dtype=np.int32)
  cam.fovy = extent  # orthographic: fovy is the view extent in length units
  sensor = spec.add_sensor(
    type=mujoco.mjtSensor.mjSENS_RANGEFINDER,
    objtype=mujoco.mjtObj.mjOBJ_CAMERA,
    objname="rf",
  )
  sensor.intprm[0] = DATASPEC  # per ray: [dist, normal_x, normal_y, normal_z]

  m = spec.compile()
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  adr, dim = m.sensor_adr[0], m.sensor_dim[0]
  out = d.sensordata[adr : adr + dim].reshape(grid, grid, 4)
  return out[..., 0], out[..., 1:4]


def _cube(
  quat: list[float], half: float = CUBE_HALF
) -> Callable[[mujoco.MjSpec], None]:
  def add(spec: mujoco.MjSpec) -> None:
    body = spec.worldbody.add_body(pos=[0, 0, 0], quat=quat)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, half])

  return add


def sense(cube_quat: list[float]) -> tuple[np.ndarray, np.ndarray]:
  return rangefinder(_cube(cube_quat))


# --------------------------------------------------------------------------------------
# Mode: verify against analytic ground truth.
# --------------------------------------------------------------------------------------
def run_check() -> None:
  ok = True

  def check(name: str, cond: bool) -> None:
    nonlocal ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL':4}] {name}")

  def flat(top_z: float) -> Callable[[mujoco.MjSpec], None]:
    def add(spec: mujoco.MjSpec) -> None:
      body = spec.worldbody.add_body(pos=[0, 0, top_z - 0.05])
      body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, 0.5, 0.05])

    return add

  # 1. Exact distance + normal on a known flat surface (cam at z=0.2, surface at z=0).
  dep, nrm = rangefinder(flat(0.0), cam_z=0.2)
  hit = dep > 0
  print("Test 1: flat surface, top at z=0, orthographic cam at z=0.2")
  check(f"all rays hit ({hit.sum()}/{hit.size})", bool(hit.all()))
  check(
    f"depth == 0.2 to machine precision (got {dep[hit].mean():.6f}, std {dep[hit].std():.0e})",
    abs(dep[hit].mean() - 0.2) < 1e-5 and dep[hit].std() < 1e-6,
  )
  check(
    f"normal == (0,0,1) (got {nrm[hit].mean(0).round(5)})",
    bool(np.allclose(nrm[hit].mean(0), [0, 0, 1], atol=1e-5)),
  )

  # 2. Raising the surface by 0.05 drops the depth by exactly 0.05.
  dep2, _ = rangefinder(flat(0.05), cam_z=0.2)
  print("Test 2: surface raised by 0.05")
  check(
    f"depth == 0.15 (got {dep2[dep2 > 0].mean():.6f})",
    abs(dep2[dep2 > 0].mean() - 0.15) < 1e-5,
  )

  # 3. A plane tilted by a known angle reports that exact surface normal.
  th = math.radians(25)
  dep3, nrm3 = rangefinder(_cube([math.cos(th / 2), math.sin(th / 2), 0, 0], half=0.06))
  normals = nrm3[dep3 > 0]
  dominant = np.array(
    Counter(tuple(np.round(v, 2)) for v in normals).most_common(1)[0][0]
  )
  dominant = dominant / np.linalg.norm(dominant)
  expected = np.array([0, -math.sin(th), math.cos(th)])
  print(f"Test 3: surface tilted {math.degrees(th):.0f} deg about x")
  check(
    f"dominant normal == {expected.round(3)} (got {dominant.round(3)})",
    bool(np.allclose(dominant, expected, atol=2e-2)),
  )

  # 4. A corner-up cube exposes its three rotated faces, peaking at the tip.
  qc = quat_align([1, 1, 1], [0, 0, 1])
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, np.array(qc))
  rot = rot.reshape(3, 3)
  faces = [rot @ np.array(v, float) for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]
  dep4, nrm4 = rangefinder(_cube(qc, half=0.03))
  hit4 = dep4 > 0
  measured = nrm4[hit4]
  matched = sum(1 for f in faces if (measured @ f).max() > 0.98)
  print("Test 4: corner-up cube exposes its three faces")
  check(f"all 3 analytic face normals recovered (matched {matched}/3)", matched == 3)
  row, col = np.unravel_index(int(np.argmin(np.where(hit4, dep4, np.inf))), dep4.shape)
  n = dep4.shape[0]
  check(
    f"depth peak (corner tip) near center (row {row}, col {col} of {n})",
    abs(row - n // 2) < n // 4 and abs(col - n // 2) < n // 4,
  )

  print("\nAll ground-truth checks passed." if ok else "\n[!] a check FAILED.")
  if not ok:
    raise SystemExit(1)


# --------------------------------------------------------------------------------------
# Mode: figure of face / edge / corner depth + normal maps.
# --------------------------------------------------------------------------------------
def normal_rgb(normal: np.ndarray, depth: np.ndarray) -> np.ndarray:
  """Map surface normals to color; misses (no hit) are black."""
  rgb = (np.clip(normal, -1, 1) + 1) * 0.5
  rgb[depth <= 0] = 0.0
  return rgb


def run_figure(out: Path) -> None:
  import matplotlib.pyplot as plt

  frames = []
  for name in ("face", "edge", "corner"):
    depth, normal = sense(POSES[name])
    hit = depth > 0
    height = np.where(hit, depth.max() - depth, np.nan)  # height above lowest hit
    n_faces = len({tuple(np.round(v, 1)) for v in normal[hit]}) if hit.any() else 0
    frames.append((name, height, normal_rgb(normal, depth), int(hit.sum()), n_faces))
    print(
      f"  {name:<7} rays hit={int(hit.sum()):3d}/{GRID * GRID}  distinct normals~{n_faces}"
    )

  fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
  for col, (name, height, nrgb, nhit, nfaces) in enumerate(frames):
    axes[0, col].imshow(height, cmap="viridis", interpolation="nearest")
    axes[0, col].set_title(f"{name}\ndepth (height), {nhit} rays hit", fontsize=10)
    axes[1, col].imshow(nrgb, interpolation="nearest")
    axes[1, col].set_title(
      f"surface normal (RGB)\n~{nfaces} distinct faces", fontsize=10
    )
    for ax in (axes[0, col], axes[1, col]):
      ax.set_xticks([])
      ax.set_yticks([])
  axes[0, 0].set_ylabel("depth", fontsize=11)
  axes[1, 0].set_ylabel("normal", fontsize=11)
  fig.suptitle(
    f"camera rangefinder (raycast array): dense local geometry of a 45mm cube "
    f"({GRID}x{GRID} rays, orthographic {EXTENT * 1000:.0f}mm)",
    fontsize=12,
  )
  fig.tight_layout()
  fig.savefig(out, dpi=110, bbox_inches="tight")
  print(f"\nwrote {out}")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
    "--check", action="store_true", help="verify against analytic ground truth"
  )
  ap.add_argument(
    "--out",
    type=Path,
    default=Path(__file__).resolve().parent.parent / "rangefinder_geometry.png",
    help="figure output path",
  )
  args = ap.parse_args()
  if args.check:
    run_check()
  else:
    run_figure(args.out)


if __name__ == "__main__":
  main()
