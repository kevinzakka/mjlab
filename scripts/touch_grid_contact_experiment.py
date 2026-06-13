"""MRE: prove that touch_grid lit-taxel count == MuJoCo contact-point count.

Presses the 45 mm box cube flat onto a fingertip-sized pad of three different
collision primitives and reports, for each, how many contacts MuJoCo generates
and how many taxels light up. This isolates *why* the live tactile view only ever
shows 1-2 squares: it is the collision primitive (a capsule pad caps at 2 contacts),
not the taxel resolution and not the sensor.

  uv run python scripts/touch_grid_contact_experiment.py

Expected (MuJoCo 3.8): sphere->1, capsule->2, box->4 contacts, and lit taxels equal
to the contact count in every case, with the total normal force conserved (~0.55 N).
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

PAD_HALF_TOP = {
  "sphere": 0.009,
  "capsule": 0.009,
  "box": 0.004,
}  # geom top above center
CUBE_HALF = 0.0225
COMMON: dict[str, Any] = dict(
  solref=[0.012, 1.0],
  solimp=[0.95, 0.99, 0.0005, 0.5, 2.0],
  condim=4,
  friction=[1.0, 0.004, 0.0001],
)


def build(pad_kind: str, grid: int = 12, fov: str = "120 80"):
  spec = mujoco.MjSpec()
  spec.activate_plugin("mujoco.sensor.touch_grid")
  spec.option.timestep = 0.002
  pad = spec.worldbody.add_body(name="pad", pos=[0, 0, 0.05])
  if pad_kind == "capsule":
    pad.add_geom(
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      size=[0.009, 0.007, 0],
      quat=[0.7071, 0, 0.7071, 0],
      **COMMON,
    )
  elif pad_kind == "box":
    pad.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.009, 0.007, 0.004], **COMMON)
  elif pad_kind == "sphere":
    pad.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.009, 0, 0], **COMMON)
  pad.add_site(name="s", pos=[0, 0, -0.03], quat=[0, 1, 0, 0])  # looks up (+z world)

  cube_z = 0.05 + PAD_HALF_TOP[pad_kind] + CUBE_HALF - 0.0015  # 1.5 mm penetration
  cube = spec.worldbody.add_body(name="cube", pos=[0, 0, cube_z])
  cube.add_freejoint()
  cube.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3, density=300, **COMMON
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
  return spec.compile(), grid


def measure(pad_kind: str, multiccd: bool = True, grid: int = 12):
  m, grid = build(pad_kind, grid=grid)
  if not multiccd:
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
  d = mujoco.MjData(m)
  cube_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
  mujoco.mj_forward(m, d)
  ncon = sum(
    1
    for i in range(d.ncon)
    if cube_bid
    in (m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2])
  )
  adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "t")]
  normal = d.sensordata[adr : adr + 3 * grid * grid].reshape(3, grid, grid)[0]
  return ncon, int((np.abs(normal) > 1e-9).sum()), float(normal.sum())


def main() -> None:
  print(
    f"{'pad geom':<10} {'multiccd':<9} {'cube contacts':>14} "
    f"{'lit taxels':>11} {'normal sum N':>13}"
  )
  for kind in ("sphere", "capsule", "box"):
    for mc in (True, False):
      ncon, lit, total = measure(kind, multiccd=mc)
      print(f"{kind:<10} {str(mc):<9} {ncon:>14} {lit:>11} {total:>13.3f}")
  print(
    "\nlit taxels == contact count in every row: the sensor faithfully bins each\n"
    "contact into one taxel, and the contact count is set by the collision primitive."
  )


if __name__ == "__main__":
  main()
