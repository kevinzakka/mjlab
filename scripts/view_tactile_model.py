"""Open the tactile sim2sim model in the MuJoCo viewer to eyeball the sensors.

Builds the (non-inverted) reorient model with the five fingertip touch_grid sensors
attached and launches the native viewer holding the home grasp. No policy -- this is
just to inspect the sensor sites and their aim.

  # macOS needs mjpython for the managed viewer:
  uv run mjpython scripts/view_tactile_model.py

What you see:
  * Five tactile sites (one per fingertip distal phalanx). Turn them on/orient-check
    via the viewer's Rendering panel: enable "Site" under the Model elements, and set
    "Frame" -> "Site" to draw each site's xyz axes -- the sensor looks down its -z
    (blue axis points away from the grasp; -blue into it).
  * The touch_grid taxels render as a colored grid ON each pad, but ONLY when that
    pad carries contact force. At the home grasp the cube is cradled and may not touch
    the fingertips, so press/curl the cube into the fingers (drag it with double-click +
    Ctrl-right-drag) to watch a pad light up. A real policy rollout (sim2sim_play
    --tactile) lights them up as it grips.
"""

from __future__ import annotations

import mujoco
import mujoco.viewer

import mjlab.tasks.reorient.scripts.sim2sim_core as core


def main() -> None:
  model, _ = core.build_model(inverted=False, tactile=True)
  data = mujoco.MjData(model)
  key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  data.qpos[:] = model.key_qpos[key]
  # Hold the home posture so the hand does not go limp while you look around.
  ctrl_adr = model.jnt_qposadr[: model.nu]
  data.ctrl[:] = data.qpos[ctrl_adr]
  mujoco.mj_forward(model, data)

  print(
    f"Tactile model: {model.nsensor} sensors "
    f"({sum(bool((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) or '').startswith('tactile_')) for i in range(model.nsensor))} touch_grid). "
    "Rendering panel: enable Site + Frame->Site to see each sensor's -z aim. "
    "Taxels render on a pad only under contact force."
  )

  def hold(m, d) -> None:
    del m
    d.ctrl[:] = d.qpos[ctrl_adr]

  mujoco.set_mjcb_control(hold)
  try:
    mujoco.viewer.launch(model, data)
  finally:
    mujoco.set_mjcb_control(None)


if __name__ == "__main__":
  main()
