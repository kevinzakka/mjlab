"""Interactive superquadric (supersphere) explorer in viser.

Sweep the MuJoCo builtin ``supersphere`` parameters and watch the shape, its convex
collision hull, and the resulting mass/inertia update live. Pure intuition tool for the
"reorient anything" exploration -- no task, no training.

    uv run python scripts/superquadric_explorer.py

then open the printed URL. Knobs: exponents e (east-west) and n (north-south), per-axis
scale, resolution (surface face count), and maxhullvert (the cap on the *collision* hull
that GJK uses -- the perf-relevant number for training).

Notes
-----
* ``e = n = 1`` is a sphere. Small exponents (->0) square it off toward a box/octahedron;
  large exponents (->inf) pinch it toward a star/pillow. Scale stretches the semi-axes.
* The surface mesh is what you see; the convex hull (toggle on) is what actually collides.
  ``maxhullvert`` shrinks the hull, not the surface -- watch the "hull verts" stat drop.
"""

from __future__ import annotations

import time

import mujoco
import numpy as np
import viser


def build(
  resolution: int,
  e: float,
  n: float,
  scale: tuple[float, float, float],
  maxhullvert: int,
) -> dict:
  """Compile a single supersphere geom and pull out its surface + hull + inertia."""
  spec = mujoco.MjSpec()
  mesh = spec.add_mesh(name="obj")
  mesh.make_supersphere(resolution=int(resolution), e=float(e), n=float(n))
  mesh.scale = list(scale)
  # maxhullvert <= 3 means "unlimited" in MuJoCo (-1); the UI uses 0 for that.
  mesh.maxhullvert = int(maxhullvert) if maxhullvert > 3 else -1
  body = spec.worldbody.add_body(name="obj_body")
  geom = body.add_geom()
  geom.type = mujoco.mjtGeom.mjGEOM_MESH
  geom.meshname = "obj"

  model = spec.compile()

  va, vn = int(model.mesh_vertadr[0]), int(model.mesh_vertnum[0])
  fa, fn = int(model.mesh_faceadr[0]), int(model.mesh_facenum[0])
  verts = np.array(model.mesh_vert[va : va + vn], dtype=np.float32)
  faces = np.array(model.mesh_face[fa : fa + fn], dtype=np.int32)

  hull_verts, hull_faces = _extract_hull(model, verts, faces)

  # Mass + principal inertia of the single body (geom uses the default density).
  mass = float(model.body_mass[1])
  inertia = np.array(model.body_inertia[1], dtype=np.float64)

  return {
    "verts": verts,
    "faces": faces,
    "hull_verts": hull_verts,
    "hull_faces": hull_faces,
    "mass": mass,
    "inertia": inertia,
  }


def _extract_hull(
  model: mujoco.MjModel, verts: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Return (vertices, triangles) of the collision convex hull.

  MuJoCo stores the hull in ``mesh_graph`` with layout (see user_mesh.cc):
  ``numvert, numface, vert_edgeadr[numvert], vert_globalid[numvert],
  edge_localid[numvert + 3*numface], face_globalid[3*numface]`` -- face indices are
  global (into this mesh's vertex block). Small meshes get no graph (graphadr < 0); then
  the surface already *is* the hull.
  """
  adr = int(model.mesh_graphadr[0])
  if adr < 0:
    return verts, faces
  g = model.mesh_graph
  numvert = int(g[adr])
  numface = int(g[adr + 1])
  face_off = adr + 2 + 3 * numvert + 3 * numface
  face_globalid = np.array(g[face_off : face_off + 3 * numface], dtype=np.int32)
  return verts, face_globalid.reshape(numface, 3)


def main() -> None:
  server = viser.ViserServer()
  server.scene.set_up_direction("+z")

  g = server.gui
  s_res = g.add_slider("resolution (faces)", min=3, max=40, step=1, initial_value=20)
  s_e = g.add_slider(
    "e  (east-west exp)", min=0.1, max=4.0, step=0.05, initial_value=1.0
  )
  s_n = g.add_slider(
    "n  (north-south exp)", min=0.1, max=4.0, step=0.05, initial_value=1.0
  )
  s_size = g.add_slider(
    "base size (m, radius)", min=0.015, max=0.06, step=0.001, initial_value=0.03
  )
  s_sx = g.add_slider("aspect x", min=0.2, max=1.5, step=0.01, initial_value=1.0)
  s_sy = g.add_slider("aspect y", min=0.2, max=1.5, step=0.01, initial_value=1.0)
  s_sz = g.add_slider("aspect z", min=0.2, max=1.5, step=0.01, initial_value=1.0)
  s_hull = g.add_slider(
    "maxhullvert (0 = uncapped)", min=0, max=200, step=1, initial_value=0
  )
  show_surface = g.add_checkbox("show surface", initial_value=True)
  show_hull = g.add_checkbox("show collision hull", initial_value=False)
  flat_shade = g.add_checkbox("flat shading (reads form better)", initial_value=True)
  s_light = g.add_slider(
    "light intensity", min=0.1, max=2.5, step=0.1, initial_value=0.3
  )
  stats = g.add_markdown("")

  def relight(_=None) -> None:
    # Studio HDRI gives soft directional gradients that make the surface readable,
    # unlike the flat default ambient. Re-applied when the intensity slider changes.
    server.scene.configure_environment_map(
      "studio", environment_intensity=s_light.value
    )

  s_light.on_update(relight)
  relight()

  controls = [
    s_res,
    s_e,
    s_n,
    s_size,
    s_sx,
    s_sy,
    s_sz,
    s_hull,
    show_surface,
    show_hull,
    flat_shade,
  ]

  state = {"busy": False}

  def rebuild(_=None) -> None:
    if state["busy"]:  # suppressed while a preset sets many sliders at once
      return
    base = s_size.value
    d = build(
      s_res.value,
      s_e.value,
      s_n.value,
      (base * s_sx.value, base * s_sy.value, base * s_sz.value),
      s_hull.value,
    )
    if show_surface.value:
      server.scene.add_mesh_simple(
        "/surface",
        d["verts"],
        d["faces"],
        color=(90, 200, 255),
        opacity=0.45 if show_hull.value else 1.0,
        flat_shading=flat_shade.value,
      )
    else:
      server.scene.add_mesh_simple("/surface", np.zeros((3, 3)), np.array([[0, 1, 2]]))
    if show_hull.value:
      server.scene.add_mesh_simple(
        "/hull",
        d["hull_verts"],
        d["hull_faces"],
        color=(255, 140, 70),
        wireframe=True,
      )
    else:
      server.scene.add_mesh_simple("/hull", np.zeros((3, 3)), np.array([[0, 1, 2]]))

    ix, iy, iz = d["inertia"]
    stats.content = (
      f"**surface**: {len(d['verts'])} verts / {len(d['faces'])} faces  \n"
      f"**collision hull**: {len(np.unique(d['hull_faces']))} verts / "
      f"{len(d['hull_faces'])} faces  \n"
      f"**mass**: {d['mass'] * 1e3:.1f} g  \n"
      f"**inertia** (diag): {ix:.2e}, {iy:.2e}, {iz:.2e}"
    )

  for c in controls:
    c.on_update(rebuild)

  # Presets: each sets every slider then rebuilds once. (res, e, n, size, sx, sy, sz, hull)
  presets = {
    "Sphere": (20, 1.0, 1.0, 0.03, 1.0, 1.0, 1.0, 0),
    "Cube": (20, 0.2, 0.2, 0.03, 1.0, 1.0, 1.0, 0),
    "Brick": (20, 0.25, 0.25, 0.03, 1.4, 0.7, 0.9, 0),
    "Cylinder": (24, 1.0, 0.25, 0.03, 1.0, 1.0, 1.0, 0),
    "Coin / disc": (24, 1.0, 0.3, 0.03, 1.2, 1.2, 0.4, 0),
    "Egg / capsule": (24, 1.0, 1.2, 0.03, 0.8, 0.8, 1.3, 0),
    "Lemon / spindle": (24, 1.0, 2.2, 0.03, 1.0, 1.0, 1.0, 0),
    "Octahedron": (20, 2.4, 2.4, 0.03, 1.0, 1.0, 1.0, 0),
    "Pillow": (24, 0.6, 1.3, 0.03, 1.0, 1.0, 1.0, 0),
    "Star (non-convex!)": (28, 3.5, 3.5, 0.03, 1.0, 1.0, 1.0, 0),
  }

  def apply_preset(vals: tuple) -> None:
    state["busy"] = True
    (
      s_res.value,
      s_e.value,
      s_n.value,
      s_size.value,
      s_sx.value,
      s_sy.value,
      s_sz.value,
      s_hull.value,
    ) = vals
    state["busy"] = False
    rebuild()

  with g.add_folder("Presets"):
    for name, vals in presets.items():
      g.add_button(name).on_click(lambda _, v=vals: apply_preset(v))

  with g.add_folder("Knob guide", expand_by_default=False):
    g.add_markdown(
      "**e** — footprint (top view): `1` round · `<1` square · `>1` diamond/star  \n"
      "**n** — profile (side view): `1` round · `<1` flat (cylinder) · `>1` pointed  \n"
      "**aspect x/y/z** — stretch each axis (sphere→ellipsoid, cube→brick)  \n"
      "**base size** — radius (m); mass ~ size^3  \n"
      "**resolution** — surface smoothness / face count  \n"
      "**maxhullvert** — caps the *collision* hull (GJK perf); surface stays smooth. "
      "On the Star, watch the hull fill in the concavities (collision is convex)."
    )

  rebuild()

  print("Superquadric explorer running. Open the URL above. Ctrl-C to quit.")
  while True:
    time.sleep(1.0)


if __name__ == "__main__":
  main()
