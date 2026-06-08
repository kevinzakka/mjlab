"""Qwerty cube prop: a per-face textured cube whose orientation is readable."""

from pathlib import Path

import mujoco

# Half edge length of the cube geom.
CUBE_HALF_EXTENT = 0.0225

_CUBE_TEXTURE_DIR = Path(__file__).parent / "textures"

# MjSpec cube-texture face order: right, left, up, down, front, back.
_CUBE_FACES = ("right", "left", "up", "down", "front", "back")


def _make_textured_cube_spec(
  name: str,
  size: float,
  rgba: tuple[float, float, float, float],
  *,
  freejoint: bool,
  collide: bool,
  mass: float | None = None,
) -> mujoco.MjSpec:
  """Build a textured-cube MjSpec used by both the physical cube and goal marker."""
  spec = mujoco.MjSpec()

  spec.add_texture(
    name=name,
    type=mujoco.mjtTexture.mjTEXTURE_CUBE,
    cubefiles=[str(_CUBE_TEXTURE_DIR / f"file{face}.png") for face in _CUBE_FACES],
  )
  mat = spec.add_material(name=name, rgba=rgba)
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = name

  body = spec.worldbody.add_body(name=name)
  if freejoint:
    body.add_freejoint(name=f"{name}_joint")
  geom_kwargs: dict = dict(
    name=f"{name}_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(size, size, size),
    material=name,
  )
  if collide:
    assert mass is not None
    geom_kwargs["mass"] = mass
  else:
    geom_kwargs.update(contype=0, conaffinity=0, density=0.0, group=2)
  body.add_geom(**geom_kwargs)
  return spec


def get_qwerty_cube_spec(
  cube_size: float = CUBE_HALF_EXTENT,
  mass: float = 0.15,
) -> mujoco.MjSpec:
  """Cube with a per-face texture so its orientation is readable in any viewer."""
  return _make_textured_cube_spec(
    "cube",
    cube_size,
    rgba=(1.0, 1.0, 1.0, 1.0),
    freejoint=True,
    collide=True,
    mass=mass,
  )


def get_qwerty_cube_goal_marker_spec(
  cube_size: float = CUBE_HALF_EXTENT,
) -> mujoco.MjSpec:
  """Visual-only translucent textured cube used as a goal marker.

  Fixed-base (mjlab wraps it as a mocap body); a reorientation command can write its
  pose each step to show a goal orientation.
  """
  return _make_textured_cube_spec(
    "goal",
    cube_size,
    rgba=(1.0, 1.0, 1.0, 0.35),
    freejoint=False,
    collide=False,
  )
