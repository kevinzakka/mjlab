# Known bugs to track

## `Entity.write_mocap_pose(env_ids=None)` breaks the broadcast

**Where:** `src/mjlab/entity/data.py:205-214`

**Symptom:** Calling `entity.write_mocap_pose_to_sim(pose)` without an explicit
`env_ids` (i.e. "all envs") raises:

```
RuntimeError: expand(torch.cuda.FloatTensor{[N, 1, 3]}, size=[N, 3]):
the number of sizes provided (2) must be greater or equal to the number of
dimensions in the tensor (3)
```

Single-env runs happen to work (both sides collapse to `[1, 3]`); multi-env
training blows up immediately.

**Root cause:** `_resolve_env_ids` (data.py:227) is inconsistent between the
tensor and None paths:

```python
def _resolve_env_ids(self, env_ids):
  if env_ids is None:
    return slice(None)            # plain 1-D slice
  if isinstance(env_ids, torch.Tensor):
    return env_ids[:, None]       # reshaped to [N, 1] (extra dim!)
  return env_ids
```

`write_mocap_pose` is written around the tensor path:

```python
self.data.mocap_pos[env_ids, self.indexing.mocap_id] = pose[:, 0:3].unsqueeze(1)
```

With `env_ids` shaped `[N, 1]` and a scalar `mocap_id`, the LHS slice has shape
`[N, 1, 3]`, matching the RHS `unsqueeze(1)`. With `env_ids = slice(None)` the
mocap axis collapses to `[N, 3]` and the broadcast fails. `events.py:95` (the
fixed-base mocap reset path) always passes a tensor so it never trips, but any
other caller using the documented `env_ids: ... | None = None` default does.

**Workaround in use today:** Caller passes an explicit
`torch.arange(num_envs, device=...)` (cached in `__init__`). See
`ReorientationCommand.__init__` and `_update_command` in
`src/mjlab/tasks/manipulation/mdp/commands.py`.

**Possible fixes:**
1. Make `_resolve_env_ids(None)` return `torch.arange(num_envs)[:, None]` so
   both branches produce the same `[N, 1]` shape. Cheapest, but needs the
   manager to know `num_envs` (it does via `self.num_envs`).
2. Drop the `unsqueeze(1)` and rely on plain broadcasting for both shapes.
   Risk: silently changes behavior in `events.py` callers if the shape match
   was load-bearing somewhere.
3. Document the `env_ids=None` form as unsupported for mocap writes and raise
   a clear error.

Option (1) is the right fix and centralizes the invariant. Same bug likely
affects any other `Entity.write_*` helper that uses the same pattern; audit
them when fixing.


## Cube textures don't render in `--viewer viser`

**Symptom:** With `play --viewer viser`, the reorientation cube and the
goal-marker ghost cube render as flat-shaded boxes — the per-face PNG textures
under `src/mjlab/tasks/manipulation/assets/reorientation_cube_textures/` only
show up in the native MuJoCo viewer. Without the textures the cube's
orientation is unreadable in the browser, which defeats the whole point of
the textured-goal change.

**Open question:** is this a bug in the standalone mjviser viewer
(`../mjviser`, browser-based MuJoCo viewer used by mjlab's `ViserPlayViewer`)
or a more general gap in viser's MuJoCo-asset support — e.g. file-backed
`type="cube"` textures (`cubefiles=[...]`) not being uploaded to the browser?
Worth reproducing with a minimal vanilla `mujoco.MjSpec` cube + cube texture
in mjviser before deciding where the fix lives. The asset paths are absolute,
so it's not a working-directory issue.

**Repro:**
```
uv run python -m mjlab.scripts.play Mjlab-Reorient-Cube-Sharpa \
  --viewer viser --checkpoint-file <any sharpa_reorient checkpoint>.pt
```
Compare against the same command with `--viewer native`.
