# 🧩 Data preparation

NavOL separates three data products:

1. the processed 50-scene training asset;
2. the processed in-domain and out-of-domain benchmark archives;
3. optional raw benchmark archives for inspection or pipeline research.

Training and evaluation use the processed products. Raw archives are not read
by the runtime launchers.

## Download the processed training asset

```bash
python -m pip install -U huggingface_hub
export NAVOL_ASSET_ROOT=/absolute/path/to/navol-assets

hf download WAboutme/NavOL \
  --repo-type dataset \
  --include "datasets/train/3d_front_scene_50/**" \
  --local-dir "$NAVOL_ASSET_ROOT"
```

This places the training data at
`$NAVOL_ASSET_ROOT/datasets/train/3d_front_scene_50/`. The release contains the
canonical scene selection, merged Isaac USD and textures, reconstruction GLB,
and 50 Habitat-Sim planning meshes. It does not include a fixed reset NPY;
canonical random training does not use one.

## Download the processed benchmark

```bash
python -m pip install -U huggingface_hub

hf download WAboutme/NavOL \
  --repo-type dataset \
  --include "data/benchmarks/processed/*" \
  --local-dir downloads/navol
```

Extract both archives under the same benchmark root:

```bash
export NAVOL_ASSET_ROOT=/absolute/path/to/navol-assets
mkdir -p "$NAVOL_ASSET_ROOT/datasets/benchmarks"

python -m zipfile -e \
  downloads/navol/data/benchmarks/processed/navol_benchmark_in_domain.zip \
  "$NAVOL_ASSET_ROOT/datasets/benchmarks"

python -m zipfile -e \
  downloads/navol/data/benchmarks/processed/navol_benchmark_out_domain.zip \
  "$NAVOL_ASSET_ROOT/datasets/benchmarks"
```

Do not extract the second archive inside the first split. The result must be:

```text
datasets/benchmarks/
├── in_domain/scene_000 ... scene_007/
└── out_domain/scene_000 ... scene_007/
```

Each processed scene contains:

```text
scene_NNN/
├── scene.glb
├── scene.usd
├── navmesh_scene.glb
├── sample_100.npy
└── textures/
```

`scene.glb` is the visual source mesh, `scene.usd` is the Isaac Sim scene,
`navmesh_scene.glb` is the Habitat-Sim planning mesh, and `sample_100.npy`
contains fixed start-goal episodes. Texture references in the released USD
are scene-relative. MeshConverter's generated `config.yaml` is not needed by
training/evaluation and is omitted from the public archives.

## Validate processed archives

Validation streams the ZIP without extracting it and checks the public layout,
portable paths, required files, and episode array shape:

```bash
python scripts/data/prepare_benchmark.py validate \
  downloads/navol/data/benchmarks/processed/navol_benchmark_in_domain.zip \
  --split in_domain

python scripts/data/prepare_benchmark.py validate \
  downloads/navol/data/benchmarks/processed/navol_benchmark_out_domain.zip \
  --split out_domain
```

By default, each archive must contain eight scenes. Every
`sample_100.npy` has shape `(100, 7)` with columns:

```text
start_x, start_y, start_z, goal_x, goal_y, goal_z, initial_yaw
```

## Rebuild an incompatible USD

OpenUSD compatibility can vary between Isaac Sim releases. Keep the downloaded
ZIP as the immutable recovery source and rebuild only the extracted USD.

Inspect the two-stage conversion for one scene:

```bash
python scripts/data/prepare_benchmark.py convert-usd \
  "$NAVOL_ASSET_ROOT/datasets/benchmarks/in_domain/scene_000" \
  --dry-run
```

Then run it from the activated Isaac Lab environment:

```bash
python scripts/data/prepare_benchmark.py convert-usd \
  "$NAVOL_ASSET_ROOT/datasets/benchmarks/in_domain/scene_000"
```

To rebuild a full split, pass the split directory:

```bash
python scripts/data/prepare_benchmark.py convert-usd \
  "$NAVOL_ASSET_ROOT/datasets/benchmarks/in_domain"
```

The command first invokes Blender to import `scene.glb` and create a
material-preserving intermediate USD under `scene_NNN/tmp/`. It then invokes
Isaac Lab's MeshConverter to write `scene_NNN/scene.usd` with
mesh-simplification collision. Blender must be on `PATH`; otherwise use
`--blender-executable /path/to/blender`. The extracted `scene.usd` is replaced,
but the downloaded archive is never modified.

## Download raw benchmark scenes (optional)

Raw archives are useful only when inspecting or extending the preparation
pipeline:

```bash
hf download WAboutme/NavOL \
  --repo-type dataset \
  --include "data/benchmarks/raw/raw_scenes_in_domain.zip" \
  --include "data/benchmarks/raw/raw_scenes_out_domain.zip" \
  --local-dir downloads/navol
```

The raw ZIPs are source material, not drop-in runtime datasets. Extracting them
under `assets/datasets/benchmarks/` will not make evaluation work. Use the
processed archives unless you intentionally need to reproduce or change the
scene conversion.

## How the scene pipeline is organized

Reusable primitives retained in the repository are:

| Stage | Script | Runtime |
|---|---|---|
| Normalize raw GLB axes/scale and export visual/NavMesh GLBs | `scripts/preprocess/3d_front/fix_one_glb.py` or `fix_all_glb.py` | Blender |
| Merge normalized training scenes | `scripts/preprocess/3d_front/merge_scenes.py` | Blender |
| Export a material-preserving intermediate USD | `scripts/preprocess/3d_front/blender_usd.py` | Blender |
| Add Isaac-compatible collision and write the final USD | `scripts/preprocess/convert_mesh2usd.py` | Isaac Lab |
| Sample navigation start-goal pairs | `scripts/preprocess/3d_front/genpoint.py` | Habitat-Sim |

The raw source metadata must provide an `index.json` mapping normalized scene
names to raw mesh paths. The exact normalization height and source-specific
metadata are dataset construction choices and are not inferred automatically.
For this reason, the public release does not claim a one-command raw-to-paper
rebuild. It provides the processing primitives, the exact processed data
contract, and an end-to-end USD repair path.

The training asset expected by `train_navol.py` is:

```text
datasets/train/3d_front_scene_50/
├── index.json
├── selected.json
├── scene.glb
├── usd/
│   ├── config.yaml
│   ├── scene.usd
│   └── textures/
└── navmesh_scenes/scene_*.glb
```

`selected.json` records the canonical 50-scene subset. `usd/config.yaml` is a
portable record of conversion settings, not a runtime input. `sample_100.npy`
is optional and is read only when `sample_from_npy=True` is explicitly enabled.

## Build a portable benchmark archive (maintainers)

Given an immutable processed source ZIP, create a public derived archive at a
new path:

```bash
python -m pip install usd-core

python scripts/data/prepare_benchmark.py build \
  /path/to/processed-in-domain-source.zip \
  /new/output/path/navol_benchmark_in_domain.zip \
  --split in_domain
```

The builder refuses to overwrite its destination, normalizes the archive root,
rewrites absolute USD texture paths to `textures/...`, omits `config.yaml`, and
validates the result. Isaac Sim's Python can be used instead of `usd-core`.

## Troubleshooting

- **Validation reports the wrong scene count:** check that the archive contains
  exactly one split root and eight `scene_*` directories.
- **Validation reports `(99, 7)` or another shape:** regenerate the fixed pair
  file; benchmark evaluation requires exactly 100 rows.
- **USD conversion finds no scenes:** pass either one `scene_NNN` directory or
  a directory whose direct children are named `scene_*`.
- **Textures are missing after conversion:** run the Blender stage against the
  extracted `scene.glb` and keep the scene's `textures/` directory in place.
- **Habitat-Sim cannot load the planning mesh:** verify
  `navmesh_scene.glb` independently from `scene.glb`; they serve different
  consumers.
