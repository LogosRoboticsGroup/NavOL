# 🏋️ Training NavOL

This directory contains the supported training launcher. It builds a complete
command for `scripts/rsl_rl/train_navdp.py`; the low-level script is not the
recommended public entry point because it also exposes historical research
defaults.

## Requirements

- Linux and a CUDA-capable GPU setup;
- Isaac Sim 4.5 and Isaac Lab 2.1;
- the three local Python packages installed as described in the root README;
- Habitat-Sim available in the same environment;
- the Dingo USD, 50-scene training set, and NavDP initialization checkpoint.

## Download the released training inputs

```bash
python -m pip install -U huggingface_hub
export NAVOL_ASSET_ROOT=/absolute/path/to/navol-assets

hf download WAboutme/NavOL \
  --repo-type dataset \
  --include "models/navdp-cross-modal.ckpt" \
  --include "robots/dingo.usd" \
  --include "datasets/train/3d_front_scene_50/**" \
  --local-dir "$NAVOL_ASSET_ROOT"
```

This single command downloads the initialization checkpoint, robot asset, and
processed scene data required by the canonical training launcher. The four
trained NavOL `.pt` policy checkpoints under `models/checkpoints/` are for
evaluation and are not required to start training.

## Required files

With `NAVOL_ASSET_ROOT=/path/to/navol-assets`, training expects:

```text
/path/to/navol-assets/
├── models/navdp-cross-modal.ckpt
├── robots/dingo.usd
└── datasets/train/3d_front_scene_50/
    ├── index.json
    ├── selected.json
    ├── scene.glb
    ├── usd/
    │   ├── config.yaml
    │   ├── scene.usd
    │   └── textures/
    └── navmesh_scenes/
        ├── scene_000.glb
        └── ...
```

`index.json` describes the available entries, and `selected.json` records the
canonical 50-scene selection used by the launcher. The merged `usd/scene.usd`
and its `textures/` directory are the Isaac scene; `navmesh_scenes/*.glb` are
the corresponding Habitat-Sim planning meshes. `scene.glb` is retained as the
portable source for rebuilding an incompatible USD, while `usd/config.yaml`
is an informational, path-sanitized record of the MeshConverter settings and
is not read during training.

The released training asset does not contain `sample_100.npy`. Canonical
random training sets `sample_from_npy=False`; provide a compatible fixed reset
array only when explicitly enabling `sample_from_npy=True`.

The `navdp-cross-modal.ckpt` file initializes the visual navigation backbone;
it is not the final NavOL policy checkpoint. Evaluation uses a trained `.pt`
file under `models/checkpoints/` instead.

## Inspect and run

Always inspect the resolved command first:

```bash
python scripts/train/train_navol.py --dry-run
```

Start the canonical job:

```bash
python scripts/train/train_navol.py
```

If assets are outside the repository and no environment variable is set:

```bash
python scripts/train/train_navol.py \
  --asset-root /path/to/navol-assets
```

The canonical launch uses eight distributed processes. A one-process command
is useful for integration testing, but it is not the paper-scale run:

```bash
python scripts/train/train_navol.py \
  --num-processes 1 \
  --run-name navol_integration
```

## Canonical configuration

| Parameter | Value |
|---|---:|
| Distributed processes | 8 |
| Environments per process | 32 |
| Rollout steps per environment | 128 |
| Learning epochs | 10 |
| Mini-batches | 16 |
| Global mini-batch size | 2048 |
| Training iterations | 1000 |
| Learning rate | `1e-5` |
| Critic coefficient | `0.1` |
| Policy/expert action probability | `0.8 / 0.2` |
| Camera height | `(0.25, 1.25)` m |
| Camera pitch | `(-30, 0)` degrees |
| MPC | enabled |
| Camera randomization | enabled |

These values live in `source/navol/navol/training.py` and are supplied again
as explicit Hydra overrides by `train_navol.py`. The default critic collision
reduction is `mean` to preserve compatibility with the released checkpoint.

## Outputs and restart behavior

Runs are written below:

```text
logs/rsl_rl/dingo_pointgoal_distillation/<run-directory>/
```

The directory contains serialized environment/agent configuration and model
checkpoints at the configured save interval. Resume behavior is controlled by
the RSL-RL configuration and is not enabled by the canonical launcher.

## Common failures

- **`dingo.usd` not found:** re-run the training-input download above or place
  it at `$NAVOL_ASSET_ROOT/robots/dingo.usd`.
- **`index.json`, `selected.json`, or `usd/scene.usd` not found:** the training
  asset was downloaded one directory too deep; make `3d_front_scene_50/` the
  directory that directly contains `index.json` and `selected.json`.
- **`navdp-cross-modal.ckpt` not found:** re-run the training-input download
  above; the file belongs in `$NAVOL_ASSET_ROOT/models/`, not
  `models/checkpoints/`.
- **Habitat-Sim import error:** install a build matching the active Python and
  CUDA versions before launching Isaac Sim.
- **CUDA out of memory:** use fewer processes/environments for an integration
  run, then reproduce the canonical configuration on suitable hardware.
