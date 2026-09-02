# NavOL assets

This directory is the default `NAVOL_ASSET_ROOT`. Large payloads are ignored
by Git; keep this README while placing downloaded files in the layout below:

```text
assets/
├── models/
│   ├── navdp-cross-modal.ckpt
│   └── checkpoints/
│       ├── navol-mpc-iter1000.pt
│       ├── navol-mpc-iter500.pt
│       ├── navol-nompc-iter200.pt
│       └── navol-rollout128-iter100.pt
├── datasets/
│   ├── train/3d_front_scene_50/
│   │   ├── index.json
│   │   ├── selected.json
│   │   ├── scene.glb
│   │   ├── usd/scene.usd and textures/
│   │   └── navmesh_scenes/scene_*.glb
│   └── benchmarks/
│       ├── in_domain/scene_000 ... scene_007/
│       └── out_domain/scene_000 ... scene_007/
└── robots/dingo.usd
```

Download the released training inputs directly into this root:

```bash
export NAVOL_ASSET_ROOT=/absolute/path/to/navol-assets

hf download WAboutme/NavOL \
  --repo-type dataset \
  --include "models/navdp-cross-modal.ckpt" \
  --include "robots/dingo.usd" \
  --include "datasets/train/3d_front_scene_50/**" \
  --local-dir "$NAVOL_ASSET_ROOT"
```

The released training asset does not include `sample_100.npy`; canonical
random training uses `sample_from_npy=False`. The command above also downloads
the NavDP initialization checkpoint and Dingo USD to the paths shown above.

The processed benchmark archives include a sanitized `scene.usd` in every
scene. If that USD is incompatible with the local Isaac Sim/OpenUSD version,
regenerate it from the included `scene.glb` with
`scripts/data/prepare_benchmark.py convert-usd`; the command does not modify
the downloaded archive. See `scripts/data/README.md` for exact commands.
