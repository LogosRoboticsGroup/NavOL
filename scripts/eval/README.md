# 📊 Benchmark evaluation

`evaluate_benchmark.py` is the supported public evaluation launcher. It runs
every selected scene through the low-level Isaac Lab runtime at
`scripts/rsl_rl/eval_navdp.py`.

## Download the released runtime assets

```bash
python -m pip install -U huggingface_hub
export NAVOL_ASSET_ROOT=/absolute/path/to/navol-assets

hf download WAboutme/NavOL \
  --repo-type dataset \
  --include "models/checkpoints/*" \
  --include "robots/dingo.usd" \
  --local-dir "$NAVOL_ASSET_ROOT"
```

Download and extract the processed benchmark archives as described in the
[root README](../../README.md#3-download-released-artifacts) before evaluation.

## Required files

```text
${NAVOL_ASSET_ROOT}/
├── models/checkpoints/navol-mpc-iter1000.pt
├── robots/dingo.usd
└── datasets/benchmarks/
    ├── in_domain/
    │   ├── scene_000/
    │   │   ├── scene.glb
    │   │   ├── scene.usd
    │   │   ├── navmesh_scene.glb
    │   │   ├── sample_100.npy
    │   │   └── textures/
    │   └── ... scene_007/
    └── out_domain/
        └── scene_000 ... scene_007/
```

Keep only the eight `scene_*` directories at each split root; the low-level
evaluator selects scenes by their sorted directory order. Each
`sample_100.npy` must have shape `(100, 7)`.

## Inspect and run

Print all 16 commands without starting Isaac Sim:

```bash
python scripts/eval/evaluate_benchmark.py --dry-run
```

Evaluate both splits:

```bash
python scripts/eval/evaluate_benchmark.py
```

Evaluate one split:

```bash
python scripts/eval/evaluate_benchmark.py --split in_domain
```

Use a different trained policy without changing the asset directory:

```bash
python scripts/eval/evaluate_benchmark.py \
  --split out_domain \
  --checkpoint /path/to/model_1000_navdp.pt \
  --output-root results/custom_checkpoint
```

The default checkpoint is
`$NAVOL_ASSET_ROOT/models/checkpoints/navol-mpc-iter1000.pt`. MPC and critic
ranking are enabled. Each scene uses one environment and 100 episodes so the
fixed start-goal rows are consumed once.

## Outputs

Each low-level run creates a scene result directory below `--output-root` and
writes `metric_pointgoal_eval_distillation.csv`. The CSV is updated as
episodes finish. Video output is disabled by the public wrapper; use the
low-level `--save` option only when running a custom visualization job.

## Metric behavior

Episode termination and navigation success are separate values. A termination
event triggers metric finalization; success is then computed independently by
the current low-level evaluator from planar goal distance with a strict
one-metre threshold. The public wrapper preserves this implementation and
does not treat every `done` episode as successful.

## Common failures

- **Checkpoint not found:** check `--checkpoint` or the default
  `models/checkpoints/navol-mpc-iter1000.pt` path.
- **Empty USD/NPY path in the log:** a scene directory is missing `scene.usd`
  or `sample_100.npy`; validate or rebuild it using the data guide.
- **Wrong scene selected:** remove unrelated files/directories from the split
  root and keep `scene_000` through `scene_007`.
- **USD parse or texture error:** keep the original archive and regenerate
  the extracted scene's USD from `scene.glb`; see `scripts/data/README.md`.
- **Habitat-Sim/NavMesh error:** verify that `navmesh_scene.glb` exists and
  Habitat-Sim matches the current Python/CUDA environment.
