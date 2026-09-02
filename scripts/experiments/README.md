# Research recipes

This directory retains ablation and analysis recipes from NavOL research.
They are not the public defaults and are not exercised by the lightweight CI.

Use these supported entry points first:

- data: `scripts/data/prepare_benchmark.py`;
- training: `scripts/train/train_navol.py`;
- evaluation: `scripts/eval/evaluate_benchmark.py`.

The scripts under `ablations/` override canonical parameters intentionally.
Review every command with the current asset layout and target hardware before
running it. Historical cluster launchers with private paths and unavailable
checkpoint names are intentionally excluded from the public release; they can
still be recovered from the original research repository when needed for
forensic comparison.
