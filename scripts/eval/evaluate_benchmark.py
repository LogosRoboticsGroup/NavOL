"""Evaluate NavOL on every scene in the public benchmark splits."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

try:
    from navol.paths import DEFAULT_POLICY_CHECKPOINT
except ModuleNotFoundError:
    # Support dry-run command generation directly from a source checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "navol"))
    from navol.paths import DEFAULT_POLICY_CHECKPOINT


VALID_SPLITS = ("in_domain", "out_domain")


def build_evaluation_commands(
    *,
    repository_root: Path,
    asset_root: Path,
    output_root: Path,
    python_executable: str,
    splits: Sequence[str],
    checkpoint: Path | None = None,
) -> list[tuple[str, ...]]:
    repository_root = Path(repository_root).resolve()
    asset_root = Path(asset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    eval_script = repository_root / "scripts" / "rsl_rl" / "eval_navdp.py"
    if checkpoint is None:
        checkpoint = asset_root / "models" / DEFAULT_POLICY_CHECKPOINT
    checkpoint = Path(checkpoint).expanduser().resolve()

    invalid_splits = sorted(set(splits) - set(VALID_SPLITS))
    if invalid_splits:
        raise ValueError(f"unsupported benchmark splits: {invalid_splits}")

    commands: list[tuple[str, ...]] = []
    for split in splits:
        scene_root = asset_root / "datasets" / "benchmarks" / split
        for scene_index in range(8):
            commands.append(
                (
                    python_executable,
                    str(eval_script),
                    "--headless",
                    "--task",
                    "pointgoal_eval_distillation",
                    "--scene_dir",
                    str(scene_root),
                    "--scene_index",
                    str(scene_index),
                    "--scene_scale",
                    "1.0",
                    "--num_envs",
                    "1",
                    "--num_episodes",
                    "100",
                    "--save_path",
                    str(output_root / split / f"scene_{scene_index:03d}"),
                    "--checkpoint_path",
                    str(checkpoint),
                    "--use_mpc",
                )
            )
    return commands


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(os.environ.get("NAVOL_ASSET_ROOT", Path(__file__).resolve().parents[2] / "assets")),
    )
    parser.add_argument("--output-root", type=Path, default=Path("results") / "benchmark")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="policy checkpoint (default: <asset-root>/models/checkpoints/navol-mpc-iter1000.pt)",
    )
    parser.add_argument("--split", action="append", choices=VALID_SPLITS, dest="splits")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    commands = build_evaluation_commands(
        repository_root=repository_root,
        asset_root=args.asset_root,
        output_root=args.output_root,
        python_executable=args.python_executable,
        splits=tuple(args.splits or VALID_SPLITS),
        checkpoint=args.checkpoint,
    )
    for command in commands:
        print(shlex.join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=repository_root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
