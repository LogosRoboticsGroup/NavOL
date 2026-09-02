"""Launch the canonical NavOL training configuration."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

try:
    from navol.paths import DEFAULT_BASE_CHECKPOINT, DEFAULT_TRAIN_DATASET
    from navol.training import CANONICAL_TRAINING
except ModuleNotFoundError:
    # Support dry-run command generation directly from a source checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "navol"))
    from navol.paths import DEFAULT_BASE_CHECKPOINT, DEFAULT_TRAIN_DATASET
    from navol.training import CANONICAL_TRAINING


def _format_hydra_number(value: float) -> str:
    return str(value).replace("e-0", "e-").replace("e+0", "e+")


def build_training_command(
    *,
    repository_root: Path,
    asset_root: Path,
    python_executable: str,
    num_processes: int,
    run_name: str,
) -> tuple[str, ...]:
    repository_root = Path(repository_root).resolve()
    asset_root = Path(asset_root).expanduser().resolve()
    train_script = repository_root / "scripts" / "rsl_rl" / "train_navdp.py"
    scene_dir = asset_root / "datasets" / DEFAULT_TRAIN_DATASET
    base_checkpoint = asset_root / "models" / DEFAULT_BASE_CHECKPOINT

    return (
        python_executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={num_processes}",
        str(train_script),
        "--distributed",
        "--headless",
        "--task",
        "pointgoal_train_distillation",
        "--scene_dir",
        str(scene_dir),
        "--scene_scale",
        "1.0",
        "--rand_camera",
        "--num_envs",
        str(CANONICAL_TRAINING.environments_per_process),
        "--max_iterations",
        str(CANONICAL_TRAINING.iterations),
        f"agent.run_name={run_name}",
        "agent.save_interval=100",
        f"agent.num_steps_per_env={CANONICAL_TRAINING.rollout_steps}",
        f"agent.algorithm.num_learning_epochs={CANONICAL_TRAINING.learning_epochs}",
        f"agent.algorithm.num_mini_batches={CANONICAL_TRAINING.mini_batches}",
        "agent.algorithm.gradient_length=1",
        f"agent.algorithm.lambda_critic={CANONICAL_TRAINING.lambda_critic}",
        "agent.algorithm.learning_rate="
        f"{_format_hydra_number(CANONICAL_TRAINING.learning_rate)}",
        f"agent.policy.pretrained_model_path={base_checkpoint}",
        "env.events.reset_pose.params.height_range="
        f"[{CANONICAL_TRAINING.camera_height_range[0]},{CANONICAL_TRAINING.camera_height_range[1]}]",
        "env.events.reset_pose.params.pitch_range="
        f"[{CANONICAL_TRAINING.camera_pitch_range[0]},{CANONICAL_TRAINING.camera_pitch_range[1]}]",
        f"env.actions.joint_combined.navmesh_radius={CANONICAL_TRAINING.navmesh_radius}",
        "env.actions.joint_combined.action_type=rand",
        f"env.actions.joint_combined.action_rand_p={CANONICAL_TRAINING.policy_probability}",
        "env.actions.joint_combined.critic_collision_reduction="
        f"{CANONICAL_TRAINING.critic_collision_reduction}",
        "env.actions.joint_combined.search_radius="
        f"{CANONICAL_TRAINING.local_search_radius}",
        f"env.actions.joint_combined.use_mpc={str(CANONICAL_TRAINING.use_mpc).lower()}",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(os.environ.get("NAVOL_ASSET_ROOT", Path(__file__).resolve().parents[2] / "assets")),
    )
    parser.add_argument(
        "--num-processes", type=int, default=CANONICAL_TRAINING.distributed_processes
    )
    parser.add_argument("--run-name", default="navol_canonical")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    command = build_training_command(
        repository_root=repository_root,
        asset_root=args.asset_root,
        python_executable=args.python_executable,
        num_processes=args.num_processes,
        run_name=args.run_name,
    )
    print(shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=repository_root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
