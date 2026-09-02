"""Portable locations for NavOL models and datasets."""

from __future__ import annotations

import os
from pathlib import Path


ASSET_ROOT_ENV = "NAVOL_ASSET_ROOT"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ASSET_ROOT = REPOSITORY_ROOT / "assets"
DEFAULT_BASE_CHECKPOINT = "navdp-cross-modal.ckpt"
DEFAULT_POLICY_CHECKPOINT = "checkpoints/navol-mpc-iter1000.pt"
DEFAULT_ROBOT_ASSET = "dingo.usd"
DEFAULT_TRAIN_DATASET = "train/3d_front_scene_50"


def asset_root(explicit_root: str | os.PathLike[str] | None = None) -> Path:
    """Return the configured NavOL asset root.

    An explicit caller value takes precedence over ``NAVOL_ASSET_ROOT``;
    otherwise assets are resolved relative to the repository checkout.
    """

    configured = explicit_root if explicit_root is not None else os.environ.get(ASSET_ROOT_ENV)
    if configured is None:
        return DEFAULT_ASSET_ROOT
    return Path(configured).expanduser().resolve()


def model_path(
    name: str | os.PathLike[str],
    *,
    explicit_root: str | os.PathLike[str] | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve a model filename below ``assets/models``."""

    path = asset_root(explicit_root) / "models" / Path(name)
    path = path.resolve()
    if must_exist and not path.is_file():
        raise FileNotFoundError(
            f"NavOL model '{Path(name).name}' was not found at '{path}'. "
            f"Place it under '<asset-root>/models' or set {ASSET_ROOT_ENV}."
        )
    return path


def policy_checkpoint_path(
    checkpoint: str | os.PathLike[str] | None = None,
    *,
    explicit_root: str | os.PathLike[str] | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve the trained NavOL policy checkpoint.

    A CLI-provided path may live anywhere. When it is omitted, the canonical
    release filename is
    ``assets/models/checkpoints/navol-mpc-iter1000.pt``.
    """

    if checkpoint is None:
        return model_path(DEFAULT_POLICY_CHECKPOINT, explicit_root=explicit_root, must_exist=must_exist)

    path = Path(checkpoint).expanduser().resolve()
    if must_exist and not path.is_file():
        raise FileNotFoundError(
            f"NavOL policy checkpoint was not found at '{path}'. "
            "Pass an existing file with --checkpoint_path or place the canonical "
            f"checkpoint under '<asset-root>/models/{DEFAULT_POLICY_CHECKPOINT}' "
            f"and set {ASSET_ROOT_ENV}."
        )
    return path


def robot_path(
    name: str | os.PathLike[str],
    *,
    explicit_root: str | os.PathLike[str] | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve a robot asset below ``assets/robots``."""

    path = (asset_root(explicit_root) / "robots" / Path(name)).resolve()
    if must_exist and not path.is_file():
        raise FileNotFoundError(
            f"NavOL robot asset '{Path(name).name}' was not found at '{path}'. "
            f"Place it under '<asset-root>/robots' or set {ASSET_ROOT_ENV}."
        )
    return path


def dataset_path(
    relative_path: str | os.PathLike[str],
    *,
    explicit_root: str | os.PathLike[str] | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve a dataset directory below ``assets/datasets``."""

    path = (asset_root(explicit_root) / "datasets" / Path(relative_path)).resolve()
    if must_exist and not path.is_dir():
        raise FileNotFoundError(
            f"NavOL dataset '{relative_path}' was not found at '{path}'. "
            f"Place it under '<asset-root>/datasets' or set {ASSET_ROOT_ENV}."
        )
    return path
