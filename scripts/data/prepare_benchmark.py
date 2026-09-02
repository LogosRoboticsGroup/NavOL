"""Build, validate, and repair portable NavOL benchmark archives.

The source archives are treated as immutable. Derived archives retain a
sanitized ``scene.usd`` with scene-relative texture references and omit the
unused MeshConverter ``config.yaml`` sidecar.
"""

from __future__ import annotations

import argparse
import gc
import io
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import numpy as np


VALID_SPLITS = ("in_domain", "out_domain")
EXCLUDED_GENERATED_FILES = frozenset(("config.yaml",))
REQUIRED_SCENE_FILES = frozenset(
    ("scene.glb", "scene.usd", "navmesh_scene.glb", "sample_100.npy")
)
CONVERSION_SOURCE_FILES = REQUIRED_SCENE_FILES - {"scene.usd"}
FORBIDDEN_BINARY_FRAGMENTS = (b"/" + b"inspire" + b"/", b"new" + b"dog")
SCAN_SUFFIXES = (".glb", ".npy", ".yaml", ".yml", ".json", ".txt", ".md", ".usd")
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
UsdSanitizer = Callable[[bytes, frozenset[str]], bytes]


class ArchiveSummary(NamedTuple):
    archive: Path
    split: str
    scene_count: int
    file_count: int


class ConversionCommand(NamedTuple):
    scene_id: str
    stage: str
    argv: tuple[str, ...]


def _normalized_member_parts(name: str) -> tuple[str, ...]:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = path.parts
    if (
        not parts
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in parts)
        or ":" in parts[0]
    ):
        raise ValueError(f"unsafe ZIP member: {name}")
    return parts


def _validate_split(split: str) -> None:
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")


def _portable_name(name: str, source_root: str, split: str) -> str | None:
    parts = _normalized_member_parts(name)
    if parts[0] != source_root:
        raise ValueError(f"archive contains multiple roots: {source_root!r} and {parts[0]!r}")
    if len(parts) == 1:
        return None
    if parts[-1] in EXCLUDED_GENERATED_FILES:
        return None
    return PurePosixPath(split, *parts[1:]).as_posix()


def _portable_zip_info(source: zipfile.ZipInfo, filename: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=filename, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = source.external_attr
    info.create_system = source.create_system
    info.flag_bits = source.flag_bits & ~0x08
    return info


def _copy_member(
    source_archive: zipfile.ZipFile,
    destination_archive: zipfile.ZipFile,
    source_info: zipfile.ZipInfo,
    destination_name: str,
) -> None:
    destination_info = _portable_zip_info(source_info, destination_name)
    with source_archive.open(source_info, "r") as source_stream:
        with destination_archive.open(destination_info, "w", force_zip64=True) as destination_stream:
            shutil.copyfileobj(source_stream, destination_stream, length=8 * 1024 * 1024)


def _write_member_payload(
    destination_archive: zipfile.ZipFile,
    source_info: zipfile.ZipInfo,
    destination_name: str,
    payload: bytes,
) -> None:
    destination_info = _portable_zip_info(source_info, destination_name)
    destination_archive.writestr(destination_info, payload)


def _is_absolute_asset_path(path: str) -> bool:
    return path.startswith(("/", "\\")) or WINDOWS_ABSOLUTE_PATH.match(path) is not None


def portable_texture_asset_path(authored_path: str, scene_files: set[str] | frozenset[str]) -> str:
    """Map an absolute scene texture reference to its portable archive path."""

    normalized = authored_path.replace("\\", "/")
    marker = "/textures/"
    marker_index = normalized.lower().find(marker)
    if marker_index < 0:
        raise ValueError(f"unsupported absolute USD asset path: {authored_path}")
    portable = "textures/" + normalized[marker_index + len(marker) :]
    if portable not in scene_files:
        raise ValueError(
            f"USD texture '{portable}' is not present in the scene archive"
        )
    return portable


def _rewrite_usd_file(
    source_path: Path,
    destination_path: Path,
    scene_files: frozenset[str],
    Sdf,
    Usd,
) -> bytes:
    """Rewrite one USD file and release its OpenUSD handles before returning."""

    stage = Usd.Stage.Open(str(source_path), load=Usd.Stage.LoadNone)
    if stage is None:
        raise ValueError("OpenUSD could not open scene.usd")

    for prim in stage.TraverseAll():
        for attribute in prim.GetAttributes():
            if not attribute.HasAuthoredValueOpinion():
                continue
            value = attribute.Get()
            if isinstance(value, Sdf.AssetPath):
                authored_path = value.path
                if _is_absolute_asset_path(authored_path):
                    portable = portable_texture_asset_path(authored_path, scene_files)
                    attribute.Set(Sdf.AssetPath(portable))
            elif attribute.GetTypeName() == Sdf.ValueTypeNames.AssetArray and value is not None:
                rewritten = []
                changed = False
                for asset_path in value:
                    authored_path = asset_path.path
                    if _is_absolute_asset_path(authored_path):
                        authored_path = portable_texture_asset_path(authored_path, scene_files)
                        changed = True
                    rewritten.append(Sdf.AssetPath(authored_path))
                if changed:
                    attribute.Set(rewritten)

    exported = stage.GetRootLayer().ExportToString()
    for authored_path in re.findall(r"@([^@]+)@", exported):
        if _is_absolute_asset_path(authored_path):
            raise ValueError(
                f"scene.usd still contains an absolute asset path: {authored_path}"
            )
    if not stage.GetRootLayer().Export(str(destination_path), args={"format": "usdc"}):
        raise ValueError("OpenUSD could not export the sanitized scene.usd")
    return destination_path.read_bytes()


def sanitize_usd_payload(payload: bytes, scene_files: frozenset[str]) -> bytes:
    """Return a USDC payload whose absolute texture paths are scene-relative.

    OpenUSD is imported lazily so archive validation and conversion planning
    remain usable in dependency-light environments.
    """

    try:
        from pxr import Sdf, Usd
    except ImportError as error:
        raise RuntimeError(
            "building a release archive requires OpenUSD; run this command with "
            "Isaac Sim Python or install the usd-core package"
        ) from error

    with tempfile.TemporaryDirectory(prefix="navol-usd-") as temporary_directory:
        source_path = Path(temporary_directory) / "source.usd"
        destination_path = Path(temporary_directory) / "scene.usd"
        source_path.write_bytes(payload)
        sanitized = _rewrite_usd_file(
            source_path,
            destination_path,
            scene_files,
            Sdf,
            Usd,
        )
        gc.collect()
        return sanitized


def build_portable_archive(
    source: Path,
    destination: Path,
    split: str,
    *,
    usd_sanitizer: UsdSanitizer | None = None,
) -> ArchiveSummary:
    """Create a portable derived ZIP without modifying ``source``.

    The destination must not already exist. USD texture paths are sanitized,
    the unused config sidecar is omitted, and the archive root is normalized
    to ``split``.
    """

    _validate_split(split)
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source archive not found: {source}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite destination: {destination}")
    if source == destination:
        raise ValueError("source and destination must be different files")
    if usd_sanitizer is None:
        usd_sanitizer = sanitize_usd_payload

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(source, "r") as source_archive:
            source_infos = [info for info in source_archive.infolist() if not info.is_dir()]
            if not source_infos:
                raise ValueError(f"source archive is empty: {source}")
            all_parts = [_normalized_member_parts(info.filename) for info in source_infos]
            source_roots = {parts[0] for parts in all_parts}
            if len(source_roots) != 1:
                raise ValueError(f"archive must have exactly one root, found: {sorted(source_roots)}")
            source_root = next(iter(source_roots))

            members = []
            for info in source_infos:
                destination_name = _portable_name(info.filename, source_root, split)
                if destination_name is not None:
                    members.append((destination_name, info))

            scene_files: dict[str, set[str]] = {}
            for destination_name, _ in members:
                parts = PurePosixPath(destination_name).parts
                if len(parts) >= 3:
                    scene_files.setdefault(parts[1], set()).add(
                        PurePosixPath(*parts[2:]).as_posix()
                    )

            with zipfile.ZipFile(
                destination,
                "x",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
                allowZip64=True,
            ) as destination_archive:
                for destination_name, source_info in sorted(members, key=lambda item: item[0]):
                    parts = PurePosixPath(destination_name).parts
                    if parts[-1] == "scene.usd":
                        scene_id = parts[1]
                        sanitized = usd_sanitizer(
                            source_archive.read(source_info),
                            frozenset(scene_files[scene_id]),
                        )
                        _write_member_payload(
                            destination_archive,
                            source_info,
                            destination_name,
                            sanitized,
                        )
                    else:
                        _copy_member(source_archive, destination_archive, source_info, destination_name)

        return validate_portable_archive(destination, split)
    except BaseException:
        if destination.exists():
            destination.unlink()
        raise


def _scan_stream_for_forbidden_fragments(stream) -> set[bytes]:
    found: set[bytes] = set()
    tail = b""
    while True:
        chunk = stream.read(8 * 1024 * 1024)
        if not chunk:
            break
        lowered = tail + chunk.lower()
        found.update(fragment for fragment in FORBIDDEN_BINARY_FRAGMENTS if fragment in lowered)
        tail = lowered[-64:]
    return found


def validate_portable_archive(
    archive_path: Path,
    split: str,
    *,
    expected_scenes: int | None = None,
) -> ArchiveSummary:
    """Validate the public archive layout and fixed episode arrays."""

    _validate_split(split)
    archive_path = Path(archive_path).expanduser().resolve()
    scene_files: dict[str, set[str]] = {}
    file_count = 0

    with zipfile.ZipFile(archive_path, "r") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            file_count += 1
            parts = _normalized_member_parts(info.filename)
            if len(parts) < 3 or parts[0] != split or not parts[1].startswith("scene_"):
                raise ValueError(f"unexpected portable archive member: {info.filename}")
            scene_id = parts[1]
            relative_name = PurePosixPath(*parts[2:]).as_posix()
            scene_files.setdefault(scene_id, set()).add(relative_name)
            if parts[-1] in EXCLUDED_GENERATED_FILES:
                raise ValueError(f"unused generated file is not portable: {info.filename}")
            if info.filename.lower().endswith(SCAN_SUFFIXES):
                with archive.open(info, "r") as stream:
                    found = _scan_stream_for_forbidden_fragments(stream)
                if found:
                    labels = ", ".join(sorted(item.decode("ascii") for item in found))
                    raise ValueError(f"non-portable content in {info.filename}: {labels}")

        if not scene_files:
            raise ValueError("portable archive contains no scene directories")
        if expected_scenes is not None and len(scene_files) != expected_scenes:
            raise ValueError(f"expected {expected_scenes} scenes, found {len(scene_files)}")

        for scene_id, files in sorted(scene_files.items()):
            missing = REQUIRED_SCENE_FILES - files
            if missing:
                raise ValueError(f"{scene_id} is missing required files: {sorted(missing)}")
            pair_name = f"{split}/{scene_id}/sample_100.npy"
            pairs = np.load(io.BytesIO(archive.read(pair_name)), allow_pickle=False)
            if pairs.shape != (100, 7):
                raise ValueError(f"{pair_name}: expected shape (100, 7), found {pairs.shape}")
            if not np.isfinite(pairs).all():
                raise ValueError(f"{pair_name}: start-goal pairs contain non-finite values")

    return ArchiveSummary(archive_path, split, len(scene_files), file_count)


def plan_usd_conversion(
    dataset_root: Path,
    converter_script: Path,
    blender_converter_script: Path,
    python_executable: str,
    blender_executable: str,
) -> list[ConversionCommand]:
    """Return Blender and Isaac Lab conversion commands without executing them."""

    dataset_root = Path(dataset_root).expanduser().resolve()
    converter_script = Path(converter_script).expanduser().resolve()
    blender_converter_script = Path(blender_converter_script).expanduser().resolve()
    if not converter_script.is_file():
        raise FileNotFoundError(f"converter script not found: {converter_script}")
    if not blender_converter_script.is_file():
        raise FileNotFoundError(f"Blender converter script not found: {blender_converter_script}")

    if dataset_root.is_dir() and dataset_root.name.startswith("scene_"):
        scene_dirs = [dataset_root]
    else:
        scene_dirs = sorted(path for path in dataset_root.glob("scene_*") if path.is_dir())

    commands: list[ConversionCommand] = []
    for scene_dir in scene_dirs:
        for required in CONVERSION_SOURCE_FILES:
            required_path = scene_dir / required
            if not required_path.is_file():
                raise FileNotFoundError(f"missing benchmark scene file: {required_path}")
        intermediate_usd = (scene_dir / "tmp" / "scene.usd").resolve()
        commands.extend(
            (
                ConversionCommand(
                    scene_dir.name,
                    "blender",
                    (
                        blender_executable,
                        "--background",
                        "--python",
                        str(blender_converter_script),
                        "--",
                        "--input",
                        str((scene_dir / "scene.glb").resolve()),
                        "--output",
                        str(intermediate_usd),
                    ),
                ),
                ConversionCommand(
                    scene_dir.name,
                    "isaac",
                    (
                        python_executable,
                        str(converter_script),
                        str(intermediate_usd),
                        str((scene_dir / "scene.usd").resolve()),
                        "--collision-approximation",
                        "meshSimplification",
                        "--headless",
                    ),
                ),
            )
        )
    if not commands:
        raise ValueError(f"no scene_* directories found under {dataset_root}")
    return commands


def run_usd_conversion(commands: Iterable[ConversionCommand], *, dry_run: bool) -> None:
    for command in commands:
        print(" ".join(command.argv))
        if not dry_run:
            subprocess.run(command.argv, check=True)


def _default_converter_script() -> Path:
    return Path(__file__).resolve().parents[1] / "preprocess" / "convert_mesh2usd.py"


def _default_blender_converter_script() -> Path:
    return Path(__file__).resolve().parents[1] / "preprocess" / "3d_front" / "blender_usd.py"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="build a portable derived ZIP")
    build_parser.add_argument("source", type=Path)
    build_parser.add_argument("destination", type=Path)
    build_parser.add_argument("--split", required=True, choices=VALID_SPLITS)

    validate_parser = subparsers.add_parser("validate", help="validate a portable derived ZIP")
    validate_parser.add_argument("archive", type=Path)
    validate_parser.add_argument("--split", required=True, choices=VALID_SPLITS)
    validate_parser.add_argument("--expected-scenes", type=int, default=8)

    convert_parser = subparsers.add_parser("convert-usd", help="generate local USD files with Isaac Lab")
    convert_parser.add_argument("dataset_root", type=Path)
    convert_parser.add_argument("--converter", type=Path, default=_default_converter_script())
    convert_parser.add_argument(
        "--blender-converter",
        type=Path,
        default=_default_blender_converter_script(),
    )
    convert_parser.add_argument("--python-executable", default=sys.executable)
    convert_parser.add_argument("--blender-executable", default="blender")
    convert_parser.add_argument("--dry-run", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "build":
        summary = build_portable_archive(args.source, args.destination, args.split)
    elif args.command == "validate":
        summary = validate_portable_archive(
            args.archive,
            args.split,
            expected_scenes=args.expected_scenes,
        )
    else:
        commands = plan_usd_conversion(
            args.dataset_root,
            args.converter,
            args.blender_converter,
            args.python_executable,
            args.blender_executable,
        )
        run_usd_conversion(commands, dry_run=args.dry_run)
        return 0
    print(f"validated {summary.scene_count} scenes in {summary.archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
