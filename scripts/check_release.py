"""Run dependency-light checks over NavOL's public release surface."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import NamedTuple


REQUIRED_FILES = (
    Path("README.md"),
    Path("docs/README_zh-CN.md"),
    Path("LICENSE"),
    Path("THIRD_PARTY_NOTICES.md"),
    Path("assets/README.md"),
    Path("scripts/data/README.md"),
    Path("scripts/train/README.md"),
    Path("scripts/eval/README.md"),
    Path("scripts/data/prepare_benchmark.py"),
    Path("scripts/train/train_navol.py"),
    Path("scripts/eval/evaluate_benchmark.py"),
    Path("source/navol/pyproject.toml"),
    Path("source/navol/README.md"),
    Path("source/navol/setup.py"),
    Path("source/navol/config/extension.toml"),
    Path("source/navol/LICENSE"),
    Path("source/navol/THIRD_PARTY_NOTICES.md"),
    Path(".pre-commit-config.yaml"),
    Path(".flake8"),
    Path(".github/workflows/ci.yml"),
)
PUBLIC_ROOT_SCRIPT_ALLOWLIST = frozenset()
ROOT_SCRIPT_SUFFIXES = frozenset((".bat", ".ps1", ".py", ".sh"))
TEXT_SUFFIXES = frozenset(
    (".cfg", ".ini", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml")
)
SCANNED_DIRECTORIES = (
    Path(".github"),
    Path(".vscode"),
    Path("scripts/data"),
    Path("scripts/eval"),
    Path("scripts/experiments"),
    Path("scripts/preprocess"),
    Path("scripts/rsl_rl"),
    Path("scripts/train"),
    Path("source/navol"),
)
SKIPPED_PARTS = frozenset(("__pycache__", "build", "dist"))
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
FORBIDDEN_FRAGMENTS = (
    "/" + "inspire" + "/",
    "\\" + "inspire" + "\\",
    "new" + "dog",
)


class ReleaseIssue(NamedTuple):
    path: Path
    line: int | None
    message: str


def _iter_public_text_files(root: Path) -> Iterable[Path]:
    yielded: set[Path] = set()
    for required in REQUIRED_FILES:
        path = root / required
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            yielded.add(path)
            yield path

    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES and path not in yielded:
            yielded.add(path)
            yield path

    for relative_directory in SCANNED_DIRECTORIES:
        directory = root / relative_directory
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in TEXT_SUFFIXES
                and not any(part in SKIPPED_PARTS or part.endswith(".egg-info") for part in path.parts)
                and path not in yielded
            ):
                yielded.add(path)
                yield path


def check_release(repository_root: Path) -> list[ReleaseIssue]:
    """Return deterministic issues without reading asset payloads or archives."""

    root = Path(repository_root).expanduser().resolve()
    issues: list[ReleaseIssue] = []
    for relative_path in REQUIRED_FILES:
        if not (root / relative_path).is_file():
            issues.append(ReleaseIssue(relative_path, None, f"missing required file: {relative_path}"))

    for path in sorted(root.iterdir()):
        if (
            path.is_file()
            and path.suffix.lower() in ROOT_SCRIPT_SUFFIXES
            and path.name not in PUBLIC_ROOT_SCRIPT_ALLOWLIST
        ):
            issues.append(
                ReleaseIssue(path.relative_to(root), None, "unexpected top-level script")
            )

    for path in sorted(_iter_public_text_files(root)):
        relative_path = path.relative_to(root)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            issues.append(ReleaseIssue(relative_path, None, "text file is not valid UTF-8"))
            continue
        for line_number, line in enumerate(lines, start=1):
            lowered = line.lower()
            if any(fragment in lowered for fragment in FORBIDDEN_FRAGMENTS):
                issues.append(
                    ReleaseIssue(relative_path, line_number, "contains a private path or legacy package name")
                )
            elif WINDOWS_ABSOLUTE_PATH.search(line):
                issues.append(ReleaseIssue(relative_path, line_number, "contains an absolute Windows path"))
    return issues


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="NavOL repository root (default: inferred from this script)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    issues = check_release(args.repository_root)
    for issue in issues:
        location = str(issue.path) if issue.line is None else f"{issue.path}:{issue.line}"
        print(f"ERROR {location}: {issue.message}")
    if issues:
        print(f"release check failed with {len(issues)} issue(s)")
        return 1
    print("release check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
