import importlib.util
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPOSITORY_ROOT / "scripts" / "check_release.py"


def load_module():
    spec = importlib.util.spec_from_file_location("check_release", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_minimal_release(root: Path) -> None:
    required_files = (
        "README.md",
        "docs/README_zh-CN.md",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "assets/README.md",
        "scripts/data/README.md",
        "scripts/train/README.md",
        "scripts/eval/README.md",
        "scripts/data/prepare_benchmark.py",
        "scripts/train/train_navol.py",
        "scripts/eval/evaluate_benchmark.py",
        "source/navol/pyproject.toml",
        "source/navol/README.md",
        "source/navol/setup.py",
        "source/navol/config/extension.toml",
        "source/navol/LICENSE",
        "source/navol/THIRD_PARTY_NOTICES.md",
        ".pre-commit-config.yaml",
        ".flake8",
        ".github/workflows/ci.yml",
    )
    for relative_path in required_files:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("NavOL public release\n", encoding="utf-8")


class ReleaseCheckTests(unittest.TestCase):
    def test_valid_release_surface_has_no_issues(self):
        module = load_module()
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            write_minimal_release(root)

            self.assertEqual(module.check_release(root), [])

    def test_missing_required_file_is_reported(self):
        module = load_module()
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            write_minimal_release(root)
            (root / "THIRD_PARTY_NOTICES.md").unlink()

            issues = module.check_release(root)

            self.assertTrue(
                any("THIRD_PARTY_NOTICES.md" in issue.message for issue in issues),
                issues,
            )

    def test_personal_path_in_public_script_is_reported(self):
        module = load_module()
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            write_minimal_release(root)
            script = root / "scripts" / "train" / "cluster.sh"
            script.write_text(
                "python /" + "inspire/ssd/private/train.py\n",
                encoding="utf-8",
            )

            issues = module.check_release(root)

            self.assertTrue(
                any(issue.path == Path("scripts/train/cluster.sh") for issue in issues),
                issues,
            )

    def test_unexpected_root_debug_script_is_reported(self):
        module = load_module()
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            write_minimal_release(root)
            (root / "debug.py").write_text("print('debug')\n", encoding="utf-8")

            issues = module.check_release(root)

            self.assertTrue(
                any(
                    issue.path == Path("debug.py")
                    and "unexpected top-level script" in issue.message
                    for issue in issues
                ),
                issues,
            )

    def test_asset_payloads_and_archives_are_not_scanned(self):
        module = load_module()
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            write_minimal_release(root)
            payload = root / "assets" / "models" / "checkpoint.pt"
            payload.parent.mkdir(parents=True)
            private_path = "/" + "inspire/ssd/private"
            payload.write_text(private_path, encoding="utf-8")
            archive = root / "benchmark.zip"
            archive.write_text(private_path, encoding="utf-8")

            self.assertEqual(module.check_release(root), [])


if __name__ == "__main__":
    unittest.main()
