from contextlib import contextmanager
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_ROOT = REPOSITORY_ROOT / "source" / "navol"


@contextmanager
def change_directory(path):
    previous_directory = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_directory)


class PackageMetadataTests(unittest.TestCase):
    def test_setup_exposes_navol_distribution(self):
        completed = subprocess.run(
            [sys.executable, "setup.py", "--name"],
            cwd=EXTENSION_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip().splitlines()[-1], "navol")

    def test_setup_exposes_bsd_3_clause_license(self):
        completed = subprocess.run(
            [sys.executable, "setup.py", "--license"],
            cwd=EXTENSION_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip().splitlines()[-1], "BSD-3-Clause")

    def test_extension_identifies_navol(self):
        with (EXTENSION_ROOT / "config" / "extension.toml").open("rb") as stream:
            extension = tomllib.load(stream)
        self.assertEqual(extension["package"]["title"], "NavOL")
        self.assertEqual(extension["python"]["module"][0]["name"], "navol")

    def test_setup_includes_every_navol_subpackage(self):
        captured = {}

        def capture_setup(**kwargs):
            captured.update(kwargs)

        with change_directory(EXTENSION_ROOT), patch(
            "setuptools.setup", side_effect=capture_setup
        ):
            runpy.run_path(str(EXTENSION_ROOT / "setup.py"), run_name="__main__")

        packages = set(captured["packages"])
        required = {
            "navol",
            "navol.assets",
            "navol.evaluation",
            "navol.tasks",
            "navol.tasks.manager_based.navdp.mdp.utils",
            "navol.terrains",
            "navol.wrapper",
        }
        self.assertTrue(required.issubset(packages), required - packages)

    def test_setup_declares_non_isaac_runtime_dependencies(self):
        captured = {}

        def capture_setup(**kwargs):
            captured.update(kwargs)

        with change_directory(EXTENSION_ROOT), patch(
            "setuptools.setup", side_effect=capture_setup
        ):
            runpy.run_path(str(EXTENSION_ROOT / "setup.py"), run_name="__main__")

        dependency_names = {
            requirement.split(";")[0].split("[")[0].split("=")[0].split(">")[0].strip().lower()
            for requirement in captured["install_requires"]
        }
        required = {
            "imageio",
            "matplotlib",
            "numpy",
            "opencv-python",
            "open3d",
            "psutil",
            "rpyc",
            "scipy",
            "tqdm",
            "trimesh",
        }
        self.assertTrue(required.issubset(dependency_names), required - dependency_names)

    def test_setup_declares_public_project_and_supported_python_versions(self):
        captured = {}

        def capture_setup(**kwargs):
            captured.update(kwargs)

        with change_directory(EXTENSION_ROOT), patch(
            "setuptools.setup", side_effect=capture_setup
        ):
            runpy.run_path(str(EXTENSION_ROOT / "setup.py"), run_name="__main__")

        self.assertEqual(captured["url"], "https://github.com/WAboutMe/NavOL")
        self.assertEqual(captured["python_requires"], ">=3.10")
        self.assertIn("Programming Language :: Python :: 3.10", captured["classifiers"])
        self.assertIn("Programming Language :: Python :: 3.11", captured["classifiers"])

    def test_wheel_contains_navol_modules_and_bsd_license(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            package_root = temporary_root / "package"
            distribution_root = temporary_root / "dist"
            build_root = temporary_root / "build"
            shutil.copytree(EXTENSION_ROOT, package_root)

            subprocess.run(
                [
                    sys.executable,
                    "setup.py",
                    "bdist_wheel",
                    "--dist-dir",
                    str(distribution_root),
                    "--bdist-dir",
                    str(build_root),
                ],
                cwd=package_root,
                check=True,
                capture_output=True,
                text=True,
            )

            wheels = list(distribution_root.glob("navol-*.whl"))
            self.assertEqual(len(wheels), 1, wheels)
            with zipfile.ZipFile(wheels[0]) as archive:
                names = set(archive.namelist())
                required_modules = {
                    "navol/__init__.py",
                    "navol/paths.py",
                    "navol/training.py",
                    "navol/evaluation/metrics.py",
                }
                self.assertFalse(required_modules - names, required_modules - names)
                license_members = [
                    name
                    for name in names
                    if name.endswith(".dist-info/licenses/LICENSE")
                    or name.endswith(".dist-info/LICENSE")
                ]
                self.assertEqual(len(license_members), 1, license_members)
                license_text = archive.read(license_members[0]).decode("utf-8")
                self.assertIn("BSD 3-Clause License", license_text)
                notice_members = [
                    name
                    for name in names
                    if name.endswith(".dist-info/licenses/THIRD_PARTY_NOTICES.md")
                    or name.endswith(".dist-info/THIRD_PARTY_NOTICES.md")
                ]
                self.assertEqual(len(notice_members), 1, notice_members)

            forbidden_payload_parts = (".ckpt", ".pt", ".usd", ".zip")
            self.assertFalse(
                [name for name in names if name.lower().endswith(forbidden_payload_parts)]
            )


if __name__ == "__main__":
    unittest.main()
