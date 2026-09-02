import importlib.util
import gc
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPOSITORY_ROOT / "scripts" / "data" / "prepare_benchmark.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_benchmark", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def npy_payload(shape=(100, 7)) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.zeros(shape, dtype=np.float64), allow_pickle=False)
    return stream.getvalue()


def write_source_archive(path: Path, *, unsafe_member: str | None = None) -> None:
    root = "historical_benchmark"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if unsafe_member is not None:
            archive.writestr(unsafe_member, b"unsafe")
            return
        archive.writestr(f"{root}/scene_000/scene.glb", b"portable scene")
        archive.writestr(f"{root}/scene_000/navmesh_scene.glb", b"portable navmesh")
        archive.writestr(f"{root}/scene_000/sample_100.npy", npy_payload())
        archive.writestr(f"{root}/scene_000/textures/wall.png", b"texture")
        archive.writestr(f"{root}/scene_000/scene.usd", b"machine-specific USD")
        archive.writestr(f"{root}/scene_000/config.yaml", b"machine-specific config")


class PortableArchiveTests(unittest.TestCase):
    def test_build_includes_sanitized_usd_without_changing_source(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.zip"
            destination = root / "portable.zip"
            write_source_archive(source)
            source_before = source.read_bytes()

            sanitizer_calls = []

            def sanitize_usd(payload, scene_files):
                sanitizer_calls.append((payload, scene_files))
                return b"sanitized portable USD"

            summary = module.build_portable_archive(
                source,
                destination,
                "in_domain",
                usd_sanitizer=sanitize_usd,
            )

            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(summary.scene_count, 1)
            with zipfile.ZipFile(destination) as archive:
                names = set(archive.namelist())
                self.assertIn("in_domain/scene_000/scene.glb", names)
                self.assertIn("in_domain/scene_000/navmesh_scene.glb", names)
                self.assertIn("in_domain/scene_000/sample_100.npy", names)
                self.assertIn("in_domain/scene_000/textures/wall.png", names)
                self.assertEqual(
                    archive.read("in_domain/scene_000/scene.usd"),
                    b"sanitized portable USD",
                )
                self.assertNotIn("in_domain/scene_000/config.yaml", names)
            self.assertEqual(len(sanitizer_calls), 1)
            self.assertEqual(sanitizer_calls[0][0], b"machine-specific USD")
            self.assertIn("textures/wall.png", sanitizer_calls[0][1])

    def test_build_refuses_to_overwrite_destination(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.zip"
            destination = root / "portable.zip"
            write_source_archive(source)
            destination.write_bytes(b"keep me")

            with self.assertRaises(FileExistsError):
                module.build_portable_archive(source, destination, "in_domain")

            self.assertEqual(destination.read_bytes(), b"keep me")

    def test_build_rejects_path_traversal(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "unsafe.zip"
            destination = root / "portable.zip"
            write_source_archive(source, unsafe_member="../escape.txt")

            with self.assertRaisesRegex(ValueError, "unsafe ZIP member"):
                module.build_portable_archive(source, destination, "out_domain")

            self.assertFalse(destination.exists())

    def test_validate_rejects_wrong_start_goal_shape(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "invalid.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("in_domain/scene_000/scene.glb", b"scene")
                archive.writestr("in_domain/scene_000/navmesh_scene.glb", b"navmesh")
                archive.writestr("in_domain/scene_000/scene.usd", b"portable USD")
                archive.writestr("in_domain/scene_000/sample_100.npy", npy_payload((99, 7)))

            with self.assertRaisesRegex(ValueError, r"expected shape \(100, 7\)"):
                module.validate_portable_archive(archive_path, "in_domain")

    def test_absolute_texture_path_maps_to_existing_scene_texture(self):
        module = load_module()
        path = (
            "/cluster/project/NavOL/benchmark/scene_000/"
            "textures/materials/wall.png"
        )

        result = module.portable_texture_asset_path(
            path,
            {"scene.glb", "textures/materials/wall.png"},
        )

        self.assertEqual(result, "textures/materials/wall.png")

    def test_absolute_texture_path_rejects_missing_texture(self):
        module = load_module()

        with self.assertRaisesRegex(ValueError, "not present in the scene archive"):
            module.portable_texture_asset_path(
                "/cluster/project/scene_000/textures/missing.png",
                {"scene.glb"},
            )

    def test_real_openusd_sanitizer_releases_its_temporary_layer(self):
        try:
            from pxr import Sdf
        except ImportError:
            self.skipTest("OpenUSD is not installed in this Python environment")

        module = load_module()
        usda = '''#usda 1.0
def Shader "Texture"
{
    asset inputs:texture = @C:/cluster/NavOL/scene_000/textures/wall.png@
}
'''
        source_layer = Sdf.Layer.CreateAnonymous()
        self.assertTrue(source_layer.ImportFromString(usda))
        with tempfile.TemporaryDirectory() as source_directory:
            binary_path = Path(source_directory) / "fixture.usdc"
            self.assertTrue(source_layer.Export(str(binary_path), args={"format": "usdc"}))
            payload = binary_path.read_bytes()
        del source_layer

        sanitized = module.sanitize_usd_payload(
            payload,
            frozenset({"scene.glb", "textures/wall.png"}),
        )
        self.assertNotIn(b"C:/cluster/", sanitized)

        with tempfile.TemporaryDirectory() as temporary_directory:
            usd_path = Path(temporary_directory) / "scene.usd"
            usd_path.write_bytes(sanitized)
            layer = Sdf.Layer.FindOrOpen(str(usd_path))
            self.assertIsNotNone(layer)
            self.assertIn("@textures/wall.png@", layer.ExportToString())
            del layer
            gc.collect()

    def test_plan_conversion_uses_blender_then_isaac_without_running_them(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scene = root / "scene_000"
            scene.mkdir()
            (scene / "scene.glb").write_bytes(b"scene")
            (scene / "navmesh_scene.glb").write_bytes(b"navmesh")
            np.save(scene / "sample_100.npy", np.zeros((100, 7)), allow_pickle=False)
            converter = REPOSITORY_ROOT / "scripts" / "preprocess" / "convert_mesh2usd.py"

            blender_converter = (
                REPOSITORY_ROOT / "scripts" / "preprocess" / "3d_front" / "blender_usd.py"
            )
            commands = module.plan_usd_conversion(
                root,
                converter,
                blender_converter,
                "python",
                "blender",
            )

            self.assertEqual(len(commands), 2)
            self.assertEqual(commands[0].scene_id, "scene_000")
            self.assertEqual(commands[0].stage, "blender")
            self.assertEqual(
                commands[0].argv,
                (
                    "blender",
                    "--background",
                    "--python",
                    str(blender_converter.resolve()),
                    "--",
                    "--input",
                    str((scene / "scene.glb").resolve()),
                    "--output",
                    str((scene / "tmp" / "scene.usd").resolve()),
                ),
            )
            self.assertEqual(commands[1].scene_id, "scene_000")
            self.assertEqual(commands[1].stage, "isaac")
            self.assertEqual(
                commands[1].argv,
                (
                    "python",
                    str(converter.resolve()),
                    str((scene / "tmp" / "scene.usd").resolve()),
                    str((scene / "scene.usd").resolve()),
                    "--collision-approximation",
                    "meshSimplification",
                    "--headless",
                ),
            )

    def test_plan_conversion_accepts_one_scene_directory(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            scene = Path(temporary_directory) / "scene_000"
            scene.mkdir()
            (scene / "scene.glb").write_bytes(b"scene")
            (scene / "navmesh_scene.glb").write_bytes(b"navmesh")
            np.save(scene / "sample_100.npy", np.zeros((100, 7)), allow_pickle=False)

            commands = module.plan_usd_conversion(
                scene,
                REPOSITORY_ROOT / "scripts" / "preprocess" / "convert_mesh2usd.py",
                REPOSITORY_ROOT
                / "scripts"
                / "preprocess"
                / "3d_front"
                / "blender_usd.py",
                "python",
                "blender",
            )

            self.assertEqual([command.scene_id for command in commands], ["scene_000", "scene_000"])


if __name__ == "__main__":
    unittest.main()
