import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "source" / "navol"))

import navol.paths as navol_paths
from navol.paths import asset_root, dataset_path, model_path, policy_checkpoint_path, robot_path


class AssetPathTests(unittest.TestCase):
    def test_canonical_hugging_face_asset_contract(self):
        self.assertEqual(navol_paths.DEFAULT_BASE_CHECKPOINT, "navdp-cross-modal.ckpt")
        self.assertEqual(
            navol_paths.DEFAULT_POLICY_CHECKPOINT,
            "checkpoints/navol-mpc-iter1000.pt",
        )
        self.assertEqual(navol_paths.DEFAULT_ROBOT_ASSET, "dingo.usd")
        self.assertEqual(navol_paths.DEFAULT_TRAIN_DATASET, "train/3d_front_scene_50")

    def test_explicit_root_has_highest_precedence(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as explicit_dir:
            with patch.dict(os.environ, {"NAVOL_ASSET_ROOT": str(REPOSITORY_ROOT / "ignored")}, clear=False):
                self.assertEqual(asset_root(explicit_dir), Path(explicit_dir).resolve())

    def test_environment_root_overrides_repository_default(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as environment_dir:
            with patch.dict(os.environ, {"NAVOL_ASSET_ROOT": environment_dir}, clear=False):
                self.assertEqual(asset_root(), Path(environment_dir).resolve())

    def test_repository_assets_are_the_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(asset_root(), REPOSITORY_ROOT / "assets")

    def test_missing_model_reports_name_and_override(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as assets_dir:
            with self.assertRaisesRegex(FileNotFoundError, "navdp-cross-modal.ckpt.*NAVOL_ASSET_ROOT"):
                model_path("navdp-cross-modal.ckpt", explicit_root=assets_dir)

    def test_existing_model_is_resolved_under_models(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as assets_dir:
            expected = Path(assets_dir) / "models" / "checkpoints" / "navol-mpc-iter1000.pt"
            expected.parent.mkdir(parents=True)
            expected.touch()
            self.assertEqual(
                model_path("checkpoints/navol-mpc-iter1000.pt", explicit_root=assets_dir),
                expected.resolve(),
            )

    def test_existing_robot_is_resolved_under_robots(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as assets_dir:
            expected = Path(assets_dir) / "robots" / "dingo.usd"
            expected.parent.mkdir(parents=True)
            expected.touch()
            self.assertEqual(robot_path("dingo.usd", explicit_root=assets_dir), expected.resolve())

    def test_training_dataset_is_resolved_under_datasets(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as assets_dir:
            expected = Path(assets_dir) / "datasets" / "train" / "3d_front_scene_50"
            expected.mkdir(parents=True)
            self.assertEqual(
                dataset_path("train/3d_front_scene_50", explicit_root=assets_dir),
                expected.resolve(),
            )

    def test_policy_checkpoint_defaults_to_mpc_iteration_1000(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as assets_dir:
            expected = Path(assets_dir) / "models" / "checkpoints" / "navol-mpc-iter1000.pt"
            expected.parent.mkdir(parents=True)
            expected.touch()
            self.assertEqual(
                policy_checkpoint_path(explicit_root=assets_dir),
                expected.resolve(),
            )

    def test_explicit_policy_checkpoint_has_highest_precedence(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as assets_dir:
            expected = Path(assets_dir) / "downloaded" / "model_2000_navdp.pt"
            expected.parent.mkdir()
            expected.touch()
            self.assertEqual(
                policy_checkpoint_path(expected, explicit_root=REPOSITORY_ROOT / "ignored"),
                expected.resolve(),
            )

    def test_missing_explicit_policy_checkpoint_reports_cli_flag(self):
        missing = REPOSITORY_ROOT / "missing-model.pt"
        with self.assertRaisesRegex(FileNotFoundError, "missing-model.pt.*--checkpoint_path"):
            policy_checkpoint_path(missing)


if __name__ == "__main__":
    unittest.main()
