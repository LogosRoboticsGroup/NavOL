import sys
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "source" / "navol"))

from navol.paths import dataset_path


class BenchmarkManifestTests(unittest.TestCase):
    def test_every_available_scene_has_100_start_goal_pairs(self):
        domain_roots = [
            dataset_path("benchmarks/in_domain", must_exist=False),
            dataset_path("benchmarks/out_domain", must_exist=False),
        ]
        scenes_by_root = {
            root: sorted(path for path in root.glob("scene_*") if path.is_dir())
            for root in domain_roots
            if root.is_dir()
        }
        available_roots = {root: scenes for root, scenes in scenes_by_root.items() if scenes}
        if not available_roots:
            self.skipTest(
                "benchmark assets are not installed; set NAVOL_ASSET_ROOT to validate them"
            )

        for domain_root, scene_dirs in available_roots.items():
            with self.subTest(domain=domain_root.name):
                for scene_dir in scene_dirs:
                    pair_path = scene_dir / "sample_100.npy"
                    self.assertTrue(pair_path.is_file(), f"missing benchmark pairs: {pair_path}")
                    pairs = np.load(pair_path, allow_pickle=False)
                    self.assertEqual(
                        pairs.shape,
                        (100, 7),
                        f"expected 100 seven-value start-goal pairs in {pair_path}",
                    )


if __name__ == "__main__":
    unittest.main()
