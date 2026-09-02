import sys
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "source" / "navol"))

from navol.training import CANONICAL_TRAINING, compute_critic_targets, select_policy_action


class CanonicalTrainingTests(unittest.TestCase):
    def test_paper_training_defaults_are_centralized(self):
        self.assertEqual(CANONICAL_TRAINING.rollout_steps, 128)
        self.assertEqual(CANONICAL_TRAINING.iterations, 1000)
        self.assertEqual(CANONICAL_TRAINING.learning_epochs, 10)
        self.assertEqual(CANONICAL_TRAINING.mini_batches, 16)
        self.assertEqual(CANONICAL_TRAINING.environments_per_process, 32)
        self.assertEqual(CANONICAL_TRAINING.distributed_processes, 8)
        self.assertEqual(CANONICAL_TRAINING.batch_size, 2048)
        self.assertEqual(CANONICAL_TRAINING.camera_height_range, (0.25, 1.25))
        self.assertEqual(CANONICAL_TRAINING.camera_pitch_range, (-30.0, 0.0))
        self.assertEqual(CANONICAL_TRAINING.policy_probability, 0.8)
        self.assertTrue(CANONICAL_TRAINING.use_mpc)
        self.assertEqual(CANONICAL_TRAINING.navmesh_radius, 0.25)
        self.assertEqual(CANONICAL_TRAINING.local_search_radius, 0.1)
        self.assertEqual(CANONICAL_TRAINING.critic_collision_reduction, "mean")

    def test_isaac_configs_reference_the_canonical_defaults(self):
        config_paths = (
            REPOSITORY_ROOT
            / "source/navol/navol/tasks/manager_based/navdp/configs/dingo/agents/rsl_rl_distillation_cfg.py",
            REPOSITORY_ROOT
            / "source/navol/navol/tasks/manager_based/navdp/configs/dingo/navdp_pointgoal_cfg.py",
            REPOSITORY_ROOT
            / "source/navol/navol/tasks/manager_based/navdp/mdp/actions_cfg.py",
            REPOSITORY_ROOT
            / "source/navol/navol/tasks/manager_based/navdp/mdp/events.py",
        )
        for path in config_paths:
            source = path.read_text(encoding="utf-8")
            self.assertIn("CANONICAL_TRAINING", source, path)

    def test_low_level_launcher_preserves_canonical_camera_default(self):
        source = (REPOSITORY_ROOT / "scripts/rsl_rl/train_navdp.py").read_text(encoding="utf-8")
        self.assertIn(
            'parser.add_argument("--rand_camera", action=argparse.BooleanOptionalAction, default=None',
            source,
        )
        self.assertIn("if args_cli.rand_camera is not None:", source)


class ActionMixingTests(unittest.TestCase):
    def test_random_mode_uses_policy_with_configured_probability(self):
        self.assertTrue(select_policy_action("rand", 0.8, 0.7999))
        self.assertFalse(select_policy_action("rand", 0.8, 0.8))

    def test_explicit_modes_are_deterministic(self):
        self.assertTrue(select_policy_action("policy", 0.8, 1.0))
        self.assertFalse(select_policy_action("command", 0.8, 0.0))

    def test_action_mixing_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            select_policy_action("invalid", 0.8, 0.5)
        with self.assertRaises(ValueError):
            select_policy_action("rand", 1.1, 0.5)
        with self.assertRaises(ValueError):
            select_policy_action("rand", 0.8, -0.1)


class CriticTargetTests(unittest.TestCase):
    def test_mean_reduction_preserves_existing_checkpoint_semantics(self):
        distances = np.array([[0.1, 0.6, 0.2]], dtype=np.float32)
        actual = compute_critic_targets(distances, collision_reduction="mean")
        np.testing.assert_allclose(actual, np.array([-2.0 / 3.0 + 0.01]), rtol=1e-6)

    def test_sum_reduction_matches_the_paper_equation(self):
        distances = np.array([[0.1, 0.6, 0.2]], dtype=np.float32)
        actual = compute_critic_targets(distances, collision_reduction="sum")
        np.testing.assert_allclose(actual, np.array([-1.99]), rtol=1e-6)

    def test_critic_target_rejects_unknown_reduction(self):
        with self.assertRaises(ValueError):
            compute_critic_targets(np.array([[0.1, 0.2]]), collision_reduction="median")


if __name__ == "__main__":
    unittest.main()
