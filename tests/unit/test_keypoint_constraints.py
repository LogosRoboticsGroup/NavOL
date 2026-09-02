import ast
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NAVMESH_UTILS = (
    REPOSITORY_ROOT
    / "source/navol/navol/tasks/manager_based/navdp/mdp/utils/navmesh_utils.py"
)
GENPOINT = REPOSITORY_ROOT / "scripts/preprocess/3d_front/genpoint.py"


def function_defaults(path: Path, function_name: str) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    names = [argument.arg for argument in function.args.args]
    values = [ast.literal_eval(default) for default in function.args.defaults]
    return dict(zip(names[-len(values) :], values))


class KeypointConstraintTests(unittest.TestCase):
    def test_goal_generation_is_strict_by_default(self):
        defaults = function_defaults(NAVMESH_UTILS, "navmesh_generate_goal")
        self.assertEqual(defaults["min_keypoints"], 5)
        self.assertFalse(defaults["relax_min_keypoints"])

        source = NAVMESH_UTILS.read_text(encoding="utf-8")
        self.assertIn("if relax_min_keypoints and try_keypoints == try_keypoints_interval:", source)

    def test_offline_pair_generation_defaults_to_five_keypoints(self):
        tree = ast.parse(GENPOINT.read_text(encoding="utf-8"))
        min_keypoint_option = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--min-keypoints"
        )
        default = next(
            ast.literal_eval(keyword.value)
            for keyword in min_keypoint_option.keywords
            if keyword.arg == "default"
        )
        self.assertEqual(default, 5)


if __name__ == "__main__":
    unittest.main()
