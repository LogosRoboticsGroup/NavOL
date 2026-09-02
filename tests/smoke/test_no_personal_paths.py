import ast
import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTROLLED_ROOTS = (
    REPOSITORY_ROOT / "source" / "navol" / "navol",
    REPOSITORY_ROOT / "scripts" / "rsl_rl",
    REPOSITORY_ROOT / "scripts" / "data",
    REPOSITORY_ROOT / "scripts" / "train",
    REPOSITORY_ROOT / "scripts" / "eval",
)
FORBIDDEN_FRAGMENTS = (
    "/" + "inspire" + "/",
    "\\" + "inspire" + "\\",
    "new" + "dog",
)
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class PersonalPathTests(unittest.TestCase):
    def test_maintained_python_has_no_personal_absolute_paths(self):
        violations = []
        for root in CONTROLLED_ROOTS:
            for path in root.rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        value = node.value.lower()
                        if (
                            any(fragment in value for fragment in FORBIDDEN_FRAGMENTS)
                            or WINDOWS_ABSOLUTE_PATH.match(node.value)
                        ):
                            violations.append(f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}")
        self.assertEqual(violations, [], "personal paths found: " + ", ".join(violations))


if __name__ == "__main__":
    unittest.main()
