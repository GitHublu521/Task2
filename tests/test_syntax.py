"""Syntax gate: verify all Python source files compile without errors."""

import os
import py_compile
import unittest


# All Python files that must parse successfully
SOURCE_FILES = [
    "run.py",
    "config.json",  # Not Python, skip — handled separately
    "dataset.py",
    "train.py",
    "utils.py",
    "models/__init__.py",
    "models/mlp.py",
    "models/cnn.py",
    "tasks/__init__.py",
    "tasks/task1_initialization.py",
    "tasks/task2_regularization.py",
    "tasks/task3_batchnorm.py",
    "tasks/task4_robustness.py",
    "tasks/task5_hparam_search.py",
]

# Filter to only .py files
PY_FILES = [f for f in SOURCE_FILES if f.endswith(".py")]

# Project root (one level up from tests/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestSyntax(unittest.TestCase):
    """Verify all source files compile without syntax errors."""

    def test_all_files_compile(self):
        """Every .py file should be valid Python."""
        errors = []
        for rel_path in PY_FILES:
            full_path = os.path.join(ROOT, rel_path)
            if not os.path.exists(full_path):
                errors.append(f"MISSING: {rel_path}")
                continue
            try:
                py_compile.compile(full_path, doraise=True)
            except py_compile.PyCompileError as e:
                errors.append(f"SYNTAX ERROR in {rel_path}: {e}")

        if errors:
            self.fail("\n".join(errors))

    def test_config_json_valid(self):
        """config.json should be valid JSON."""
        import json
        config_path = os.path.join(ROOT, "config.json")
        self.assertTrue(os.path.exists(config_path), "config.json not found")
        with open(config_path) as f:
            try:
                json.load(f)
            except json.JSONDecodeError as e:
                self.fail(f"config.json is not valid JSON: {e}")


if __name__ == "__main__":
    unittest.main()
