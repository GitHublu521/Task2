"""Results gate: verify result files, quality, and summary for each task.

Scoring: 20 points per task, 100 total.
  - Task 1: Weight Initialization       (20 pts)
  - Task 2: Regularization              (20 pts)
  - Task 3: BatchNorm                   (20 pts)
  - Task 4: Robustness                  (20 pts)
  - Task 5: Hyperparameter Search       (20 pts)
"""

import json
import os
import re
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SUMMARY = os.path.join(ROOT, "report.md")

# Required result files per task
TASK1_FILES = [
    "task1_summary.csv",
    "task1_activation_stds.json",
    "task1_activation_stds.png",
    "task1_loss_curves.png",
    "task1_acc_curves.png",
    "task1_val_acc_comparison.png",
]

TASK2_FILES = [
    "task2_summary.csv",
    "task2_loss_curves.png",
    "task2_acc_curves.png",
    "task2_val_acc_comparison.png",
]

TASK3_FILES = [
    "task3_summary.csv",
    "task3_loss_curves.png",
    "task3_acc_curves.png",
    "task3_val_acc_comparison.png",
]

TASK4_FILES = [
    "task4_summary.csv",
    "task4_cifar10c_results.json",
    "task4_heatmap.png",
    "task4_clean_acc.png",
]

TASK5_FILES = [
    "task5_trials.csv",
    "task5_best_params.json",
    "task5_best_model_curves.png",
]


def _check_files(test_case, file_list, task_name):
    """Assert all files in file_list exist under results/."""
    missing = [f for f in file_list
               if not os.path.exists(os.path.join(RESULTS, f))]
    if missing:
        test_case.fail(f"{task_name}: missing files {missing}")


def _check_summary_section(test_case, section_title):
    """Assert the report.md section for a task has no remaining TODO placeholders."""
    if not os.path.exists(SUMMARY):
        test_case.fail("report.md not found")

    with open(SUMMARY) as f:
        content = f.read()

    # Extract the section between this task header and the next ## header (or EOF)
    pattern = rf"(## {re.escape(section_title)}.*?)(?=\n## |\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        test_case.fail(f"Section '{section_title}' not found in report.md")

    section = match.group(1)
    todo_count = section.count("<!-- TODO")
    if todo_count > 0:
        test_case.fail(
            f"report.md section '{section_title}' still has "
            f"{todo_count} unfilled TODO placeholder(s)"
        )


class TestTask1(unittest.TestCase):
    """Task 1: Weight Initialization — 20 points."""

    def test_task1(self):
        # Check result files
        _check_files(self, TASK1_FILES, "Task 1")

        # Check activation stds JSON has all 5 init methods
        stds_path = os.path.join(RESULTS, "task1_activation_stds.json")
        with open(stds_path) as f:
            data = json.load(f)
        expected = {"default", "xavier_uniform", "xavier_normal",
                    "kaiming_uniform", "kaiming_normal"}
        missing = expected - set(data.keys())
        if missing:
            self.fail(f"Task 1: missing init methods in activation stds: {missing}")

        # Check report section filled
        _check_summary_section(self, "3. Task 1: Weight Initialization")


class TestTask2(unittest.TestCase):
    """Task 2: Regularization — 20 points."""

    def test_task2(self):
        _check_files(self, TASK2_FILES, "Task 2")
        _check_summary_section(self, "4. Task 2: Regularization")


class TestTask3(unittest.TestCase):
    """Task 3: BatchNorm — 20 points."""

    def test_task3(self):
        _check_files(self, TASK3_FILES, "Task 3")
        _check_summary_section(self, "5. Task 3: BatchNorm")


class TestTask4(unittest.TestCase):
    """Task 4: Robustness — 20 points."""

    def test_task4(self):
        _check_files(self, TASK4_FILES, "Task 4")
        _check_summary_section(self, "6. Task 4: Robustness to Distribution Shift")


class TestTask5(unittest.TestCase):
    """Task 5: Hyperparameter Search — 20 points."""

    def test_task5(self):
        _check_files(self, TASK5_FILES, "Task 5")

        # Check sufficient Optuna trials
        import pandas as pd
        trials_path = os.path.join(RESULTS, "task5_trials.csv")
        df = pd.read_csv(trials_path)
        self.assertGreaterEqual(
            len(df), 10,
            f"Task 5: only {len(df)} Optuna trials. Need at least 10."
        )

        _check_summary_section(self, "7. Task 5: Hyperparameter Search")


if __name__ == "__main__":
    unittest.main()
