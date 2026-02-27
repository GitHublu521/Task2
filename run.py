#!/usr/bin/env python3
"""Entry point for Lab 2: Ablation Study on CIFAR-10.

Usage:
    python run.py --task 1        # Run Task 1 only
    python run.py --task 2        # Run Task 2 only
    python run.py --task all      # Run all tasks
"""

import argparse
import sys

from tasks import run_task1, run_task2, run_task3, run_task4, run_task5


TASKS = {
    "1": ("Task 1: Weight Initialization", run_task1),
    "2": ("Task 2: Regularization", run_task2),
    "3": ("Task 3: BatchNorm", run_task3),
    "4": ("Task 4: Robustness (CIFAR-10-C)", run_task4),
    "5": ("Task 5: Hyperparameter Search", run_task5),
}


def run_task_safe(task_id: str):
    """Run a single task with graceful NotImplementedError handling."""
    name, func = TASKS[task_id]
    print(f"\n{'#' * 60}")
    print(f"# {name}")
    print(f"{'#' * 60}\n")
    try:
        func()
    except NotImplementedError as e:
        print(f"\n  SKIPPED: {name} has unimplemented TODOs")
        print(f"  Detail: {e}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Lab 2: Ablation Study on CIFAR-10"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["1", "2", "3", "4", "5", "all"],
        help="Which task to run (1-5, or 'all').",
    )
    args = parser.parse_args()

    if args.task == "all":
        task_ids = ["1", "2", "3", "4", "5"]
    else:
        task_ids = [args.task]

    results = {}
    for tid in task_ids:
        success = run_task_safe(tid)
        results[tid] = "DONE" if success else "SKIPPED"

    # Summary
    print(f"\n{'=' * 60}")
    print("Run Summary")
    print(f"{'=' * 60}")
    for tid, status in results.items():
        name = TASKS[tid][0]
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
