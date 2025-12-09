import os
from dataclasses import dataclass

from datasets import Dataset
from Classes.Utils.data_utils import load_jsonl

# your OpenRouter wrapper
from Classes.Models.qwen72b import Qwen72B


test_dir = "./Data/Test/"

# ---- Baseline tasks: only test files ----
TASKS = {
    "hanoi":          "towers_hanoi_test",
    "fibonacci":      "fibonacci_test",
    "sliding_puzzle": "sliding_puzzle_test",
    "nqueens":        "nqueens_test",
}


def load_dataset(path):
    return Dataset.from_list(load_jsonl(path))


test_sets = {}

print("Loading test datasets...")

for task_name, test_file in TASKS.items():
    test_path = os.path.join(test_dir, test_file)
    print(f"  • Loading {task_name} test from {test_path}")
    test_sets[task_name] = load_dataset(test_path)

print("\nCreating Qwen72B (OpenRouter API model)")
model = Qwen72B()


def process_task_baseline(task_name, test_ds):
    print(f"\n=== Baseline evaluation (OpenRouter) for task: {task_name} ===")

    # just inference via OpenRouter, no training/tuning
    metrics_base = model.evaluate_api_model(
        test_ds=test_ds,
        save_name=f"{task_name}_Qwen72B",
    )

    print("Baseline metrics:", metrics_base)
    # ApiModel summary only has accuracy + total_examples
    print("Sequence-Level Accuracy:", metrics_base.get("accuracy"))
    # eval_loss will be None for API models; printed here for consistency
    print("HF Eval Loss (API, None):", metrics_base.get("eval_loss"))

    return metrics_base


results = {}

# run baseline loop
for task in TASKS:
    results[task] = process_task_baseline(task, test_sets[task])

print("\nAll baseline results (Qwen72B via OpenRouter):")
for task, metrics in results.items():
    print(f"  {task}: {metrics}")
