# baseline_gptoss120b_single.py

import os
from datasets import Dataset
from Classes.Utils.data_utils import load_jsonl
from Classes.Models.gptoss120b import GPTOSS120B

test_dir = "./Data/Test/"
TEST_FILE = "nqueens_test"   # <--- single task


def load_dataset(path):
    return Dataset.from_list(load_jsonl(path))


print("Loading N-Queens test dataset...")
test_path = os.path.join(test_dir, TEST_FILE)
test_ds = load_dataset(test_path)

print("\nCreating GPTOSS120B (OpenRouter API model)")
model = GPTOSS120B()   # <-- uses your wrapper + env var OPENROUTER_API_KEY


def run_single_baseline():
    print("\n=== Baseline evaluation (OpenRouter) for N-Queens ===")

    metrics = model.evaluate_api_model(
        test_ds=test_ds,
        save_name="nqueens_gptoss120b_single",
    )

    print("Baseline metrics:", metrics)
    print("Sequence-Level Accuracy:", metrics.get("accuracy"))
    print("HF Eval Loss (API, None):", metrics.get("eval_loss"))
    return metrics


results = run_single_baseline()

print("\nFinal Result:")
print(results)
