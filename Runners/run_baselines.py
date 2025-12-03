from Classes.Models.qwen5b import Qwen5B
from Classes.Utils.data_utils import load_jsonl
from datasets import Dataset
import os

train_dir = './Data/Train/'
test_dir  = './Data/Test/'

# game list
TASKS = {
    "hanoi": {
        "train": "towers_hanoi_train",
        "test":  "towers_hanoi_test",
    },
    "fibonacci": {
        "train": "fibonacci_train",
        "test":  "fibonacci_test",
    },
    "sliding": {
        "train": "sliding_puzzle_train",
        "test":  "sliding_puzzle_test",
    }
}


def load_dataset(path):
    return Dataset.from_list(load_jsonl(path))


train_sets = {}
test_sets  = {}

print("Loading datasets...")

# load individual datasets
for task, files in TASKS.items():
    train_path = os.path.join(train_dir, files["train"])
    test_path  = os.path.join(test_dir,  files["test"])

    print(f"  • Loading {task} train/test")
    train_sets[task] = load_dataset(train_path)
    test_sets[task]  = load_dataset(test_path)

# intialize model (qwen 5b in this case)
print("\nCreating Qwen5B")
model = Qwen5B()

# the task loop itself (create folds, evaluate base model, get metrics)
def process_task(task_name, train_ds, test_ds):
    print(f"\n=== Processing task: {task_name} ===")

    print("Creating train/val folds")
    train_folds, val_folds = model.create_folds(
        train_ds, n_folds=5, seed=42
    )

    print("Evaluating base model")
    metrics = model.evaluate_base_model(
        test_ds,
        f"{task_name}_test_base"
    )

    print("Base model metrics:", metrics)
    print("HF Token-Level Accuracy:", metrics.get("eval_accuracy"))
    print("Sequence-Level Accuracy:", metrics.get("accuracy"))

    return metrics

results = {}

# run task loop
for task in TASKS:
    results[task] = process_task(task, train_sets[task], test_sets[task])



'''
# --------------------------
# Train or load the tuned model
# --------------------------
print('training / loading tuned model')
tuned_model, tuned_tokenizer = model.load_or_train_tuned_model(
    train_folds, val_folds, n_trials=50
)

# --------------------------
# Evaluate tuned model
# --------------------------
print('evaluating tuned Qwen5B')
metrics_tuned = model.evaluate_tuned_model(
    tuned_model, tuned_tokenizer, hanoi_test_data, "hanoi_test_tuned"
)

print("Tuned model metrics:", metrics_tuned)
print("HF Token-Level Accuracy:", metrics_tuned.get("eval_accuracy"))
print("Sequence-Level Accuracy:", metrics_tuned.get("accuracy"))
'''
