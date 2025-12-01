from Classes.Models.qwen5b import Qwen5B
from Classes.Utils.data_utils import load_jsonl
from datasets import Dataset

train_dir = './Data/Train/'
test_dir = './Data/Test/'

print('loading training data')
hanoi_train_data = load_jsonl(train_dir + 'towers_hanoi_train')
fib_train_data   = load_jsonl(train_dir + 'fibonacci_train')
sliding_train_data = load_jsonl(train_dir + 'sliding_puzzle_train')

print('loading test data')
hanoi_test_data = load_jsonl(test_dir + 'towers_hanoi_test')
fib_test_data   = load_jsonl(test_dir + 'fibonacci_test')
sliding_test_data = load_jsonl(test_dir + 'sliding_puzzle_test')

hanoi_train_data = Dataset.from_list(hanoi_train_data)
hanoi_test_data = Dataset.from_list(hanoi_test_data)

fib_train_ds = Dataset.from_list(fib_train_data)
fib_test_ds  = Dataset.from_list(fib_test_data)
sliding_train_ds = Dataset.from_list(sliding_train_data)
sliding_test_ds  = Dataset.from_list(sliding_test_data)

print('creating Qwen5B')
model = Qwen5B()


print('creating train and val folds')
train_folds, val_folds = model.create_folds(hanoi_train_data, n_folds=5, seed=42)


print('evaluating base Qwen5B')
metrics_base = model.evaluate_base_model(hanoi_test_data, "hanoi_test_base")

print("Base model metrics:", metrics_base)


print("HF Token-Level Accuracy:", metrics_base.get("eval_accuracy"))
print("Sequence-Level Accuracy:", metrics_base.get("accuracy"))


print('creating train and val folds for Fibonacci')
fib_train_folds, fib_val_folds = model.create_folds(
    fib_train_ds, n_folds=5, seed=42
)

print('evaluating base Qwen5B on Fibonacci')
metrics_base_fib = model.evaluate_base_model(fib_test_ds, "fibonacci_test_base")

print("Base model metrics (Fibonacci):", metrics_base_fib)
print("HF Token-Level Accuracy (Fibonacci):", metrics_base_fib.get("eval_accuracy"))
print("Sequence-Level Accuracy (Fibonacci):", metrics_base_fib.get("accuracy"))

print('creating train and val folds for Sliding Puzzle')
sliding_train_folds, sliding_val_folds = model.create_folds(
    sliding_train_ds, n_folds=5, seed=42
)

print('evaluating base Qwen5B on Sliding Puzzle')
metrics_base_sliding = model.evaluate_base_model(sliding_test_ds, "sliding_test_base")

print("Base model metrics (Sliding):", metrics_base_sliding)
print("HF Token-Level Accuracy (Sliding):", metrics_base_sliding.get("eval_accuracy"))
print("Sequence-Level Accuracy (Sliding):", metrics_base_sliding.get("accuracy"))



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
