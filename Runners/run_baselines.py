from Classes.Models.qwen5b import Qwen5B
from Classes.Utils.data_utils import load_jsonl
from datasets import Dataset

train_dir = './Data/Train/'
test_dir = './Data/Test/'

print('loading training data')
hanoi_train_data = load_jsonl(train_dir + 'towers_hanoi_train')

print('loading test data')
hanoi_test_data = load_jsonl(test_dir + 'towers_hanoi_test')


hanoi_train_data = Dataset.from_list(hanoi_train_data)
hanoi_test_data = Dataset.from_list(hanoi_test_data)

print('creating Qwen5B')
model = Qwen5B()


print('creating train and val folds')
train_folds, val_folds = model.create_folds(hanoi_train_data, n_folds=5, seed=42)


print('evaluating base Qwen5B')
metrics_base = model.evaluate_base_model(hanoi_test_data, "hanoi_test_base")

print("Base model metrics:", metrics_base)


print("HF Token-Level Accuracy:", metrics_base.get("eval_accuracy"))
print("Sequence-Level Accuracy:", metrics_base.get("accuracy"))

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