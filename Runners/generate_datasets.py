from Classes.Generators.HanoiGame import generate_hanoi_dataset
from Classes.Generators.FibonacciGame import generate_fibonacci_dataset
from Classes.Utils.data_utils import save_jsonl

train_data, test_data = generate_hanoi_dataset(
    num_examples=100,
    min_disks=3,
    max_disks=8,
    min_future_steps=1,
    max_future_steps=8,
    num_shots=0,             
    num_fewshot_examples=3,
    test_fraction=0.1,
    seed=42,
)

save_jsonl(train_data, "../Data/Train/towers_hanoi_train")
save_jsonl(test_data, "../Data/Test/towers_hanoi_test")


fib_train, fib_test = generate_fibonacci_dataset(
    num_examples=100,
    min_terms=10,
    max_terms=40,
    min_future_steps=1,
    max_future_steps=8,
    num_shots=0,            
    num_fewshot_examples=3,
    test_fraction=0.1,
    seed=123,
)

save_jsonl(fib_train, "../Data/Train/fibonacci_train")
save_jsonl(fib_test, "../Data/Test/fibonacci_test")