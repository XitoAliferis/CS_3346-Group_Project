from Classes.Generators.HanoiGame import generate_hanoi_dataset
from Classes.Generators.FibonacciGame import generate_fibonacci_dataset
from Classes.Generators.SlidingPuzzleGame import generate_sliding_puzzle_dataset
from Classes.Generators.NQueensGame import generate_nqueens_dataset
from Classes.Utils.data_utils import save_jsonl


num_examples = 6000
test_size = 300
size_x = 500
size_y = 1500
size_z = 3000
seed = 42

test_data, train_x, train_y, train_z = generate_hanoi_dataset(
    num_examples=num_examples,
    min_disks=3,
    max_disks=8,
    min_future_steps=1,
    max_future_steps=8,
    num_shots=0,
    num_fewshot_examples=3,
    seed=seed,
    test_size=test_size,
    size_x=size_x,
    size_y=size_y,
    size_z=size_z,
)


save_jsonl(train_x, f"../Data/Train/towers_hanoi_train_{size_x}")
save_jsonl(train_y, f"../Data/Train/towers_hanoi_train_{size_y}")
save_jsonl(train_z, f"../Data/Train/towers_hanoi_train_{size_z}")
save_jsonl(test_data, f"../Data/Test/towers_hanoi_test")


fib_test, fib_train_x, fib_train_y, fib_train_z = generate_fibonacci_dataset(
    num_examples=num_examples,
    min_terms=10,
    max_terms=50,
    min_future_steps=1,
    max_future_steps=8,
    num_shots=0,
    num_fewshot_examples=3,
    test_size=test_size,
    size_x=size_x,
    size_y=size_y,
    size_z=size_z,
    seed=seed,
)

save_jsonl(fib_train_x, f"../Data/Train/fibonacci_train_{size_x}")
save_jsonl(fib_train_y, f"../Data/Train/fibonacci_train_{size_y}")
save_jsonl(fib_train_z, f"../Data/Train/fibonacci_train_{size_z}")
save_jsonl(fib_test, "../Data/Test/fibonacci_test")


sp_test, sp_train_x, sp_train_y, sp_train_z = generate_sliding_puzzle_dataset(
    num_examples=num_examples,
    board_size=3,
    min_scramble_moves=10,
    max_scramble_moves=30,
    min_future_steps=1,
    max_future_steps=6,
    num_shots=0,
    num_fewshot_examples=3,
    test_size=test_size,
    size_x=size_x,
    size_y=size_y,
    size_z=size_z,
    seed=seed,
)


save_jsonl(sp_train_x, f"../Data/Train/sliding_puzzle_train_{size_x}")
save_jsonl(sp_train_y, f"../Data/Train/sliding_puzzle_train_{size_y}")
save_jsonl(sp_train_z, f"../Data/Train/sliding_puzzle_train_{size_z}")
save_jsonl(sp_test, "../Data/Test/sliding_puzzle_test")



nq_train_x, nq_test_x, nq_train_y, nq_test_y, nq_train_z, nq_test_z = generate_nqueens_dataset(
    firstSplit=size_x, 
    secondSplit=size_y, 
    thirdSplit=size_z, 
    amountForTesting=test_size,

    randomSeed=seed,

    minimumN=4, 
    logProgress=True,
)

save_jsonl(nq_train_x, f"../Data/Train/nqueens_train_{size_x}")