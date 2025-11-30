from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import random


@dataclass
class FibonacciGame:
    min_terms: int = 10
    max_terms: int = 40

    def fib_sequence(self, n_terms: int) -> List[int]:
        """Generate the first n_terms of the Fibonacci sequence."""
        if n_terms <= 0:
            return []
        if n_terms == 1:
            return [0]

        seq = [0, 1]
        while len(seq) < n_terms:
            seq.append(seq[-1] + seq[-2])
        return seq[:n_terms]

    def seq_to_text(self, seq: List[int]) -> str:
        """Space-separated Fibonacci numbers on one line."""
        return " ".join(str(x) for x in seq)

    def target_to_text(self, future: List[int]) -> str:
        """
        Target format: EXACTLY one integer per line.
        This keeps the strict format similar to Hanoi's 'X->Y' per line.
        """
        return "\n".join(str(x) for x in future)

    def make_example(
        self,
        full_seq: List[int],
        start_idx: int,
        horizon_k: int,
        num_shots: int = 0,
        fewshot_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Creates a single example:
        - prompt: prefix of Fibonacci seq + instruction
        - target: next horizon_k Fibonacci numbers, one per line
        """

        # prefix includes element at start_idx (sequence up to "current" term)
        prefix = full_seq[: start_idx + 1]
        future = full_seq[start_idx + 1 : start_idx + 1 + horizon_k]

        prefix_text = self.seq_to_text(prefix)
        target_text = self.target_to_text(future)

        # ---- FEWSHOT SECTION ----
        fewshot_blocks: List[str] = []
        if num_shots > 0 and fewshot_examples:
            k = min(num_shots, len(fewshot_examples))
            demos = random.sample(fewshot_examples, k)

            for demo in demos:
                block = (
                    "Example:\n"
                    "Fibonacci sequence prediction.\n"
                    f"Sequence so far:\n{demo['prefix_text']}\n\n"
                    f"Next {demo['future_steps']} Fibonacci numbers:\n"
                    f"{demo['target_text']}\n\n"
                )
                fewshot_blocks.append(block)

        fewshot_section = "".join(fewshot_blocks)

        # ---- PROMPT ----
        prompt = (
            f"{fewshot_section}"
            f"Now solve this new instance.\n"
            f"You are given the beginning of a Fibonacci-like sequence.\n"
            f"Sequence so far (from left to right):\n"
            f"{prefix_text}\n\n"
            f"Provide the next {horizon_k} numbers in the sequence.\n\n"
            f"STRICT OUTPUT FORMAT (MANDATORY):\n"
            f"---------------------------------\n"
            f"You MUST output:\n"
            f"- EXACTLY {horizon_k} lines\n"
            f"- NOTHING except integers\n"
            f"- Each line MUST contain a single integer (no spaces)\n"
            f"- No English words\n"
            f"- No explanations\n"
            f"- No blank lines\n"
            f"- No punctuation\n"
            f"- No numbering\n\n"
            f"Any deviation from the required {horizon_k} lines is INVALID.\n"
        )

        return {
            "total_terms": len(full_seq),
            "start_idx": start_idx,
            "future_steps": horizon_k,
            "prompt": prompt,
            "target": target_text,
        }


def generate_fibonacci_dataset(
    num_examples: int,
    min_terms: int = 10,
    max_terms: int = 40,
    min_future_steps: int = 1,
    max_future_steps: int = 8,
    num_shots: int = 0,
    num_fewshot_examples: int = 3,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Similar design as generate_hanoi_dataset, but for Fibonacci:
    - Each row: prefix of sequence, ask for next K terms
    - Label: next K terms, one per line
    - Uniqueness defined by (total_terms, start_idx, horizon_k)
    """
    rng = random.Random(seed)
    game = FibonacciGame(min_terms=min_terms, max_terms=max_terms)

    examples: List[Dict[str, Any]] = []
    seen_keys = set()
    demo_keys = set()

    # cache: total_terms -> fib sequence
    cache: Dict[int, List[int]] = {}

    # -------- FEWSHOT POOL --------
    fewshot_examples: List[Dict[str, Any]] = []

    if num_shots > 0:
        attempts = 0
        max_attempts = num_fewshot_examples * 50

        while len(fewshot_examples) < num_fewshot_examples and attempts < max_attempts:
            attempts += 1

            total_terms = rng.randint(min_terms, max_terms)
            if total_terms not in cache:
                cache[total_terms] = game.fib_sequence(total_terms)
            full_seq = cache[total_terms]

            max_k_allowed = min(max_future_steps, total_terms - 2)
            if max_k_allowed < min_future_steps:
                continue

            horizon_k = rng.randint(min_future_steps, max_k_allowed)
            max_start = total_terms - 1 - horizon_k
            if max_start < 1:
                continue

            start_idx = rng.randint(1, max_start)
            key = (total_terms, start_idx, horizon_k)
            if key in demo_keys:
                continue

            prefix = full_seq[: start_idx + 1]
            future = full_seq[start_idx + 1 : start_idx + 1 + horizon_k]

            prefix_text = game.seq_to_text(prefix)
            target_text = game.target_to_text(future)

            fewshot_examples.append(
                {
                    "total_terms": total_terms,
                    "start_idx": start_idx,
                    "future_steps": horizon_k,
                    "prefix_text": prefix_text,
                    "target_text": target_text,
                }
            )
            demo_keys.add(key)

        if len(fewshot_examples) < num_fewshot_examples:
            print(
                f"Warning: only created {len(fewshot_examples)} few-shot demos "
                f"out of requested {num_fewshot_examples}."
            )

    # -------- MAIN DATASET --------
    attempts = 0
    max_attempts = num_examples * 50

    while len(examples) < num_examples and attempts < max_attempts:
        attempts += 1

        total_terms = rng.randint(min_terms, max_terms)
        if total_terms not in cache:
            cache[total_terms] = game.fib_sequence(total_terms)
        full_seq = cache[total_terms]

        max_k_allowed = min(max_future_steps, total_terms - 2)
        if max_k_allowed < min_future_steps:
            continue

        horizon_k = rng.randint(min_future_steps, max_k_allowed)
        max_start = total_terms - 1 - horizon_k
        if max_start < 1:
            continue

        start_idx = rng.randint(1, max_start)

        key = (total_terms, start_idx, horizon_k)
        if key in seen_keys or key in demo_keys:
            continue

        seen_keys.add(key)

        ex = game.make_example(
            full_seq=full_seq,
            start_idx=start_idx,
            horizon_k=horizon_k,
            num_shots=num_shots,
            fewshot_examples=fewshot_examples,
        )
        examples.append(ex)

    if len(examples) < num_examples:
        print(
            f"Warning: only generated {len(examples)} unique Fibonacci examples "
            f"out of requested {num_examples}."
        )

    # -------- TRAIN / TEST SPLIT --------
    n_total = len(examples)
    n_test = max(1, int(round(n_total * test_fraction)))
    n_train = n_total - n_test

    indices = list(range(n_total))
    rng.shuffle(indices)

    test_idx = set(indices[:n_test])
    train_examples = [examples[i] for i in range(n_total) if i not in test_idx]
    test_examples = [examples[i] for i in range(n_total) if i in test_idx]

    assert len(train_examples) == n_train
    assert len(test_examples) == n_test

    return train_examples, test_examples
