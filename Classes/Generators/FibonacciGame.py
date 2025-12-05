from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import random


@dataclass
class FibonacciGame:
    min_terms: int = 10
    max_terms: int = 40
    # NEW: only show this many most-recent prefix terms in the prompt
    max_visible_prefix_terms: int = 12

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

        # Logical prefix / target
        prefix = full_seq[: start_idx + 1]
        future = full_seq[start_idx + 1 : start_idx + 1 + horizon_k]

        # Only *show* the last max_visible_prefix_terms in the prompt
        visible_prefix = prefix[-self.max_visible_prefix_terms :]
        visible_prefix_text = self.seq_to_text(visible_prefix)
        target_text = self.target_to_text(future)

        # ---- FEWSHOT (inside chat format) ----
        fewshot_blocks: List[str] = []
        if num_shots > 0 and fewshot_examples:
            k = min(num_shots, len(fewshot_examples))
            demos = random.sample(fewshot_examples, k)

            for demo in demos:
                demo_block = (
                    "<|im_start|>user\n"
                    "Fibonacci sequence prediction.\n"
                    "You are given a contiguous slice of a Fibonacci-like sequence.\n"
                    "Sequence so far (from left to right):\n"
                    f"{demo['visible_prefix_text']}\n"
                    f"Provide the next {demo['future_steps']} numbers.\n"
                    "STRICT OUTPUT FORMAT (MANDATORY):\n"
                    "---------------------------------\n"
                    "You MUST output:\n"
                    f"- EXACTLY {demo['future_steps']} lines\n"
                    "- NOTHING except integers\n"
                    "- Each line MUST contain a single integer (no spaces)\n"
                    "- No English words\n"
                    "- No explanations\n"
                    "- No blank lines\n"
                    "- No punctuation\n"
                    "- No numbering\n\n"
                    f"Any deviation from the required {demo['future_steps']} lines is INVALID.\n"
                    "<|im_end|>\n"
                    "<|im_start|>assistant\n"
                    f"{demo['target_text']}\n"
                    "<|im_end|>\n"
                )
                fewshot_blocks.append(demo_block)

        fewshot_section = "".join(fewshot_blocks)

        # ---- MAIN PROMPT (user turn) ----
        user_prompt = (
            "Now solve this new instance.\n"
            "You are given a contiguous slice of a Fibonacci-like sequence.\n"
            "Sequence so far (from left to right):\n"
            f"{visible_prefix_text}\n\n"
            f"Provide the next {horizon_k} numbers in the sequence.\n\n"
            "STRICT OUTPUT FORMAT (MANDATORY):\n"
            "---------------------------------\n"
            "You MUST output:\n"
            f"- EXACTLY {horizon_k} lines\n"
            "- NOTHING except integers\n"
            "- Each line MUST contain a single integer (no spaces)\n"
            "- No English words\n"
            "- No explanations\n"
            "- No blank lines\n"
            "- No punctuation\n"
            "- No numbering\n\n"
            f"Any deviation from the required {horizon_k} lines is INVALID.\n"
        )

        prompt = (
            f"{fewshot_section}"
            "<|im_start|>user\n"
            f"{user_prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        target = f"{target_text}\n<|im_end|>"

        return {
            "total_terms": len(full_seq),
            "start_idx": start_idx,
            "future_steps": horizon_k,
            "prompt": prompt,
            "target": target,
        }


def generate_fibonacci_dataset(
    num_examples: int,
    min_terms: int = 10,
    max_terms: int = 40,
    min_future_steps: int = 1,
    max_future_steps: int = 8,
    num_shots: int = 0,
    num_fewshot_examples: int = 3,
    test_size: int = 300,
    size_x: int = 500,
    size_y: int = 1500,
    size_z: int = 3000,
    seed: int = 0,
) -> Tuple[
    List[Dict[str, Any]],  # test set
    List[Dict[str, Any]],  # set x
    List[Dict[str, Any]],  # set y
    List[Dict[str, Any]],  # set z
]:
    """
    Similar design as generate_hanoi_dataset, but for Fibonacci:
    - Each row: prefix of sequence, ask for next K terms
    - Label: next K terms, one per line
    - Difficulty bucket: (total_terms, future_steps)
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
            visible_prefix = prefix[-game.max_visible_prefix_terms :]
            future = full_seq[start_idx + 1 : start_idx + 1 + horizon_k]

            prefix_text = game.seq_to_text(prefix)
            visible_prefix_text = game.seq_to_text(visible_prefix)
            target_text = game.target_to_text(future)

            fewshot_examples.append(
                {
                    "total_terms": total_terms,
                    "start_idx": start_idx,
                    "future_steps": horizon_k,
                    "prefix_text": prefix_text,                # full (metadata)
                    "visible_prefix_text": visible_prefix_text,  # truncated (used in prompt)
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

    # -------- STRATIFIED SPLIT BY DIFFICULTY (total_terms, future_steps) --------
    def stratified_split(
        examples: List[Dict[str, Any]],
        test_size: int,
        size_x: int,
        size_y: int,
        size_z: int,
        rng: random.Random,
    ):
        total_needed = test_size + size_x + size_y + size_z
        assert total_needed <= len(examples), (
            "Requested split sizes exceed dataset size: "
            f"needed={total_needed}, available={len(examples)}"
        )

        # bucket by (sequence length, horizon) as difficulty proxy
        buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for ex in examples:
            key = (ex["total_terms"], ex["future_steps"])
            buckets.setdefault(key, []).append(ex)

        for key in buckets:
            rng.shuffle(buckets[key])

        splits = {
            "test": [],
            "x": [],
            "y": [],
            "z": [],
        }
        remaining = {
            "test": test_size,
            "x": size_x,
            "y": size_y,
            "z": size_z,
        }

        def total_remaining():
            return sum(remaining.values())

        # spread each bucket across the four splits
        for key, ex_list in buckets.items():
            if total_remaining() == 0:
                break
            for ex in ex_list:
                if total_remaining() == 0:
                    break
                candidates = [s for s, r in remaining.items() if r > 0]
                if not candidates:
                    break
                # greedy largest-remaining allocation (matches Hanoi for consistency)
                best_split = max(candidates, key=lambda s: remaining[s])
                splits[best_split].append(ex)
                remaining[best_split] -= 1

        # sanity checks
        assert len(splits["test"]) == test_size, "test_size mismatch"
        assert len(splits["x"]) == size_x, "size_x mismatch"
        assert len(splits["y"]) == size_y, "size_y mismatch"
        assert len(splits["z"]) == size_z, "size_z mismatch"

        return (
            splits["test"],
            splits["x"],
            splits["y"],
            splits["z"],
        )

    return stratified_split(
        examples,
        test_size=test_size,
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        rng=rng,
    )
