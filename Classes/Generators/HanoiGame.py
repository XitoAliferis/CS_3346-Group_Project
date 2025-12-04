from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import random
from copy import deepcopy


@dataclass
class HanoiGame:
    min_disks: int = 2
    max_disks: int = 8
    peg_labels: Tuple[str, str, str] = ("A", "B", "C")

    def generate_optimal_moves(
        self,
        n_disks: int,
        src: int = 0,
        dst: int = 2,
        aux: int = 1,
    ) -> List[Tuple[int, int]]:
        moves: List[Tuple[int, int]] = []

        def solve(k: int, s: int, d: int, a: int):
            if k == 0:
                return
            solve(k - 1, s, a, d)
            moves.append((s, d))
            solve(k - 1, a, d, s)

        solve(n_disks, src, dst, aux)
        return moves

    def build_state_sequence(
        self,
        n_disks: int,
        moves: List[Tuple[int, int]],
        src: int,
    ) -> List[List[List[int]]]:
        pegs: List[List[int]] = [[], [], []]
        pegs[src] = list(range(n_disks, 0, -1))

        states: List[List[List[int]]] = [deepcopy(pegs)]
        for (m_src, m_dst) in moves:
            disk = pegs[m_src].pop()
            pegs[m_dst].append(disk)
            states.append(deepcopy(pegs))
        return states

    def state_to_text(self, state: List[List[int]]) -> str:
        lines = []
        for label, peg in zip(self.peg_labels, state):
            if peg:
                disks_str = " ".join(str(d) for d in peg)
                lines.append(f"Peg {label}: {disks_str}")
            else:
                lines.append(f"Peg {label}: empty")
        return "\n".join(lines)

    def moves_to_text(
        self,
        moves: List[Tuple[int, int]],
    ) -> str:
        lines = []
        for src, dst in moves:
            lines.append(f"{self.peg_labels[src]}->{self.peg_labels[dst]}")
        return "\n".join(lines)

    def make_example(
        self,
        n_disks: int,
        states: List[List[List[int]]],
        moves: List[Tuple[int, int]],
        start_idx: int,
        horizon_k: int,
        src: int,
        dst: int,
        aux: int,
        num_shots: int = 0,
        fewshot_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        current_state = states[start_idx]
        future_moves = moves[start_idx : start_idx + horizon_k]

        query_state_text = self.state_to_text(current_state)
        query_moves_text = self.moves_to_text(future_moves)
        dst_label = self.peg_labels[dst]

        # ---- FEWSHOT (chat formatted) ----
        fewshot_blocks: List[str] = []
        if num_shots > 0 and fewshot_examples:
            k = min(num_shots, len(fewshot_examples))
            demo_samples = random.sample(fewshot_examples, k)
            for demo in demo_samples:
                demo_dst_label = self.peg_labels[demo["dst"]]
                demo_user = (
                    "You are solving the Towers of Hanoi puzzle.\n"
                    f"Number of disks: {demo['n_disks']}.\n"
                    f"The goal is to move all disks to peg {demo_dst_label}.\n"
                    f"The pegs are labeled {', '.join(self.peg_labels)}.\n\n"
                    f"Current configuration:\n{demo['state_text']}\n\n"
                    f"Provide the next {demo['future_steps']} optimal moves.\n\n"
                    "STRICT OUTPUT FORMAT (MANDATORY):\n"
                    "---------------------------------\n"
                    "You MUST output:\n"
                    f"- EXACTLY {demo['future_steps']} lines\n"
                    "- NOTHING except moves\n"
                    "- Each line MUST match the regex: ^[A-C]->[A-C]$\n"
                    "- No English words\n"
                    "- No explanations\n"
                    "- No punctuation except '->'\n"
                    "- No blank lines\n"
                    "- No numbering\n"
                    "- No spaces before or after\n"
                    "- No commentary\n\n"
                    f"Any deviation from the required {demo['future_steps']} lines is INVALID.\n"
                )

                block = (
                    "<|im_start|>user\n"
                    f"{demo_user}"
                    "<|im_end|>\n"
                    "<|im_start|>assistant\n"
                    f"{demo['moves_text']}\n"
                    "<|im_end|>\n"
                )
                fewshot_blocks.append(block)

        fewshot_section = "".join(fewshot_blocks)

        user_prompt = (
            "Now solve this new instance.\n"
            f"You are solving the Towers of Hanoi puzzle with {n_disks} disks.\n"
            f"The goal is to move all disks to peg {dst_label}.\n"
            f"The pegs are labeled {', '.join(self.peg_labels)}.\n\n"
            f"Current configuration:\n{query_state_text}\n\n"
            f"Provide the next {horizon_k} optimal moves.\n\n"
            "STRICT OUTPUT FORMAT (MANDATORY):\n"
            "---------------------------------\n"
            "You MUST output:\n"
            f"- EXACTLY {horizon_k} lines\n"
            "- NOTHING except moves\n"
            "- Each line MUST match the regex: ^[A-C]->[A-C]$\n"
            "- No English words\n"
            "- No explanations\n"
            "- No punctuation except '->'\n"
            "- No blank lines\n"
            "- No numbering\n"
            "- No spaces before or after\n"
            "- No commentary\n\n"
            f"Any deviation from the required {horizon_k} lines is INVALID.\n"
        )

        prompt = (
            f"{fewshot_section}"
            "<|im_start|>user\n"
            f"{user_prompt}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        target = f"{query_moves_text}\n<|im_end|>"

        return {
            "n_disks": n_disks,
            "start_step": start_idx,
            "future_steps": horizon_k,
            "src": src,
            "dst": dst,
            "aux": aux,
            "num_shots": num_shots,
            "prompt": prompt,
            "target": target,
        }


def generate_hanoi_dataset(
    num_examples: int,
    min_disks: int = 2,
    max_disks: int = 8,
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
    rng = random.Random(seed)
    game = HanoiGame(min_disks=min_disks, max_disks=max_disks)

    examples: List[Dict[str, Any]] = []
    seen_keys = set()
    demo_keys = set()

    cache: Dict[
        Tuple[int, int, int, int],
        Tuple[List[Tuple[int, int]], List[List[List[int]]]],
    ] = {}

    # ---------- few-shot pool ----------
    fewshot_examples: List[Dict[str, Any]] = []
    if num_shots > 0:
        attempts = 0
        max_attempts = num_fewshot_examples * 50
        while len(fewshot_examples) < num_fewshot_examples and attempts < max_attempts:
            attempts += 1

            n_disks = rng.randint(min_disks, max_disks)
            peg_idxs = [0, 1, 2]
            src, dst = rng.sample(peg_idxs, 2)
            aux = next(i for i in peg_idxs if i not in (src, dst))

            cache_key = (n_disks, src, dst, aux)
            if cache_key not in cache:
                moves = game.generate_optimal_moves(n_disks, src=src, dst=dst, aux=aux)
                states = game.build_state_sequence(n_disks, moves, src=src)
                cache[cache_key] = (moves, states)
            else:
                moves, states = cache[cache_key]

            total_moves = len(moves)
            if total_moves < min_future_steps:
                continue

            max_k_allowed = min(max_future_steps, total_moves)
            horizon_k = rng.randint(min_future_steps, max_k_allowed)
            max_start = total_moves - horizon_k
            if max_start < 0:
                continue

            start_idx = rng.randint(0, max_start)

            key = (n_disks, src, dst, aux, start_idx, horizon_k)
            if key in demo_keys:
                continue

            state_text = game.state_to_text(states[start_idx])
            moves_text = game.moves_to_text(moves[start_idx : start_idx + horizon_k])

            fewshot_examples.append(
                {
                    "n_disks": n_disks,
                    "start_step": start_idx,
                    "future_steps": horizon_k,
                    "state_text": state_text,
                    "moves_text": moves_text,
                    "src": src,
                    "dst": dst,
                    "aux": aux,
                }
            )
            demo_keys.add(key)

        if len(fewshot_examples) < num_fewshot_examples:
            print(
                f"Warning: only created {len(fewshot_examples)} few-shot demos "
                f"out of requested {num_fewshot_examples}."
            )

    # ---------- main dataset ----------
    attempts = 0
    max_attempts = num_examples * 50
    while len(examples) < num_examples and attempts < max_attempts:
        attempts += 1

        n_disks = rng.randint(min_disks, max_disks)
        peg_idxs = [0, 1, 2]
        src, dst = rng.sample(peg_idxs, 2)
        aux = next(i for i in peg_idxs if i not in (src, dst))

        cache_key = (n_disks, src, dst, aux)
        if cache_key not in cache:
            moves = game.generate_optimal_moves(n_disks, src=src, dst=dst, aux=aux)
            states = game.build_state_sequence(n_disks, moves, src=src)
            cache[cache_key] = (moves, states)
        else:
            moves, states = cache[cache_key]

        total_moves = len(moves)
        if total_moves < min_future_steps:
            continue

        max_k_allowed = min(max_future_steps, total_moves)
        horizon_k = rng.randint(min_future_steps, max_k_allowed)
        max_start = total_moves - horizon_k
        if max_start < 0:
            continue

        start_idx = rng.randint(0, max_start)

        key = (n_disks, src, dst, aux, start_idx, horizon_k)
        if key in seen_keys or key in demo_keys:
            continue
        seen_keys.add(key)

        ex = game.make_example(
            n_disks=n_disks,
            states=states,
            moves=moves,
            start_idx=start_idx,
            horizon_k=horizon_k,
            src=src,
            dst=dst,
            aux=aux,
            num_shots=num_shots,
            fewshot_examples=fewshot_examples,
        )
        examples.append(ex)

    if len(examples) < num_examples:
        print(
            f"Warning: only generated {len(examples)} unique examples "
            f"out of requested {num_examples}."
        )

    # ---------- stratified split by difficulty (n_disks, future_steps) ----------
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

        # bucket examples by (n_disks, future_steps) as a difficulty proxy
        buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for ex in examples:
            key = (ex["n_disks"], ex["future_steps"])
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

        # iterate over buckets; within each bucket spread examples across splits
        for key, ex_list in buckets.items():
            if total_remaining() == 0:
                break
            for ex in ex_list:
                if total_remaining() == 0:
                    break
                # choose split with largest remaining capacity
                candidates = [s for s, r in remaining.items() if r > 0]
                if not candidates:
                    break
                best_split = max(candidates, key=lambda s: remaining[s])
                splits[best_split].append(ex)
                remaining[best_split] -= 1

        # final sanity checks
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
