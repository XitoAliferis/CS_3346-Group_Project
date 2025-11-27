# imports 
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import random
from copy import deepcopy


# general hanoi game class
@dataclass
class HanoiGame:
    min_disks: int = 2
    max_disks: int = 8
    peg_labels: Tuple[str, str, str] = ("A", "B", "C")

    # generating the optimal moves to move the tower of n_disks from the src position to the dst position using the aux peg
    # returns: a list of (from peg, to peg)
    # based off this: https://www.geeksforgeeks.org/dsa/c-program-for-tower-of-hanoi/
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

    # builds a state sequence, which simulates the pegs over time given the move list
    # inputs number of disks, moves, and the source peg
    # returns the states at index i, the states after applying i moves
    # and the state[0] which is the intial config 
    def build_state_sequence(
        self,
        n_disks: int,
        moves: List[Tuple[int, int]],
        src: int,
    ) -> List[List[List[int]]]:

        # pegs as lists of disks, bottom -> top.
        pegs: List[List[int]] = [[], [], []]
        pegs[src] = list(range(n_disks, 0, -1))  # all disks on source peg

        states: List[List[List[int]]] = [deepcopy(pegs)]
        for (m_src, m_dst) in moves:
            disk = pegs[m_src].pop()
            pegs[m_dst].append(disk)
            states.append(deepcopy(pegs))
        return states

    # formatting the text so it's readable to LLMs
    def state_to_text(self, state: List[List[int]]) -> str:
        lines = []
        for label, peg in zip(self.peg_labels, state):
            if peg:
                # show from bottom to top
                disks_str = " ".join(str(d) for d in peg)
                lines.append(f"Peg {label}: {disks_str}")
            else:
                lines.append(f"Peg {label}: empty")
        return "\n".join(lines)

    # represents moves as "X->Y"
    def moves_to_text(
        self,
        moves: List[Tuple[int, int]],
    ) -> str:
        lines = []
        for src, dst in moves:
            lines.append(f"{self.peg_labels[src]}->{self.peg_labels[dst]}")
        return "\n".join(lines)

    # makes a single example based on the params defined
    # makes the prompt which is a text description of the current state + an instruction
    # allows for oneshot or fewshot examples too
    #   zero-shot -> num_shots = 0 
    #   one-shot  -> num_shots = 1 
    #   few-shot  -> num_shots > 1 
    # target is the next K moves (optimal) and it is printed once per line as "X->Y"

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

        # get current state & upcoming moves
        current_state = states[start_idx]
        future_moves = moves[start_idx : start_idx + horizon_k]

        query_state_text = self.state_to_text(current_state)
        query_moves_text = self.moves_to_text(future_moves)

        src_label = self.peg_labels[src]
        dst_label = self.peg_labels[dst]

        # ---- FEWSHOT ----
        fewshot_blocks = []
        if num_shots > 0 and fewshot_examples:
            k = min(num_shots, len(fewshot_examples))
            demo_samples = random.sample(fewshot_examples, k)

            for demo in demo_samples:
                demo_src_label = self.peg_labels[demo["src"]]
                demo_dst_label = self.peg_labels[demo["dst"]]

                # FEWSHOT IS NOW STRICT — GOOD FOR TRAINING
                block = (
                    f"Example:\n"
                    f"Puzzle with {demo['n_disks']} disks. Goal = peg {demo_dst_label}.\n"
                    f"State:\n{demo['state_text']}\n\n"
                    f"Next {demo['future_steps']} optimal moves:\n"
                    f"{demo['moves_text']}\n\n"
                )
                fewshot_blocks.append(block)

        fewshot_section = "".join(fewshot_blocks)

        # ---- STRICT FORMAT PROMPT ----
        prompt = (
            f"{fewshot_section}"
            f"Now solve this new instance.\n"
            f"You are solving the Towers of Hanoi puzzle with {n_disks} disks.\n"
            f"The goal is to move all disks to peg {dst_label}.\n"
            f"The pegs are labeled {', '.join(self.peg_labels)}.\n\n"
            f"Current configuration:\n{query_state_text}\n\n"
            f"Provide the next {horizon_k} optimal moves.\n\n"
            f"STRICT OUTPUT FORMAT (MANDATORY):\n"
            f"---------------------------------\n"
            f"You MUST output:\n"
            f"- EXACTLY {horizon_k} lines\n"
            f"- NOTHING except moves\n"
            f"- Each line MUST match the regex: ^[A-C]->[A-C]$\n"
            f"- No English words\n"
            f"- No explanations\n"
            f"- No punctuation except '->'\n"
            f"- No blank lines\n"
            f"- No numbering\n"
            f"- No spaces before or after\n"
            f"- No commentary\n\n"
            f"Any deviation from the required {horizon_k} lines is INVALID.\n"
        )

        target = query_moves_text

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

# generate the entire dataset
# input params include the total examples produced, the min and max disks, min and max steps, number of shots (examples in the prompt), and the seed
# returns the full dataset
# notes
#  - each row shows a current state, asks for K next moves and the label is the next K moves as "X->Y" lines
#  - no two rows share the same config (defined as the combination of: n_disks, src, dst, aux, start_step, future_steps)
#  - if there are shots (examples), we create them first and ensure they never appear in the dataset (avoiding leakage)
def generate_hanoi_dataset(
    num_examples: int,
    min_disks: int = 2,
    max_disks: int = 8,
    min_future_steps: int = 1,
    max_future_steps: int = 8,
    num_shots: int = 0,
    num_fewshot_examples: int = 3,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)  # seed for reproducibility
    game = HanoiGame(min_disks=min_disks, max_disks=max_disks)

    examples: List[Dict[str, Any]] = []  # all rows
    seen_keys = set()   # keys for dataset rows (n_disks, src, dst, aux, start_idx, horizon_k)
    demo_keys = set()   # keys reserved for few-shot demos

    # cache: (n_disks, src, dst, aux) -> (moves, states)
    cache: Dict[Tuple[int, int, int, int], Tuple[List[Tuple[int, int]], List[List[List[int]]]]] = {}

    # ---------- build few-shot pool (if needed) ----------
    fewshot_examples: List[Dict[str, Any]] = []

    if num_shots > 0:
        attempts = 0
        max_attempts = num_fewshot_examples * 50

        while len(fewshot_examples) < num_fewshot_examples and attempts < max_attempts:
            attempts += 1

            # pick random number of disks
            n_disks = rng.randint(min_disks, max_disks)

            # randomize source, destination, auxiliary pegs
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

    # ---------- generate main dataset ----------
    attempts = 0
    max_attempts = num_examples * 50

    while len(examples) < num_examples and attempts < max_attempts:
        attempts += 1

        # number of disks
        n_disks = rng.randint(min_disks, max_disks)

        # randomize source, destination, auxiliary for this example
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

    # ---------- split into 90% train / 10% test ----------
    n_total = len(examples)
    n_test = max(1, int(round(n_total * test_fraction)))  # ensure at least 1
    n_train = n_total - n_test

    indices = list(range(n_total))
    rng.shuffle(indices)

    test_idx = set(indices[:n_test])
    train_examples = [examples[i] for i in range(n_total) if i not in test_idx]
    test_examples  = [examples[i] for i in range(n_total) if i in test_idx]

    # sanity
    assert len(train_examples) == n_train
    assert len(test_examples) == n_test

    return train_examples, test_examples