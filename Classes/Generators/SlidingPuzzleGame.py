from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from collections import deque
import random


Move = str  # "UP", "DOWN", "LEFT", "RIGHT"


@dataclass
class SlidingPuzzleGame:
    board_size: int = 3  # supports 3x3 by default

    @property
    def solved_state(self) -> Tuple[int, ...]:
        """Return the canonical solved state for the board size."""
        n = self.board_size * self.board_size
        return tuple(list(range(1, n)) + [0])  # 0 denotes the blank

    # low-level helpers
    def _to_grid(self, state: Tuple[int, ...]) -> List[List[int]]:
        n = self.board_size
        return [list(state[i * n : (i + 1) * n]) for i in range(n)]

    def _find_blank(self, state: Tuple[int, ...]) -> Tuple[int, int]:
        idx = state.index(0)
        n = self.board_size
        return divmod(idx, n)

    def valid_moves(self, state: Tuple[int, ...]) -> List[Move]:
        row, col = self._find_blank(state)
        moves: List[Move] = []
        if row > 0:
            moves.append("UP")
        if row < self.board_size - 1:
            moves.append("DOWN")
        if col > 0:
            moves.append("LEFT")
        if col < self.board_size - 1:
            moves.append("RIGHT")
        return moves

    def apply_move(self, state: Tuple[int, ...], move: Move) -> Tuple[int, ...]:
        grid = self._to_grid(state)
        r, c = self._find_blank(state)

        drc = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1),
        }
        dr, dc = drc[move]
        nr, nc = r + dr, c + dc

        grid[r][c], grid[nr][nc] = grid[nr][nc], grid[r][c]
        flat: List[int] = []
        for row in grid:
            flat.extend(row)
        return tuple(flat)

    # scramble / solve 
    def scramble_state(
        self,
        min_moves: int,
        max_moves: int,
        rng: random.Random,
    ) -> Tuple[int, ...]:
        """Scramble from the solved state with random legal moves (always solvable)."""
        state = self.solved_state
        steps = rng.randint(min_moves, max_moves)
        last_move: Optional[Move] = None
        opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

        for _ in range(steps):
            moves = self.valid_moves(state)
            if last_move:
                moves = [m for m in moves if m != opposite[last_move]]
            move = rng.choice(moves)
            state = self.apply_move(state, move)
            last_move = move
        return state

    def solve_puzzle(self, start: Tuple[int, ...]) -> Optional[List[Move]]:
        """
        Solve via BFS to get the shortest sequence of moves back to the solved state.
        Returns the move list or None if unsolvable (should not happen with scramble_state).
        """
        target = self.solved_state
        if start == target:
            return []

        q = deque([start])
        visited: Dict[Tuple[int, ...], Tuple[Tuple[int, ...], Move]] = {start: (start, "")}

        while q:
            state = q.popleft()
            for move in self.valid_moves(state):
                nxt = self.apply_move(state, move)
                if nxt in visited:
                    continue
                visited[nxt] = (state, move)
                if nxt == target:
                    # reconstruct
                    path: List[Move] = []
                    cur = nxt
                    while cur != start:
                        prev, m = visited[cur]
                        path.append(m)
                        cur = prev
                    path.reverse()
                    return path
                q.append(nxt)
        return None

    def build_state_sequence(
        self, start: Tuple[int, ...], moves: List[Move]
    ) -> List[Tuple[int, ...]]:
        """Return [state0, state1, ...] applying moves in order."""
        states = [start]
        cur = start
        for m in moves:
            cur = self.apply_move(cur, m)
            states.append(cur)
        return states

    # formatting 
    def state_to_text(self, state: Tuple[int, ...]) -> str:
        """Readable grid with '_' for the blank."""
        grid = self._to_grid(state)
        lines = []
        for row in grid:
            line = " ".join(str(x) if x != 0 else "_" for x in row)
            lines.append(line)
        return "\n".join(lines)

    def moves_to_text(self, moves: List[Move]) -> str:
        return "\n".join(moves)

    # example builder 
    def make_example(
        self,
        start_state: Tuple[int, ...],
        solution_moves: List[Move],
        start_idx: int,
        horizon_k: int,
        num_shots: int = 0,
        fewshot_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        states = self.build_state_sequence(start_state, solution_moves)
        current_state = states[start_idx]
        future_moves = solution_moves[start_idx : start_idx + horizon_k]

        state_text = self.state_to_text(current_state)
        moves_text = self.moves_to_text(future_moves)

        fewshot_blocks: List[str] = []
        if num_shots > 0 and fewshot_examples:
            k = min(num_shots, len(fewshot_examples))
            demos = random.sample(fewshot_examples, k)
            for demo in demos:
                block = (
                    "Example:\n"
                    f"{demo['board_size']}x{demo['board_size']} sliding puzzle.\n"
                    f"Current board:\n{demo['state_text']}\n\n"
                    f"Next {demo['future_steps']} optimal moves:\n"
                    f"{demo['moves_text']}\n\n"
                )
                fewshot_blocks.append(block)

        fewshot_section = "".join(fewshot_blocks)

        prompt = (
            f"{fewshot_section}"
            f"Now solve this new instance.\n"
            f"You are solving a {self.board_size}x{self.board_size} sliding puzzle.\n"
            f"The goal state has the blank '_' in the bottom-right corner and numbers in order.\n"
            f"Current board:\n{state_text}\n\n"
            f"Provide the next {horizon_k} optimal moves toward the goal.\n\n"
            f"STRICT OUTPUT FORMAT (MANDATORY):\n"
            f"---------------------------------\n"
            f"You MUST output:\n"
            f"- EXACTLY {horizon_k} lines\n"
            f"- ONLY the words: UP, DOWN, LEFT, or RIGHT\n"
            f"- One move per line\n"
            f"- No extra text, no punctuation, no numbering, no blank lines\n\n"
            f"Any deviation from the required {horizon_k} lines is INVALID.\n"
        )

        return {
            "board_size": self.board_size,
            "start_step": start_idx,
            "future_steps": horizon_k,
            "scrambled_steps": len(solution_moves),
            "prompt": prompt,
            "target": moves_text,
        }


def generate_sliding_puzzle_dataset(
    num_examples: int,
    board_size: int = 3,
    min_scramble_moves: int = 10,
    max_scramble_moves: int = 30,
    min_future_steps: int = 1,
    max_future_steps: int = 6,
    num_shots: int = 0,
    num_fewshot_examples: int = 3,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate a dataset for sliding puzzles:
    - Each row: current board state, ask for next K optimal moves (UP/DOWN/LEFT/RIGHT).
    - Label: next K moves, one per line.
    - Uniqueness keyed by (start_state, horizon_k).
    """
    rng = random.Random(seed)
    game = SlidingPuzzleGame(board_size=board_size)

    examples: List[Dict[str, Any]] = []
    seen_keys = set()
    demo_keys = set()

    fewshot_examples: List[Dict[str, Any]] = []
    max_attempts = num_examples * 50

    # few-shot pool
    if num_shots > 0:
        attempts = 0
        while len(fewshot_examples) < num_fewshot_examples and attempts < max_attempts:
            attempts += 1
            scrambled = game.scramble_state(min_scramble_moves, max_scramble_moves, rng)
            solution = game.solve_puzzle(scrambled)
            if solution is None:
                continue

            if len(solution) < min_future_steps:
                continue

            horizon_k = rng.randint(
                min_future_steps, min(max_future_steps, len(solution))
            )
            max_start = len(solution) - horizon_k
            if max_start < 0:
                continue

            start_idx = rng.randint(0, max_start)
            states = game.build_state_sequence(scrambled, solution)
            state = states[start_idx]

            key = (state, horizon_k)
            if key in demo_keys:
                continue

            state_text = game.state_to_text(state)
            moves_text = game.moves_to_text(solution[start_idx : start_idx + horizon_k])

            fewshot_examples.append(
                {
                    "board_size": board_size,
                    "start_step": start_idx,
                    "future_steps": horizon_k,
                    "state_text": state_text,
                    "moves_text": moves_text,
                }
            )
            demo_keys.add(key)

    # main dataset
    attempts = 0
    while len(examples) < num_examples and attempts < max_attempts:
        attempts += 1

        scrambled = game.scramble_state(min_scramble_moves, max_scramble_moves, rng)
        solution = game.solve_puzzle(scrambled)
        if solution is None:
            continue

        if len(solution) < min_future_steps:
            continue

        horizon_k = rng.randint(min_future_steps, min(max_future_steps, len(solution)))
        max_start = len(solution) - horizon_k
        if max_start < 0:
            continue

        start_idx = rng.randint(0, max_start)
        states = game.build_state_sequence(scrambled, solution)
        state = states[start_idx]

        key = (state, horizon_k)
        if key in seen_keys or key in demo_keys:
            continue
        seen_keys.add(key)

        ex = game.make_example(
            start_state=scrambled,
            solution_moves=solution,
            start_idx=start_idx,
            horizon_k=horizon_k,
            num_shots=num_shots,
            fewshot_examples=fewshot_examples,
        )
        examples.append(ex)

    if len(examples) < num_examples:
        print(
            f"Warning: only generated {len(examples)} unique sliding puzzle examples "
            f"out of requested {num_examples}."
        )

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
