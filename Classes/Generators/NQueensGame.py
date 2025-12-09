from __future__ import annotations  # i hate python.
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ortools.sat.python import cp_model

@dataclass
class NQueensLayout:
    """
    Representation convention (matches your original code):

    - Board is n x n.
    - queenColumnPositions[c] = r means:
        * there is a queen in column c, row r
        * there is exactly one queen per column in a full solution
    """
    n: int
    queenColumnPositions: List[int]

    def isValid(self) -> bool:
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # same row
                if self.queenColumnPositions[i] == self.queenColumnPositions[j]:
                    return False
                # same diagonal
                if abs(self.queenColumnPositions[i] - self.queenColumnPositions[j]) == j - i:
                    return False
        return True

    def toTextTable(self) -> str:
        """
        Full board text (all columns assigned).
        X = queen, ∙ = empty.
        """
        table: List[str] = []
        for row in range(self.n):
            row_cells: List[str] = []
            for col in range(self.n):
                if self.queenColumnPositions[col] == row:
                    row_cells.append("X")
                else:
                    row_cells.append("∙")
            table.append(" ".join(row_cells))
        return "\n".join(table)

    def toTextColumnList(self) -> str:
        return ",".join(str(i) for i in self.queenColumnPositions)

    def asDict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "queenColumnPositions": self.queenColumnPositions,
        }

    @staticmethod
    def fromPositions(positions: List[int]) -> NQueensLayout:
        return NQueensLayout(len(positions), positions)


# ---------------------------------------------------------------------------
# Solver helpers (unchanged idea, slightly cleaned)
# ---------------------------------------------------------------------------

def calculateNQueensSolutions(
    n: int,
    maxCount: int,
    logProgress: bool = False,
) -> List[NQueensLayout]:
    """
    Generate up to maxCount distinct solutions for the n-queens problem.

    Follows the standard CP-SAT formulation:
    - one IntVar per column (value = row index)
    - all_different for rows
    - all_different for major and minor diagonals
    """
    model = cp_model.CpModel()

    # cp-sat API uses NewIntVar
    queenVars = [model.NewIntVar(0, n - 1, f"x_{i}") for i in range(n)]

    model.AddAllDifferent(queenVars)
    model.AddAllDifferent(queenVars[i] + i for i in range(n))
    model.AddAllDifferent(queenVars[i] - i for i in range(n))

    class NQueenSolutionPrinter(cp_model.CpSolverSolutionCallback):
        def __init__(
            self,
            queens: List[cp_model.IntVar],
            maxCount: int,
            outputList: List[NQueensLayout],
            logProgress: bool = False,
        ):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self._solved = 0
            self._queens = queens
            self._maxCount = maxCount
            self._outputList = outputList
            self._logProgress = logProgress

        def on_solution_callback(self):
            positions = [self.Value(var) for var in self._queens]
            self._outputList.append(NQueensLayout.fromPositions(positions))
            self._solved += 1
            if self._logProgress and (self._solved % 100 == 0 or self._maxCount <= 100):
                print(
                    f"[NQueensSolver] n = {n}, "
                    f"found {self._solved} of {self._maxCount} requested"
                )
            if self._solved >= self._maxCount:
                self.StopSearch()

    results: List[NQueensLayout] = []
    solver = cp_model.CpSolver()
    solutionPrinter = NQueenSolutionPrinter(queenVars, maxCount, results, logProgress)
    solver.parameters.enumerate_all_solutions = True
    if logProgress:
        print(f"[NQueensSolver] Starting solve for n = {n}")
    solver.Solve(model, solutionPrinter)
    if logProgress:
        print(
            f"[NQueensSolver] Done for n = {n}, "
            f"found {len(results)} of {maxCount} requested "
            f"({len(results) / maxCount * 100:.2f}%)"
        )
    return results


# ---------------------------------------------------------------------------
# N-Queens "game" for dataset generation
# ---------------------------------------------------------------------------

@dataclass
class NQueensGame:
    """
    Dataset view of N-Queens.

    We treat a *solution* as a sequence of decisions:
      - we fill columns from left to right: column 0, 1, 2, ..., n-1
      - at step k we choose a row index in [0, n-1] for column k

    A training example is:
      - given a partial board (queens in first `start_col` columns),
      - predict the next `horizon_k` row indices (one per line).
    """

    def partial_state_to_text(self, layout: NQueensLayout, upto_col: int) -> str:
        """
        Compact Hanoi-style representation:
        rows: r0 r1 r2 ...
        Only columns [0, upto_col) are shown.
        """
        rows = layout.queenColumnPositions[:upto_col]
        if len(rows) == 0:
            return "rows: (none)"
        return "rows: " + " ".join(str(r) for r in rows)

    def moves_to_text(self, rows: List[int]) -> str:
        """Each row index on its own line (0-based)."""
        return "\n".join(str(r) for r in rows)

    def make_example(
        self,
        layout: NQueensLayout,
        start_col: int,
        horizon_k: int,
        num_shots: int = 0,
        fewshot_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:

        n = layout.n
        state_text = self.partial_state_to_text(layout, upto_col=start_col)
        future_rows = layout.queenColumnPositions[start_col : start_col + horizon_k]
        moves_text = "\n".join(str(r) for r in future_rows)

        # ---- FEWSHOT (chat formatted) ----
        fewshot_blocks: List[str] = []
        if num_shots > 0 and fewshot_examples:
            k = min(num_shots, len(fewshot_examples))
            demos = random.sample(fewshot_examples, k)
            for demo in demos:
                demo_user = (
                    f"You are solving the N-Queens puzzle (size {demo['n']}).\n"
                    "Each column must contain exactly one queen, row = 0..N-1.\n\n"
                    f"Partial placement:\n{demo['state_text']}\n\n"
                    f"Provide the next {demo['future_steps']} row indices.\n\n"
                    "STRICT OUTPUT FORMAT:\n"
                    f"- EXACTLY {demo['future_steps']} lines\n"
                    "- integers only, no text\n"
                    "- no blank lines\n"
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

        # ---- MAIN PROMPT ----
        user_prompt = (
            f"You are solving the N-Queens puzzle (size {n}).\n"
            "A queen must be placed in each column, row = 0..N-1.\n\n"
            f"Partial placement:\n{state_text}\n\n"
            f"Provide the next {horizon_k} row indices.\n\n"
            "STRICT OUTPUT FORMAT:\n"
            f"- EXACTLY {horizon_k} lines\n"
            "- integers only, no text\n"
            "- no blank lines\n"
        )

        prompt = (
            f"{fewshot_section}"
            "<|im_start|>user\n"
            f"{user_prompt}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        target = f"{moves_text}\n<|im_end|>"

        return {
            "n": n,
            "start_step": start_col,
            "future_steps": horizon_k,
            "prompt": prompt,
            "target": target,
        }



# ---------------------------------------------------------------------------
# Dataset generation (aligned with Hanoi / Sliding)
# ---------------------------------------------------------------------------

def generate_nqueens_dataset(
    num_examples: int,
    min_n: int = 4,
    max_n: int = 10,
    min_future_steps: int = 1,
    max_future_steps: int = 4,
    num_shots: int = 0,
    num_fewshot_examples: int = 3,
    test_size: int = 300,
    size_x: int = 500,
    size_y: int = 1500,
    size_z: int = 3000,
    seed: int = 0,
    max_solutions_per_n: int = 512,
) -> Tuple[
    List[Dict[str, Any]],  # test set
    List[Dict[str, Any]],  # set x
    List[Dict[str, Any]],  # set y
    List[Dict[str, Any]],  # set z
]:
    """
    Generate a dataset for N-Queens that mirrors the style of:

      - generate_hanoi_dataset
      - generate_sliding_puzzle_dataset

    Each example:
      - shows a partial N-Queens board (queens in first `start_step` columns),
      - asks for the next `future_steps` row indices (0-based), one per line.

    Returns 4 splits: test, x, y, z, stratified by (n, future_steps).
    """
    rng = random.Random(seed)
    game = NQueensGame()

    # cache: n -> list of solutions for that n
    solution_cache: Dict[int, List[NQueensLayout]] = {}

    def get_solutions_for_n(n: int) -> List[NQueensLayout]:
        if n not in solution_cache:
            sols = calculateNQueensSolutions(n, maxCount=max_solutions_per_n, logProgress=False)
            solution_cache[n] = sols
        return solution_cache[n]

    examples: List[Dict[str, Any]] = []
    seen_keys = set()
    demo_keys = set()

    # ---------- FEW-SHOT POOL ----------
    fewshot_examples: List[Dict[str, Any]] = []
    if num_shots > 0:
        attempts = 0
        max_attempts = num_fewshot_examples * 50
        while len(fewshot_examples) < num_fewshot_examples and attempts < max_attempts:
            attempts += 1

            n = rng.randint(min_n, max_n)
            sols = get_solutions_for_n(n)
            if not sols:
                continue

            layout = rng.choice(sols)

            # cannot ask for more steps than there are columns
            max_k_allowed = min(max_future_steps, n)
            if max_k_allowed < min_future_steps:
                continue

            horizon_k = rng.randint(min_future_steps, max_k_allowed)
            max_start = n - horizon_k
            if max_start < 0:
                continue

            start_col = rng.randint(0, max_start)

            # Uniqueness key: (n, prefix configuration, horizon_k)
            prefix = tuple(layout.queenColumnPositions[:start_col])
            key = (n, prefix, horizon_k)
            if key in demo_keys:
                continue

            state_text = game.partial_state_to_text(layout, upto_col=start_col)
            future_rows = layout.queenColumnPositions[start_col : start_col + horizon_k]
            moves_text = game.moves_to_text(future_rows)

            fewshot_examples.append(
                {
                    "n": n,
                    "start_step": start_col,
                    "future_steps": horizon_k,
                    "state_text": state_text,
                    "moves_text": moves_text,
                }
            )
            demo_keys.add(key)

        if len(fewshot_examples) < num_fewshot_examples:
            print(
                f"Warning: only created {len(fewshot_examples)} N-Queens few-shot demos "
                f"out of requested {num_fewshot_examples}."
            )

    # ---------- MAIN DATASET ----------
    attempts = 0
    max_attempts = num_examples * 50
    while len(examples) < num_examples and attempts < max_attempts:
        attempts += 1

        n = rng.randint(min_n, max_n)
        sols = get_solutions_for_n(n)
        if not sols:
            continue

        layout = rng.choice(sols)

        max_k_allowed = min(max_future_steps, n)
        if max_k_allowed < min_future_steps:
            continue

        horizon_k = rng.randint(min_future_steps, max_k_allowed)
        max_start = n - horizon_k
        if max_start < 0:
            continue

        start_col = rng.randint(0, max_start)

        prefix = tuple(layout.queenColumnPositions[:start_col])
        key = (n, prefix, horizon_k)
        if key in seen_keys or key in demo_keys:
            continue
        seen_keys.add(key)

        ex = game.make_example(
            layout=layout,
            start_col=start_col,
            horizon_k=horizon_k,
            num_shots=num_shots,
            fewshot_examples=fewshot_examples,
        )
        examples.append(ex)

    if len(examples) < num_examples:
        print(
            f"Warning: only generated {len(examples)} unique N-Queens examples "
            f"out of requested {num_examples}."
        )

    # ---------- STRATIFIED SPLIT BY DIFFICULTY (n, future_steps) ----------
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

        # difficulty proxy: (board size n, prediction horizon future_steps)
        buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for ex in examples:
            key = (ex["n"], ex["future_steps"])
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

        # Greedy, same style as Hanoi / Sliding:
        for key, ex_list in buckets.items():
            if total_remaining() == 0:
                break
            for ex in ex_list:
                if total_remaining() == 0:
                    break
                candidates = [s for s, r in remaining.items() if r > 0]
                if not candidates:
                    break
                # choose split with largest remaining capacity
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
