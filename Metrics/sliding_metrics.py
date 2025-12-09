import json
import re
from typing import List, Dict, Tuple
from pathlib import Path
from Levenshtein import distance as levenshtein_distance   # pip install python-Levenshtein

# =============================
# Utility: Parse move strings
# =============================
MOVE_RE = re.compile(r"^(UP|DOWN|LEFT|RIGHT)$")


def parse_moves(block: str) -> List[str]:
    """Extract only well-formed moves like 'UP', 'DOWN', 'LEFT', 'RIGHT' from a block of text."""
    moves: List[str] = []
    for l in block.splitlines():
        l = l.strip()
        if MOVE_RE.match(l):
            moves.append(l)
    return moves


# =============================
# Utility: Parse board from prompt
# =============================
def parse_board_from_prompt(prompt: str) -> Tuple[List[int], int]:
    """
    Extract the *last* 'Current board:' grid from the prompt.

    Returns:
        (state, board_size)
        state: flattened list of ints, 0 for blank
        board_size: side length n (for an n x n board)
    """
    lines = prompt.splitlines()

    boards: List[List[str]] = []
    collecting = False
    cur_board_lines: List[str] = []

    for line in lines:
        if "Current board:" in line:
            # Finish any previous board
            if collecting and cur_board_lines:
                boards.append(cur_board_lines)
            collecting = True
            cur_board_lines = []
            continue

        if collecting:
            stripped = line.strip()
            # Board section ends at first blank line
            if stripped == "":
                if cur_board_lines:
                    boards.append(cur_board_lines)
                    cur_board_lines = []
                collecting = False
            else:
                cur_board_lines.append(stripped)

    # In case the prompt ends right after the board without a blank line
    if collecting and cur_board_lines:
        boards.append(cur_board_lines)

    if not boards:
        return [], 0

    board_lines = boards[-1]  # use the last board (the new instance)
    if not board_lines:
        return [], 0

    first_row_tokens = board_lines[0].split()
    n = len(first_row_tokens)
    if n == 0:
        return [], 0

    state: List[int] = []
    for row in board_lines:
        tokens = row.split()
        for t in tokens:
            if t == "_":
                state.append(0)
            else:
                try:
                    state.append(int(t))
                except ValueError:
                    # Ignore unexpected tokens
                    pass

    # Basic sanity check: must be a square board
    if len(state) != n * n:
        return [], 0

    return state, n


def clone_state(state: List[int]) -> List[int]:
    return state.copy()


def apply_move_inplace(state: List[int], move: str, board_size: int) -> bool:
    """
    Apply a move to the sliding puzzle state *in place*.
    Returns True if the move is legal & applied, False otherwise.
    """
    if board_size <= 0:
        return False
    if not MOVE_RE.match(move):
        return False
    if 0 not in state:
        return False

    idx = state.index(0)
    n = board_size
    r, c = divmod(idx, n)

    dr = dc = 0
    if move == "UP":
        dr, dc = -1, 0
    elif move == "DOWN":
        dr, dc = 1, 0
    elif move == "LEFT":
        dr, dc = 0, -1
    elif move == "RIGHT":
        dr, dc = 0, 1

    nr, nc = r + dr, c + dc
    # Check bounds
    if not (0 <= nr < n and 0 <= nc < n):
        return False

    new_idx = nr * n + nc
    # Swap blank with the target tile
    state[idx], state[new_idx] = state[new_idx], state[idx]
    return True


# =============================
# 1. Syntax validity
# =============================
def syntax_accuracy(raw_pred: str, horizon_k: int) -> float:
    """
    Fraction of lines that are valid move tokens, normalized by the expected
    number of lines (horizon_k). This penalizes under/over-generation.
    """
    if horizon_k <= 0:
        return 1.0

    lines = [l.strip() for l in raw_pred.splitlines() if l.strip() != ""]
    if not lines:
        return 0.0

    valid = sum(1 for l in lines if MOVE_RE.match(l))
    # Normalize by expected #lines rather than produced #lines
    return min(1.0, valid / float(horizon_k))


# =============================
# 2. Relative accuracy (prefix match)
# =============================
def relative_accuracy(pred_moves: List[str], gold_moves: List[str]) -> float:
    K = len(gold_moves)
    if K == 0:
        return 1.0
    match = 0
    for i in range(K):
        if i < len(pred_moves) and pred_moves[i] == gold_moves[i]:
            match += 1
    return match / K


# =============================
# 3. Edit similarity
# =============================
def edit_similarity(pred_moves: List[str], gold_moves: List[str]) -> float:
    """
    Levenshtein similarity between the predicted and gold move sequences,
    treating each move as a token joined by '|'.
    """
    if not gold_moves:
        return 1.0

    pred_str = "|".join(pred_moves)
    gold_str = "|".join(gold_moves)
    d = levenshtein_distance(pred_str, gold_str)
    return 1 - (d / max(1, len(gold_str)))


# =============================
# 4. First error (by move index)
# =============================
def first_error_position(pred_moves: List[str], gold_moves: List[str]) -> int:
    """
    First index (1-based) where prediction differs from gold or is missing.
    Returns K+1 if perfect over the gold horizon.
    """
    K = len(gold_moves)
    for i in range(K):
        if i >= len(pred_moves) or pred_moves[i] != gold_moves[i]:
            return i + 1
    return K + 1


# =============================
# 5. Sliding legality-based: invalid move rate
# =============================
def invalid_move_rate(
    pred_moves: List[str],
    prompt: str,
) -> float:
    """
    Fraction of predicted moves that are illegal when applied from the
    given starting board state in the prompt.
    """
    init_state, board_size = parse_board_from_prompt(prompt)
    if board_size <= 0 or not init_state:
        return 0.0

    total = len(pred_moves)
    if total == 0:
        return 0.0

    state = clone_state(init_state)
    invalid = 0
    for mv in pred_moves:
        if not apply_move_inplace(state, mv, board_size):
            invalid += 1

    return invalid / total


# =============================
# 6. Board divergence
# =============================
def board_divergence(
    pred_moves: List[str],
    gold_moves: List[str],
    prompt: str,
) -> float:
    """
    Compare the final board state after applying the gold horizon vs
    after applying the predicted horizon, starting from the same initial board.

    Metric = fraction of tiles (including blank) that end in a different position.
    """
    init_state, board_size = parse_board_from_prompt(prompt)
    if board_size <= 0 or not init_state:
        return 0.0

    # Simulate gold moves
    gold_state = clone_state(init_state)
    for mv in gold_moves:
        apply_move_inplace(gold_state, mv, board_size)

    # Simulate prediction up to min(len(pred), len(gold))
    pred_state = clone_state(init_state)
    K = min(len(pred_moves), len(gold_moves))
    for i in range(K):
        apply_move_inplace(pred_state, pred_moves[i], board_size)

    if len(gold_state) == 0:
        return 0.0

    mismatched = sum(
        1 for i in range(len(gold_state)) if gold_state[i] != pred_state[i]
    )
    return mismatched / len(gold_state)


# =============================
# 7. Future-Legal Prefix Rate
# =============================
def future_legal_prefix_rate(pred_moves: List[str], prompt: str) -> float:
    """
    For each prefix of length j in [1..N], check whether *all* moves
    in that prefix are legal when applied from the start state.

    Metric = (# fully legal prefixes) / N
    """
    init_state, board_size = parse_board_from_prompt(prompt)
    if board_size <= 0 or not init_state:
        return 0.0

    n = len(pred_moves)
    if n == 0:
        return 1.0  # vacuously no illegal prefixes

    legal_prefixes = 0
    for upto in range(1, n + 1):
        state = clone_state(init_state)
        all_legal = True
        for mv in pred_moves[:upto]:
            if not apply_move_inplace(state, mv, board_size):
                all_legal = False
                break
        if all_legal:
            legal_prefixes += 1

    return legal_prefixes / n


# =============================
# MAIN METRIC SCRIPT
# =============================
def evaluate_task(task_name: str):

    pred_path = Path(f"Results/{task_name}/metrics/predictions.jsonl")
    if not pred_path.exists():
        print(f"Prediction file missing: {pred_path}")
        return

    print(f"\n=== Evaluating {task_name} ===")

    with open(pred_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f]

    agg = {
        # Generic sequence metrics
        "exact_match": [],
        "relative_accuracy": [],
        "syntax_accuracy": [],
        "edit_similarity": [],
        "first_error_position": [],

        # Sliding-puzzle-specific metrics
        "invalid_move_rate": [],
        "board_divergence": [],
        "future_legal_prefix_rate": [],
    }

    for ex in data:
        gold = parse_moves(ex["target"])
        raw_pred = ex.get("prediction_used") or ex.get("prediction_clean", "")
        pred = parse_moves(raw_pred)
        prompt = ex["prompt"]
        horizon_k = ex.get("future_steps", len(gold))

        # Generic metrics
        agg["exact_match"].append(1 if pred == gold and len(pred) == len(gold) else 0)
        agg["relative_accuracy"].append(relative_accuracy(pred, gold))
        agg["syntax_accuracy"].append(syntax_accuracy(raw_pred, horizon_k))
        agg["edit_similarity"].append(edit_similarity(pred, gold))
        agg["first_error_position"].append(first_error_position(pred, gold))

        # Sliding-specific metrics
        agg["invalid_move_rate"].append(invalid_move_rate(pred, prompt))
        agg["board_divergence"].append(board_divergence(pred, gold, prompt))
        agg["future_legal_prefix_rate"].append(
            future_legal_prefix_rate(pred, prompt)
        )

    summary = {
        k: (sum(v) / len(v) if len(v) > 0 else 0.0)
        for k, v in agg.items()
    }

    out_path = pred_path.parent / "detailed_metrics_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("Saved summary:", out_path)
    print(json.dumps(summary, indent=4))


# =============================
# RUN MULTIPLE TASKS
# =============================
TASKS = [
    "sliding_puzzle_Qwen72B",
    "sliding_puzzle_baseline",
    "sliding_puzzle_500_test_500_examples",
    "sliding_puzzle_1500_test_1500_examples",
    "sliding_puzzle_3000_test_3000_examples",
]

for t in TASKS:
    evaluate_task(t)
