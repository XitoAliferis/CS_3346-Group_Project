import json
import re
from typing import List, Dict
from pathlib import Path
from Levenshtein import distance as levenshtein_distance   # pip install python-Levenshtein

# =============================
# Utility: Parse move strings
# =============================
MOVE_RE = re.compile(r"^[A-C]->[A-C]$")

def parse_moves(block: str) -> List[str]:
    """Extract only well-formed moves like 'A->B' from a block of text."""
    moves = []
    for l in block.splitlines():
        l = l.strip()
        if MOVE_RE.match(l):  # <-- only accept real moves
            moves.append(l)
    return moves

# =============================
# Utility: Tower state helpers
# =============================

def parse_config_from_prompt(prompt: str) -> Dict[str, list]:
    """
    Parse starting Tower-of-Hanoi configuration from the prompt.

    Expected lines like:
        Peg A: 3 2 1
        Peg B: empty
        Peg C: empty
    """
    state = {"A": [], "B": [], "C": []}
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith("Peg "):
            peg = line[4]
            nums = line.split(":", 1)[1].strip()
            if nums.lower() == "empty":
                state[peg] = []
            else:
                state[peg] = [int(x) for x in nums.split()]
    return state


def clone_state(state: Dict[str, list]) -> Dict[str, list]:
    return {peg: stack.copy() for peg, stack in state.items()}


def apply_move(state: Dict[str, list], move: str) -> bool:
    """
    Apply a move like 'A->B' to the state.
    Returns True if legal & applied, False if illegal (state unchanged).
    """
    if not MOVE_RE.match(move):
        return False
    
    src, dst = move.split("->")
    src = src.strip()
    dst = dst.strip()

    # must have at least one disk to move
    if not state[src]:
        return False

    disk = state[src][-1]

    # cannot place larger disk on smaller one
    if state[dst] and state[dst][-1] < disk:
        return False

    state[src].pop()
    state[dst].append(disk)
    return True


def disk_positions(state: Dict[str, list]) -> Dict[int, str]:
    """Map disk -> peg from a tower state."""
    pos = {}
    for peg, stack in state.items():
        for disk in stack:
            pos[disk] = peg
    return pos


# =============================
# 1. Syntax validity
# =============================
def syntax_accuracy(pred_moves: List[str]) -> float:
    if not pred_moves:
        return 0.0
    valid = sum(1 for m in pred_moves if MOVE_RE.match(m))
    return valid / len(pred_moves)


# =============================
# 2. Relative accuracy (prefix match)
# =============================
def relative_accuracy(pred_moves: List[str], gold_moves: List[str]) -> float:
    K = len(gold_moves)
    if K == 0:
        return 1.0
    pred = pred_moves[:K]
    match = sum(1 for i in range(K) if i < len(pred) and pred[i] == gold_moves[i])
    return match / K


# =============================
# 3. Edit similarity
# =============================
def edit_similarity(pred_moves: List[str], gold_moves: List[str]) -> float:
    K = len(gold_moves)
    if K == 0:
        return 1.0
    pred_str = "|".join(pred_moves)
    gold_str = "|".join(gold_moves)
    d = levenshtein_distance(pred_str, gold_str)
    return 1 - (d / max(1, len(gold_str)))


# =============================
# 4. First error (by move index)
# =============================
def first_error_position(pred_moves: List[str], gold_moves: List[str]) -> int:
    K = len(gold_moves)
    for i in range(K):
        if i >= len(pred_moves) or pred_moves[i] != gold_moves[i]:
            return i + 1  # 1-indexed
    return K + 1  # perfect


# =============================
# 5. Hanoi legality-based metrics
# =============================
def invalid_move_rate(pred_moves: List[str], prompt: str) -> float:
    try:
        state = parse_config_from_prompt(prompt)
    except Exception:
        return 0.0
    
    total = len(pred_moves)
    if total == 0:
        return 0.0

    invalid = 0
    for mv in pred_moves:
        if not apply_move(state, mv):
            invalid += 1
    return invalid / total


# =============================
# NEW METRIC 1: Local Optimality Deviation
# =============================
def local_optimality_deviation(pred_moves: List[str], gold_moves: List[str]) -> float:
    """
    Fraction of positions in the horizon K where the prediction deviates
    from the optimal (gold) sequence.
    """
    K = len(gold_moves)
    if K == 0:
        return 0.0
    pred = pred_moves[:K]
    mismatches = sum(1 for i in range(K) if i >= len(pred) or pred[i] != gold_moves[i])
    return mismatches / K  # in [0,1]


# =============================
# NEW METRIC 2: Stability Error (oscillations)
# =============================
def stability_error(pred_moves: List[str]) -> float:
    """
    How often does the model just undo moves? (A->B followed by B->A, etc.)
    Computed over consecutive pairs of VALID moves only.
    """
    valid = [m for m in pred_moves if MOVE_RE.match(m)]
    if len(valid) <= 1:
        return 0.0

    backtracks = 0
    for i in range(1, len(valid)):
        src_prev, dst_prev = [x.strip() for x in valid[i - 1].split("->")]
        src_cur, dst_cur = [x.strip() for x in valid[i].split("->")]
        if src_prev == dst_cur and dst_prev == src_cur:
            backtracks += 1

    return backtracks / (len(valid) - 1)


# =============================
# NEW METRIC 3: Peg-Load Divergence
# =============================
def peg_load_divergence(pred_moves: List[str], gold_moves: List[str], prompt: str) -> float:
    """
    Compare the final tower configuration after the gold horizon vs
    after the predicted horizon, starting from the same initial state.

    Metric = fraction of disks that end on a *different peg* than in the gold state.
    """
    try:
        init_state = parse_config_from_prompt(prompt)
    except Exception:
        return 0.0

    total_disks = sum(len(v) for v in init_state.values())
    if total_disks == 0:
        return 0.0

    # Simulate gold sequence fully
    gold_state = clone_state(init_state)
    for mv in gold_moves:
        apply_move(gold_state, mv)

    # Simulate prediction up to K (or its own length)
    pred_state = clone_state(init_state)
    K = min(len(pred_moves), len(gold_moves))
    for i in range(K):
        apply_move(pred_state, pred_moves[i])

    gold_pos = disk_positions(gold_state)
    pred_pos = disk_positions(pred_state)

    mismatched = sum(1 for d in gold_pos if pred_pos.get(d) != gold_pos[d])
    return mismatched / total_disks


# =============================
# NEW METRIC 4: Future-Legal Prefix Rate
# =============================
def future_legal_prefix_rate(pred_moves: List[str], prompt: str) -> float:
    """
    For each prefix of length j in [1..N], check whether *all* moves
    in that prefix are legal when applied from the start state.

    Metric = (# fully legal prefixes) / N
    """
    try:
        init_state = parse_config_from_prompt(prompt)
    except Exception:
        return 0.0

    n = len(pred_moves)
    if n == 0:
        return 1.0  # vacuously no illegal prefixes

    legal_prefixes = 0
    for upto in range(1, n + 1):
        state = clone_state(init_state)
        all_legal = True
        for mv in pred_moves[:upto]:
            if not apply_move(state, mv):
                all_legal = False
                break
        if all_legal:
            legal_prefixes += 1

    return legal_prefixes / n


# =============================
# NEW METRIC 5: First Critical Divergence
# =============================
def first_critical_divergence(pred_moves: List[str], gold_moves: List[str], prompt: str) -> int:
    """
    First step where the *tower state* under predicted moves differs
    from the state under gold moves, starting from the same initial config.
    Returns 1-indexed step; K+1 if never diverges within gold horizon.
    """
    try:
        init_state = parse_config_from_prompt(prompt)
    except Exception:
        # fall back: behave like first_error_position
        return first_error_position(pred_moves, gold_moves)

    gold_state = clone_state(init_state)
    pred_state = clone_state(init_state)

    K = len(gold_moves)
    max_len = max(len(pred_moves), K)

    for i in range(max_len):
        if i < K:
            apply_move(gold_state, gold_moves[i])
        if i < len(pred_moves):
            apply_move(pred_state, pred_moves[i])

        if gold_state != pred_state:
            return i + 1  # first divergent state

    return K + 1  # never diverged within horizon


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
        # Existing metrics
        "exact_match": [],
        "relative_accuracy": [],
        "syntax_accuracy": [],
        "invalid_move_rate": [],
        "edit_similarity": [],
        "first_error_position": [],

        # New metrics
        "local_optimality_deviation": [],
        "stability_error": [],
        "peg_load_divergence": [],
        "future_legal_prefix_rate": [],
        "first_critical_divergence": [],
    }

    for ex in data:
        gold = parse_moves(ex["target"])
        pred = parse_moves(ex.get("prediction_used") or ex.get("prediction_clean", ""))
        prompt = ex["prompt"]

        # Existing metrics
        agg["exact_match"].append(1 if pred == gold else 0)
        agg["relative_accuracy"].append(relative_accuracy(pred, gold))
        agg["syntax_accuracy"].append(syntax_accuracy(pred))
        agg["invalid_move_rate"].append(invalid_move_rate(pred, prompt))
        agg["edit_similarity"].append(edit_similarity(pred, gold))
        agg["first_error_position"].append(first_error_position(pred, gold))

        # New metrics
        agg["local_optimality_deviation"].append(local_optimality_deviation(pred, gold))
        agg["stability_error"].append(stability_error(pred))
        agg["peg_load_divergence"].append(peg_load_divergence(pred, gold, prompt))
        agg["future_legal_prefix_rate"].append(future_legal_prefix_rate(pred, prompt))
        agg["first_critical_divergence"].append(first_critical_divergence(pred, gold, prompt))

    summary = {k: (sum(v) / len(v) if len(v) > 0 else 0.0) for k, v in agg.items()}

    out_path = pred_path.parent / "detailed_metrics_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("Saved summary:", out_path)
    print(json.dumps(summary, indent=4))

# =============================
# RUN MULTIPLE TASKS
# =============================
TASKS = [
    "hanoi_baseline",
    "hanoi_500_test_500_examples",
    "hanoi_1500_test_1500_examples",
    "hanoi_3000_test_3000_examples",
]

for t in TASKS:
    evaluate_task(t)
