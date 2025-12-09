import json
import re
from typing import List, Dict
from pathlib import Path
from Levenshtein import distance as levenshtein_distance   # pip install python-Levenshtein

# =============================
# Utility: Parse integer outputs
# =============================
INT_RE = re.compile(r"^-?\d+$")


def parse_int_sequence(block: str) -> List[int]:
    """
    Extract only well-formed integer lines from a block of text.
    """
    vals: List[int] = []
    for l in block.splitlines():
        s = l.strip()
        if INT_RE.match(s):
            vals.append(int(s))
    return vals


# =============================
# Utility: Parse prefix rows from prompt
# =============================
def parse_prefix_rows_from_prompt(prompt: str) -> List[int]:
    """
    From the full prompt (with possible few-shot demos),
    extract the *last* 'rows: ...' line, which is the partial placement
    of the *new instance*.

    Example line:
        rows: 2 5 3 0 7 4 6
        rows: (none)
    """
    lines = prompt.splitlines()
    last_rows_line = None

    for line in lines:
        line = line.strip()
        if line.startswith("rows:"):
            last_rows_line = line

    if last_rows_line is None:
        return []

    # Remove "rows:" prefix
    content = last_rows_line[len("rows:"):].strip()
    if not content or content.startswith("(none)"):
        return []

    rows: List[int] = []
    for tok in content.split():
        if INT_RE.match(tok):
            rows.append(int(tok))
    return rows


# =============================
# 1. Syntax validity
# =============================
def syntax_accuracy(raw_pred: str, horizon_k: int) -> float:
    """
    Fraction of non-blank output lines that are valid integers,
    normalized by the *expected* number of lines (horizon_k).

    This penalizes models that output fewer/more lines than requested.
    """
    if horizon_k <= 0:
        return 1.0

    lines = [l.strip() for l in raw_pred.splitlines() if l.strip() != ""]
    if not lines:
        return 0.0

    valid = sum(1 for l in lines if INT_RE.match(l))
    return min(1.0, valid / float(horizon_k))


# =============================
# 2. Relative accuracy (prefix match)
# =============================
def relative_accuracy(pred_vals: List[int], gold_vals: List[int]) -> float:
    K = len(gold_vals)
    if K == 0:
        return 1.0
    match = 0
    for i in range(K):
        if i < len(pred_vals) and pred_vals[i] == gold_vals[i]:
            match += 1
    return match / K


# =============================
# 3. Edit similarity
# =============================
def edit_similarity(pred_vals: List[int], gold_vals: List[int]) -> float:
    """
    Levenshtein similarity between predicted and gold sequences,
    treating each row index as a token and joining via '|'.
    """
    if not gold_vals:
        return 1.0

    pred_str = "|".join(str(x) for x in pred_vals)
    gold_str = "|".join(str(x) for x in gold_vals)
    d = levenshtein_distance(pred_str, gold_str)
    return 1 - (d / max(1, len(gold_str)))


# =============================
# 4. First error (by index)
# =============================
def first_error_position(pred_vals: List[int], gold_vals: List[int]) -> int:
    """
    First index (1-based) where prediction differs from gold,
    or where prediction is missing. Returns K+1 if perfect.
    """
    K = len(gold_vals)
    for i in range(K):
        if i >= len(pred_vals) or pred_vals[i] != gold_vals[i]:
            return i + 1
    return K + 1


# =============================
# 5. Row range validity
# =============================
def row_range_validity(pred_vals: List[int], n: int) -> float:
    """
    Fraction of predicted row indices that lie in [0, n-1].
    """
    if n <= 0:
        return 0.0
    if not pred_vals:
        return 0.0

    valid = sum(1 for r in pred_vals if 0 <= r < n)
    return valid / len(pred_vals)


# =============================
# N-Queens conflict helpers
# =============================
def count_conflicts(positions: List[int]) -> int:
    """
    Count the number of conflicting queen pairs in a partial assignment.

    positions[c] = row index of queen in column c, for c in [0..m-1].
    Conflicts:
      - same row
      - same diagonal (abs(r_i - r_j) == |j - i|)
    """
    m = len(positions)
    conflicts = 0
    for i in range(m):
        for j in range(i + 1, m):
            if positions[i] == positions[j]:
                conflicts += 1
            elif abs(positions[i] - positions[j]) == (j - i):
                conflicts += 1
    return conflicts


def total_pairs(m: int) -> int:
    return m * (m - 1) // 2


# =============================
# 6. Conflict pair rate (full partial board)
# =============================
def conflict_pair_rate(prefix_rows: List[int], pred_rows: List[int]) -> float:
    """
    Conflict density over all placed queens (prefix + predictions).

    Metric = (# conflicting pairs) / (total # possible pairs) in the
    combined partial board. If fewer than 2 queens, returns 0.0.
    """
    full = prefix_rows + pred_rows
    m = len(full)
    if m <= 1:
        return 0.0

    conf = count_conflicts(full)
    pairs = total_pairs(m)
    if pairs == 0:
        return 0.0
    return conf / pairs


# =============================
# 7. Conflict-free prefix rate
# =============================
def conflict_free_prefix_rate(prefix_rows: List[int], pred_rows: List[int]) -> float:
    """
    For each prefix of length j in [1..N] of the predicted horizon,
    check whether the combined partial board
        (prefix_rows + pred_rows[:j])
    has *no* queen conflicts.

    Metric = (# conflict-free prefixes) / N.
    """
    N = len(pred_rows)
    if N == 0:
        return 1.0  # vacuously all prefixes are fine

    legal_prefixes = 0
    for j in range(1, N + 1):
        full = prefix_rows + pred_rows[:j]
        if count_conflicts(full) == 0:
            legal_prefixes += 1

    return legal_prefixes / N


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

        # N-Queens-specific metrics
        "row_range_validity": [],
        "conflict_pair_rate": [],
        "conflict_free_prefix_rate": [],
    }

    for ex in data:
        raw_pred = ex.get("prediction_used") or ex.get("prediction_clean", "")
        gold = parse_int_sequence(ex["target"])
        pred = parse_int_sequence(raw_pred)
        prompt = ex["prompt"]
        n = ex.get("n", 0)
        horizon_k = ex.get("future_steps", len(gold))

        prefix_rows = parse_prefix_rows_from_prompt(prompt)

        # Generic metrics
        agg["exact_match"].append(1 if pred == gold and len(pred) == len(gold) else 0)
        agg["relative_accuracy"].append(relative_accuracy(pred, gold))
        agg["syntax_accuracy"].append(syntax_accuracy(raw_pred, horizon_k))
        agg["edit_similarity"].append(edit_similarity(pred, gold))
        agg["first_error_position"].append(first_error_position(pred, gold))

        # N-Queens-specific metrics
        agg["row_range_validity"].append(row_range_validity(pred, n))
        agg["conflict_pair_rate"].append(conflict_pair_rate(prefix_rows, pred))
        agg["conflict_free_prefix_rate"].append(
            conflict_free_prefix_rate(prefix_rows, pred)
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
    "nqueens_Qwen72B",
    "nqueens_baseline",
    "nqueens_500_test_500_examples",
    "nqueens_1500_test_1500_examples",
    "nqueens_3000_test_3000_examples",
]

for t in TASKS:
    evaluate_task(t)
