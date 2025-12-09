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
# Utility: Parse visible prefix from prompt
# =============================
def parse_visible_prefix_from_prompt(prompt: str) -> List[int]:
    """
    From the full prompt (which may contain few-shot demos),
    extract the *last* visible prefix line that follows
    'Sequence so far (from left to right):'.

    Returns a list of ints, or [] on failure.
    """
    lines = prompt.splitlines()
    prefix_line = None

    for i, line in enumerate(lines):
        if "Sequence so far (from left to right):" in line:
            # The next non-empty line after this one is the prefix
            for j in range(i + 1, len(lines)):
                candidate = lines[j].strip()
                if candidate == "":
                    continue
                prefix_line = candidate
                break

    if prefix_line is None:
        return []

    parts = prefix_line.split()
    nums = []
    for p in parts:
        if INT_RE.match(p):
            nums.append(int(p))
    return nums


# =============================
# 1. Syntax validity
# =============================
def syntax_accuracy(raw_pred: str, horizon_k: int) -> float:
    """
    Fraction of non-blank output lines that are valid integers,
    normalized by the *expected* number of lines (horizon_k).

    This penalizes models that output fewer/more lines than requested.
    """
    lines = [l.strip() for l in raw_pred.splitlines() if l.strip() != ""]
    if horizon_k <= 0:
        return 1.0
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
    Levenshtein similarity between the sequences, treating each number
    as a token and joining via '|'.
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
# 5. Mean Absolute Error (MAE)
# =============================
def mean_absolute_error(pred_vals: List[int], gold_vals: List[int]) -> float:
    """
    Mean absolute error over the horizon K.
    Missing predictions are treated as 0 for that position.
    """
    K = len(gold_vals)
    if K == 0:
        return 0.0

    total_err = 0.0
    for i in range(K):
        gold = gold_vals[i]
        pred = pred_vals[i] if i < len(pred_vals) else 0
        total_err += abs(pred - gold)
    return total_err / K


# =============================
# 6. Recurrence violation rate
# =============================
def recurrence_violation_rate(
    prefix_vals: List[int],
    pred_vals: List[int],
) -> float:
    """
    For the combined sequence [prefix ... prefix_last, pred ...],
    measure how often the Fibonacci recurrence is violated for positions
    that involve at least one *predicted* term.

    Recurrence: x[i] == x[i-1] + x[i-2]
    """
    if len(prefix_vals) < 2:
        # not enough context to define recurrence at boundary
        return 0.0

    full_seq = prefix_vals + pred_vals
    prefix_len = len(prefix_vals)

    total_checks = 0
    violations = 0

    # Index i is 0-based; recurrence starts at i >= 2
    for i in range(2, len(full_seq)):
        # Only consider constraints where x[i] is in the predicted part
        if i < prefix_len:
            continue

        total_checks += 1
        if full_seq[i] != full_seq[i - 1] + full_seq[i - 2]:
            violations += 1

    if total_checks == 0:
        return 0.0
    return violations / total_checks


# =============================
# 7. Recurrence-stable prefix rate
# =============================
def recurrence_prefix_stability(
    prefix_vals: List[int],
    pred_vals: List[int],
) -> float:
    """
    For each prefix of length j in [1..N] of the predicted horizon,
    check if the *combined* sequence (prefix + pred[:j]) satisfies
    the Fibonacci recurrence everywhere it is defined and involves
    any predicted term.

    Metric = (# such legal prefixes) / N
    """
    N = len(pred_vals)
    if N == 0:
        return 1.0

    if len(prefix_vals) < 2:
        # cannot apply recurrence meaningfully; treat as neutral
        return 1.0

    legal_prefixes = 0

    for j in range(1, N + 1):
        full_seq = prefix_vals + pred_vals[:j]
        prefix_len = len(prefix_vals)

        all_ok = True
        for i in range(2, len(full_seq)):
            if i < prefix_len:
                continue  # still in pure prefix region
            if full_seq[i] != full_seq[i - 1] + full_seq[i - 2]:
                all_ok = False
                break

        if all_ok:
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
        "mean_absolute_error": [],

        # Fibonacci-specific structural metrics
        "recurrence_violation_rate": [],
        "recurrence_prefix_stability": [],
    }

    for ex in data:
        raw_pred = ex.get("prediction_used") or ex.get("prediction_clean", "")
        gold = parse_int_sequence(ex["target"])
        pred = parse_int_sequence(raw_pred)
        prompt = ex["prompt"]
        horizon_k = ex.get("future_steps", len(gold))

        prefix_vals = parse_visible_prefix_from_prompt(prompt)

        # Generic metrics
        agg["exact_match"].append(1 if pred == gold and len(pred) == len(gold) else 0)
        agg["relative_accuracy"].append(relative_accuracy(pred, gold))
        agg["syntax_accuracy"].append(syntax_accuracy(raw_pred, horizon_k))
        agg["edit_similarity"].append(edit_similarity(pred, gold))
        agg["first_error_position"].append(first_error_position(pred, gold))
        agg["mean_absolute_error"].append(mean_absolute_error(pred, gold))

        # Fibonacci-structural metrics
        agg["recurrence_violation_rate"].append(
            recurrence_violation_rate(prefix_vals, pred)
        )
        agg["recurrence_prefix_stability"].append(
            recurrence_prefix_stability(prefix_vals, pred)
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
    "fibonacci_baseline",
    "fibonacci_500_test_500_examples",
    "fibonacci_1500_test_1500_examples",
    "fibonacci_3000_test_3000_examples",
]

for t in TASKS:
    evaluate_task(t)
