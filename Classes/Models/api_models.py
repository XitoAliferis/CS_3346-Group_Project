import os
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from tqdm import tqdm


# -----------------------------
# Shared text utilities
# -----------------------------
def clean_generation(text: str) -> str:
    """
    Clean raw model output before extracting the answer lines.

    - Cut off at our own markers (<SOL>)
    - Cut off at Qwen chat end marker (<|im_end|>)
    - Strip outer whitespace
    """
    for marker in ["<SOL>", "<|im_end|>"]:
        if marker in text:
            text = text.split(marker)[0]
    return text.strip()


def normalize_target_text(target: str) -> str:
    """
    Normalize the gold target text across all tasks.

    - Strip whitespace on each line
    - Drop empty lines
    - Drop chat control lines like <|im_start|>..., <|im_end|>
    - Drop code fence markers like ``` (in case they slip in)
    """
    lines = [l.strip() for l in target.splitlines()]
    clean_lines = []
    for l in lines:
        if not l:
            continue
        if l.startswith("<|im_"):  # <|im_start|>, <|im_end|>, etc.
            continue
        if l.startswith("```"):  # code fences
            continue
        clean_lines.append(l)
    return "\n".join(clean_lines)


def get_horizon_k(target: str) -> int:
    """
    How many *answer* lines we expect (ignoring <|im_end|>, etc.).
    Works for Fibonacci (integers) and both move-based games.
    """
    norm = normalize_target_text(target)
    if not norm.strip():
        return 0
    return len(norm.splitlines())


def extract_answer_lines(text: str, horizon_k: int) -> Optional[str]:
    """
    Generic extraction of the model's answer for *any* of the three games.

    Strategy:
    - Clean with clean_generation()
    - Split into lines, strip whitespace
    - Drop empty lines, chat tags, code fences
    - Take the first horizon_k surviving lines.
      If there are fewer than horizon_k lines → return None.
    """
    if horizon_k == 0:
        return ""

    text = clean_generation(text)
    lines = [l.strip() for l in text.splitlines()]
    clean_lines = []

    for l in lines:
        if not l:
            continue
        if l.startswith("<|im_"):  # Qwen chat markers
            continue
        if l.startswith("```"):  # code fences
            continue
        clean_lines.append(l)
        if len(clean_lines) == horizon_k:
            break

    if len(clean_lines) < horizon_k:
        return None

    return "\n".join(clean_lines)


# -----------------------------
# API model config
# -----------------------------
@dataclass
class ApiModelConfig:
    """
    Generic config for any OpenRouter model.

    Example:
        ApiModelConfig(
            model_name="openai/gpt-oss-120b:free"
        )
    """
    model_name: str  # e.g. "openai/gpt-oss-120b:free"

    # generation settings
    temperature: float = 0.0
    max_new_tokens: int = 256

    # OpenRouter API settings
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"

    # Optional, but recommended by OpenRouter
    referer: Optional[str] = None       # e.g. your site URL
    app_name: str = "nqueens-eval"      # shows in OpenRouter dashboard


# -----------------------------
# API-only model (inference + eval)
# -----------------------------
class ApiModel:
    """
    Lightweight model wrapper that talks to OpenRouter instead of HuggingFace.

    - No training, no tokenizer, no torch
    - Just: dataset → prompts → OpenRouter → predictions.jsonl + summary.json
    """

    def __init__(self, cfg: ApiModelConfig):
        self.cfg = cfg
        self.api_key = os.getenv(cfg.api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"{cfg.api_key_env} is not set. "
                "Export your OpenRouter API key, e.g.\n"
                "  export OPENROUTER_API_KEY='sk-...'"
            )

    # keep same directory structure style as your BaseModel
    def task_dir(self, task_name: str) -> str:
        return f"./Results/{task_name}"

    # -------------------------
    # Low-level OpenRouter call
    # -------------------------
    def _call_openrouter(self, prompt: str) -> str:
        """
        Send a single prompt to OpenRouter and return the assistant text.
        Uses Chat Completions API with a single user message.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Optional extra headers OpenRouter recommends
        if self.cfg.referer:
            headers["HTTP-Referer"] = self.cfg.referer
        if self.cfg.app_name:
            headers["X-Title"] = self.cfg.app_name

        payload: Dict[str, Any] = {
            "model": self.cfg.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_new_tokens,
        }

        url = f"{self.cfg.base_url}/chat/completions"
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected OpenRouter response format: {data}") from e

    # -------------------------
    # Evaluation / prediction loop
    # -------------------------
    def save_predictions(self, test_ds, save_name: str = "api_eval") -> Dict[str, Any]:
        """
        Generates per-example predictions via OpenRouter and saves:
        1) predictions file (JSONL)
        2) metrics summary file (JSON)

        Assumes each row in test_ds has:
            - row["prompt"] : full chat-style prompt string
            - row["target"] : gold answer string (possibly multi-line)
        """
        out_dir = f"{self.task_dir(save_name)}/metrics"
        os.makedirs(out_dir, exist_ok=True)

        pred_path = f"{out_dir}/predictions.jsonl"
        summary_path = f"{out_dir}/summary.json"

        results = []
        correct_count = 0

        print(f"[ApiModel] Using OpenRouter model: {self.cfg.model_name}")
        print(f"[ApiModel] Generating predictions → {pred_path}")

        for row in tqdm(test_ds, desc="Generating outputs via OpenRouter"):
            prompt = row["prompt"]
            target = row["target"]

            # Match your baseline behavior: insert <SOL> just before assistant
            # <|im_start|>assistant\n<SOL>
            gen_prompt = prompt.replace(
                "<|im_start|>assistant",
                "<|im_start|>assistant\n<SOL>"
            )

            gold_norm = normalize_target_text(target)
            horizon_k = get_horizon_k(target)

            # ---- 1) Call OpenRouter ----
            raw = self._call_openrouter(gen_prompt)
            pred_only = clean_generation(raw)

            # ---- 2) Evaluate the first K lines ----
            pred_lines = extract_answer_lines(pred_only, horizon_k)
            if pred_lines is None:
                is_correct = 0
            else:
                pred_norm = normalize_target_text(pred_lines)
                is_correct = int(pred_norm == gold_norm)

            correct_count += is_correct

            # raw == model's output, pred_only == cleaned version
            results.append(
                {
                    "prompt": prompt,
                    "target": target,
                    "prediction_raw": raw,
                    "prediction_clean": pred_only,
                    "prediction_used": pred_lines,
                    "correct": is_correct,
                }
            )

        # ---- 3) Save prediction file ----
        with open(pred_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # ---- 4) Save summary ----
        accuracy = correct_count / max(1, len(results))
        summary = {
            "accuracy": accuracy,
            "total_examples": len(results),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        print(f"[ApiModel] Saved metrics → {summary_path}")
        if results:
            print("RAW MODEL OUTPUT (last example):\n", results[-1]["prediction_raw"])

        return summary

    # convenience alias to mirror your HF models’ API
    def evaluate_api_model(self, test_ds, save_name: str = "api_eval") -> Dict[str, Any]:
        return self.save_predictions(test_ds, save_name=save_name)
