import os
import json
import random
import re

import evaluate
import numpy as np
import optuna
import torch
from dataclasses import dataclass
from typing import Any, Dict

from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

accuracy_metric = evaluate.load("accuracy")


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


def extract_answer_lines(text: str, horizon_k: int) -> str | None:
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


# base config
@dataclass
class BaseModelConfig:
    model_name: str = ""

    # train settings
    lr: float = 2e-5
    batch_size: int = 4
    num_epochs: int = 10
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    lora_rank: int = 16
    grad_acc_steps: int = 4
    use_lora: bool = True


# base model class
class BaseModel:
    def __init__(self, cfg: BaseModelConfig):
        self.cfg = cfg

    # file path for best params
    @property
    def best_param_path(self) -> str:
        return f"{self.task_dir(self.cfg.model_name.replace('/', '_'))}/best_params.json"


    def task_dir(self, task_name: str) -> str:
        return f"./Results/{task_name}"
    # ---------------------------------------------------------
    # 0) 5-fold cross-validation splitter
    # ---------------------------------------------------------
    def create_folds(self, dataset, n_folds: int = 5, seed: int = 0):
        """
        Splits a single dataset into n_folds train/val splits.
        Assumes `dataset` is a HuggingFace Dataset with `len()` and `.select`.

        Returns:
            train_folds: list of length n_folds
            val_folds:   list of length n_folds
        """
        n = len(dataset)
        indices = list(range(n))
        rng = random.Random(seed)
        rng.shuffle(indices)

        # Special-case: n_folds <= 1 → simple 80/20 train/val split
        if n_folds <= 1:
            if n < 2:
                # Degenerate case: all data used as train, empty val
                return [dataset], [dataset.select([])]

            split = int(0.8 * n)
            train_idx = indices[:split]
            val_idx = indices[split:]
            train_folds = [dataset.select(train_idx)]
            val_folds = [dataset.select(val_idx)]
            return train_folds, val_folds

        # Standard K-fold logic for n_folds >= 2
        fold_sizes = [n // n_folds] * n_folds
        for i in range(n % n_folds):
            fold_sizes[i] += 1

        folds = []
        start = 0
        for size in fold_sizes:
            end = start + size
            folds.append(indices[start:end])
            start = end

        train_folds = []
        val_folds = []

        for k in range(n_folds):
            val_idx = folds[k]
            train_idx = [i for j, fold in enumerate(folds) if j != k for i in fold]

            train_folds.append(dataset.select(train_idx))
            val_folds.append(dataset.select(val_idx))

        for i in range(n_folds):
            print(
                f"[FOLDS] fold {i}: "
                f"train={len(train_folds[i])}, val={len(val_folds[i])}"
            )

        return train_folds, val_folds

    # ---------------------------------------------------------
    # 1) load base model no lora
    # ---------------------------------------------------------
    def load_base_model(self):
        print(f"[BaseModel] Loading base model ONLY: {self.cfg.model_name}")

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        special_tokens = {"additional_special_tokens": ["<SOL>"]}
        tokenizer.add_special_tokens(special_tokens)

        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer

    # ---------------------------------------------------------
    # 2) load train model with lora
    # ---------------------------------------------------------
    def load_train_model(self):
        print(f"[BaseModel] Loading training model: {self.cfg.model_name}")

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        special_tokens = {"additional_special_tokens": ["<SOL>"]}
        tokenizer.add_special_tokens(special_tokens)

        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )

        model.resize_token_embeddings(len(tokenizer))

        if self.cfg.use_lora:
            print("[BaseModel] Applying LoRA adapters")
            lora_cfg = LoraConfig(
                r=self.cfg.lora_rank,
                lora_alpha=2 * self.cfg.lora_rank,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

        return model, tokenizer

    # ---------------------------------------------------------
    # 3) tokenize data
    # ---------------------------------------------------------
    def tokenize_dataset(self, dataset, tokenizer):
        def _tok(row):
            prompt = row["prompt"]
            target = row["target"]

            # encode prompt separately
            prompt_enc = tokenizer(
                prompt,
                add_special_tokens=False,
            )

            # <SOL> delimiter as its own chunk
            sol_enc = tokenizer("<SOL>\n", add_special_tokens=False)
            target_enc = tokenizer(target, add_special_tokens=False)

            eos = [tokenizer.eos_token_id]

            # full input: PROMPT + <SOL>\n + TARGET + EOS
            input_ids = (
                prompt_enc["input_ids"]
                + sol_enc["input_ids"]
                + target_enc["input_ids"]
                + eos
            )
            attention_mask = [1] * len(input_ids)

            # labels: ignore PROMPT + <SOL>\n, train only on TARGET + EOS
            num_ignored = len(prompt_enc["input_ids"]) + len(sol_enc["input_ids"])
            labels = [-100] * num_ignored + target_enc["input_ids"] + eos

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        tokenized = dataset.map(_tok, remove_columns=dataset.column_names)
        return tokenized

    # ---------------------------------------------------------
    # 4) build train args
    # ---------------------------------------------------------
    def build_training_args(self, save_name: str, train_size=None):
        return TrainingArguments(
            output_dir=f"{self.task_dir(save_name)}/training",
            per_device_train_batch_size=self.cfg.batch_size,
            gradient_accumulation_steps=self.cfg.grad_acc_steps,
            learning_rate=self.cfg.lr,
            num_train_epochs=self.cfg.num_epochs,
            weight_decay=self.cfg.weight_decay,
            warmup_ratio=self.cfg.warmup_ratio,
            bf16=True,

            max_steps=-1,
            logging_strategy="epoch",
            eval_strategy="epoch",

            prediction_loss_only=True,
            save_steps=999999,
            report_to="none",
            remove_unused_columns=False,
        )



    # ---------------------------------------------------------
    # 5) run one train
    # ---------------------------------------------------------
    def train(self, train_ds, val_ds, save_name: str):
        print(f"[TRAIN] save_name={save_name}")
        print(f"[TRAIN] raw train size: {len(train_ds)}, val size: {len(val_ds)}")

        model, tokenizer = self.load_train_model()

        train_tok = self.tokenize_dataset(train_ds, tokenizer)
        val_tok = self.tokenize_dataset(val_ds, tokenizer)

        print(
            f"[TRAIN] tokenized train size: {len(train_tok)}, "
            f"val size: {len(val_tok)}"
        )
        print(f"[TRAIN] tokenized train columns: {train_tok.column_names}")

        collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        args = self.build_training_args(save_name, train_size=len(train_tok))


        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            data_collator=collator,
            compute_metrics=None
        )

        trainer.train()
        trainer.save_model(f"{self.task_dir(save_name)}/final_tuned_model")

        # show trainable params AFTER training (they should be same modules as before,
        # but altered weights)
        if hasattr(model, "print_trainable_parameters"):
            print("[TRAIN] Trainable parameters after training:")
            model.print_trainable_parameters()

        return trainer, tokenizer

    # ---------------------------------------------------------
    # 6) generic EVAL / TEST function
    # ---------------------------------------------------------
    def evaluate_model(self, model, tokenizer, test_ds, save_name: str = "test_eval"):
        """
        Evaluates a given model+tokenizer on a test dataset.
        Returns HF metrics dict (e.g. eval_loss).
        """
        test_tok = self.tokenize_dataset(test_ds, tokenizer)

        collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        args = TrainingArguments(
            output_dir=f"{self.task_dir(save_name)}/eval",
            per_device_eval_batch_size=self.cfg.batch_size,
            do_eval=True,
            report_to="none",
            prediction_loss_only=True,
            eval_strategy="no",     # do not evaluate repeatedly
        )

        trainer = Trainer(
            model=model,
            args=args,
            eval_dataset=test_tok,
            data_collator=collator,

            # compute_metrics MUST be removed or it will force logits too
            compute_metrics=None,
        )

        metrics = trainer.evaluate()
        return metrics

    def evaluate_with_predictions(self, model, tokenizer, test_ds, save_name: str, baseline: bool):
        """Runs HF evaluate + saves prediction file + summary file."""
        hf_metrics = self.evaluate_model(model, tokenizer, test_ds, save_name)
        prediction_metrics = self.save_predictions(
            model, tokenizer, test_ds, save_name, baseline
        )

        return {**hf_metrics, **prediction_metrics}

    def evaluate_base_model(self, test_ds, save_name: str = "base_eval"):
        model, tokenizer = self.load_base_model()
        return self.evaluate_with_predictions(model, tokenizer, test_ds, save_name, baseline=True)

    def evaluate_tuned_model(self, model, tokenizer, test_ds, save_name="tuned_eval"):
        return self.evaluate_with_predictions(model, tokenizer, test_ds, save_name, baseline=False)

    # ---------------------------------------------------------
    # 7) optuna run for 5 folds
    # ---------------------------------------------------------
    def objective(self, trial, train_folds, val_folds):
        tuned_cfg = self._sample_cfg_from_trial(trial)
        model_wrapper = self.__class__(tuned_cfg)

        losses = []

        # Use however many folds we actually have (1 if n_folds=1)
        num_folds = len(train_folds)

        for i in range(num_folds):
            # train() now returns (trainer, tokenizer)
            trainer, _ = model_wrapper.train(
                train_ds=train_folds[i],
                val_ds=val_folds[i],
                save_name=f"trial_{trial.number}_fold_{i}",
            )

            # trainer is a Trainer object → this now works
            metrics = trainer.evaluate()
            losses.append(metrics["eval_loss"])

        return sum(losses) / len(losses)

    # sample config from trial
    def _sample_cfg_from_trial(self, trial) -> BaseModelConfig:
        return BaseModelConfig(
            model_name=self.cfg.model_name,
            lr=trial.suggest_float("lr", 1e-6, 5e-4, log=True),
            batch_size=trial.suggest_categorical("batch_size", [2, 4, 8]),
            num_epochs=trial.suggest_int("num_epochs", 1, 5),
            warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.4),
            weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),
            lora_rank=trial.suggest_categorical("lora_rank", [4, 8, 16, 32]),
            use_lora=True,
        )

    # run optuna
    def run_optuna_search(self, train_folds, val_folds, n_trials: int = 30):
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self.objective(trial, train_folds, val_folds),
            n_trials=n_trials,
        )
        return study

    # save best params
    def save_best_params(self, study):
        params = study.best_params
        os.makedirs(os.path.dirname(self.best_param_path), exist_ok=True)
        with open(self.best_param_path, "w") as f:
            json.dump(params, f, indent=4)
        print(f"[BaseModel] Saved best hyperparameters → {self.best_param_path}")

    # load best params
    def load_best_params(self):
        if not os.path.exists(self.best_param_path):
            return None
        with open(self.best_param_path, "r") as f:
            return json.load(f)

    # load or train tuned model
    def load_or_train_tuned_model(
        self,
        train_folds,
        val_folds,
        n_trials: int = 50,
        final_fold_idx: int = 0,
        save_name: str = "final_tuned_model",
        use_optuna: bool = True,
    ):
        if use_optuna and n_trials > 0:
            params = self.load_best_params()
            if params is None:
                print("[BaseModel] No best_params found – running Optuna.")
                study = self.run_optuna_search(
                    train_folds, val_folds, n_trials=n_trials
                )
                self.save_best_params(study)
                params = study.best_params
            else:
                print(
                    "[BaseModel] Loaded existing best hyperparameters – "
                    "skipping tuning."
                )

            tuned_cfg = BaseModelConfig(
                model_name=self.cfg.model_name,
                **params,
            )
            final_model_wrapper = self.__class__(tuned_cfg)
        else:
            print("[BaseModel] Skipping Optuna – using provided cfg.")
            final_model_wrapper = self.__class__(self.cfg)

        trainer, tokenizer = final_model_wrapper.train(
            train_ds=train_folds[final_fold_idx],
            val_ds=val_folds[final_fold_idx],
            save_name=save_name,
        )

        model = trainer.model
        return model, tokenizer
    
    def compute_metrics(self, eval_pred):
        # Should NEVER be called during training/eval now
        # but keep it safe for manual eval
        preds, labels = eval_pred
        mask = labels != -100
        preds = preds[mask]
        labels = labels[mask]

        return {
            "accuracy": accuracy_metric.compute(
                predictions=preds, references=labels
            )["accuracy"]
        }

        return {
            "accuracy": accuracy_metric.compute(
                predictions=preds, references=labels
            )["accuracy"]
        }

    def save_predictions(self, model, tokenizer, test_ds, save_name="base_eval", baseline=False):
        """
        Generates per-example predictions and saves:
        1) predictions file (JSONL)
        2) metrics summary file (JSON)

        Generic across all tasks:
        - Uses #lines in the (normalized) target as horizon_k
        - Extracts the first horizon_k clean lines from the model output
        - Compares exact string match of normalized prediction vs normalized target
        """
        out_dir = f"{self.task_dir(save_name)}/metrics"
        os.makedirs(out_dir, exist_ok=True)

        pred_path = f"{out_dir}/predictions.jsonl"
        summary_path = f"{out_dir}/summary.json"

        results = []
        correct_count = 0

        print(f"[BaseModel] Generating predictions → {pred_path}")

        for row in tqdm(test_ds, desc="Generating outputs"):
            prompt = row["prompt"]
            target = row["target"]

            # ==========
            # 1) Encode prompt exactly like training:
            #    prompt: <|im_start|>user ... <|im_end|>\n<|im_start|>assistant\n
            #    then we append "<SOL>\n" as in tokenize_dataset()
            # ==========
            if baseline:
                gen_prompt = prompt.replace("<|im_start|>assistant", "<|im_start|>assistant\n<SOL>")
            else:
                gen_prompt = prompt + "\n<SOL>\n"

            enc = tokenizer(gen_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = enc.input_ids.to(model.device)
            attention_mask = enc.attention_mask.to(model.device)

            gold_norm = normalize_target_text(target)
            horizon_k = get_horizon_k(target)
            # ==========
            # 2) Generate continuation (greedy, deterministic)
            # ==========
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=horizon_k+15,
                    do_sample=False,  # greedy
                    temperature=0.0,
                    eos_token_id=tokenizer.eos_token_id,
                )[0]

            # newly generated tokens only
            gen_ids = output_ids[input_ids.shape[-1] :]

            # NOTE: keep special tokens in decode so we can cut on <|im_end|> if needed
            raw = tokenizer.decode(gen_ids, skip_special_tokens=False)
            full_output = tokenizer.decode(output_ids, skip_special_tokens=False)
            pred_only = clean_generation(raw)

            # ==========
            # 3) Generic evaluation: compare normalized K lines
            # ==========
            

            pred_lines = extract_answer_lines(pred_only, horizon_k)
            if pred_lines is None:
                is_correct = 0
            else:
                pred_norm = normalize_target_text(pred_lines)
                is_correct = int(pred_norm == gold_norm)

            correct_count += is_correct

            results.append(
                {
                    "prompt": prompt,
                    "target": target,
                    "prediction_full": full_output,
                    "prediction_raw": raw,
                    "prediction_clean": pred_only,
                    "prediction_used": pred_lines,
                    "correct": is_correct,
                }
            )

        # ==========
        # 4) Save predictions + summary
        # ==========
        with open(pred_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        accuracy = correct_count / max(1, len(results))
        summary = {
            "accuracy": accuracy,
            "total_examples": len(results),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        print(f"[BaseModel] Saved metrics → {summary_path}")
        if baseline:
            print("RAW MODEL OUTPUT (last example):\n", results[-1]["prediction_raw"])
        else:
            print("RAW MODEL OUTPUT (last example):\n", results[-1]["prediction_full"])


        return summary
