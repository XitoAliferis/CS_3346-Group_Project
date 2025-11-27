import os
import json
import torch
import random
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import evaluate
from tqdm import tqdm
accuracy_metric = evaluate.load("accuracy")
import re
import optuna

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model



MOVE_LINE_RE = re.compile(r"^[A-C]->[A-C]$")

def extract_moves(text, horizon_k):
    lines = [l.strip() for l in text.split("\n")]
    moves = [l for l in lines if MOVE_LINE_RE.match(l)]
    if len(moves) < horizon_k:
        return None
    return "\n".join(moves[:horizon_k])

# base config
@dataclass
class BaseModelConfig:
    model_name: str = ""

    # train settings
    lr: float = 2e-5
    batch_size: int = 4
    num_epochs: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    lora_rank: int = 16

    use_lora: bool = True


# base model class
class BaseModel:
    def __init__(self, cfg: BaseModelConfig):
        self.cfg = cfg

    # file path for best params
    @property
    def best_param_path(self) -> str:
        safe_name = self.cfg.model_name.replace("/", "_")
        return f"../Results/{safe_name}_best_params.json"

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

        return train_folds, val_folds

    # ---------------------------------------------------------
    # 1) load base model no lora
    # ---------------------------------------------------------
    def load_base_model(self):
        print(f"[BaseModel] Loading base model ONLY: {self.cfg.model_name}")

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        return model, tokenizer

    # ---------------------------------------------------------
    # 2) load train model with lora
    # ---------------------------------------------------------
    def load_train_model(self):
        print(f"[BaseModel] Loading training model: {self.cfg.model_name}")

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )

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
            full = row["prompt"] + "\n" + row["target"]
            enc = tokenizer(full, truncation=True, max_length=2048)
            enc["labels"] = enc["input_ids"].copy()
            return enc

        return dataset.map(_tok)

    # ---------------------------------------------------------
    # 4) build train args
    # ---------------------------------------------------------
    def build_training_args(self, save_name: str):
        return TrainingArguments(
            output_dir=f"../Results/{save_name}",
            per_device_train_batch_size=self.cfg.batch_size,
            learning_rate=self.cfg.lr,
            num_train_epochs=self.cfg.num_epochs,
            weight_decay=self.cfg.weight_decay,
            warmup_ratio=self.cfg.warmup_ratio,
            bf16=True,
            logging_steps=20,
            save_steps=500,
            save_total_limit=2,
            report_to="none",
        )

    # ---------------------------------------------------------
    # 5) run one train
    # ---------------------------------------------------------
    def train(self, train_ds, val_ds, save_name: str):
        model, tokenizer = self.load_train_model()

        train_tok = self.tokenize_dataset(train_ds, tokenizer)
        val_tok   = self.tokenize_dataset(val_ds, tokenizer)

        collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        args     = self.build_training_args(save_name)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            data_collator=collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.save_model(f"../Results/{save_name}/final_model")

        return trainer

    # ---------------------------------------------------------
    # 6) generic EVAL / TEST function
    # ---------------------------------------------------------
    def evaluate_model(self, model, tokenizer, test_ds, save_name: str = "test_eval"):
        """
        Evaluates a given model+tokenizer on a test dataset.
        Returns HF metrics dict (e.g. eval_loss, etc.).
        """
        test_tok = self.tokenize_dataset(test_ds, tokenizer)

        collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        args = TrainingArguments(
            output_dir=f"../Results/{save_name}",
            per_device_eval_batch_size=self.cfg.batch_size,
            do_train=False,
            do_eval=True,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=args,
            eval_dataset=test_tok,
            data_collator=collator,
            compute_metrics=self.compute_metrics,
        )

        metrics = trainer.evaluate()
        return metrics

    def evaluate_with_predictions(self, model, tokenizer, test_ds, save_name: str):
        """Runs HF evaluate + saves prediction file + summary file"""
        hf_metrics = self.evaluate_model(model, tokenizer, test_ds, save_name)
        prediction_metrics = self.save_predictions(model, tokenizer, test_ds, save_name)

        return {**hf_metrics, **prediction_metrics}

    def evaluate_base_model(self, test_ds, save_name: str = "base_eval"):
        model, tokenizer = self.load_base_model()
        return self.evaluate_with_predictions(model, tokenizer, test_ds, save_name)
    
    def evaluate_tuned_model(self, model, tokenizer, test_ds, save_name="tuned_eval"):
        return self.evaluate_with_predictions(model, tokenizer, test_ds, save_name)

    # ---------------------------------------------------------
    # 7) optuna run for 5 folds
    # ---------------------------------------------------------
    def objective(self, trial, train_folds, val_folds):
        tuned_cfg = self._sample_cfg_from_trial(trial)

        model_wrapper = self.__class__(tuned_cfg)

        losses = []
        for i in range(5):
            trainer = model_wrapper.train(
                train_ds=train_folds[i],
                val_ds=val_folds[i],
                save_name=f"trial_{trial.number}_fold_{i}",
            )
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
    ):
        params = self.load_best_params()

        if params is None:
            print("[BaseModel] No best_params found – running Optuna.")
            study = self.run_optuna_search(train_folds, val_folds, n_trials=n_trials)
            self.save_best_params(study)
            params = study.best_params
        else:
            print("[BaseModel] Loaded existing best hyperparameters – skipping tuning.")

        tuned_cfg = BaseModelConfig(
            model_name=self.cfg.model_name,
            **params,
        )
        final_model_wrapper = self.__class__(tuned_cfg)

        trainer = final_model_wrapper.train(
            train_ds=train_folds[final_fold_idx],
            val_ds=val_folds[final_fold_idx],
            save_name=save_name,
        )

        model, tokenizer = final_model_wrapper.load_train_model()
        return model, tokenizer
    
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred

        # logits: [batch, seq_len, vocab]
        # we take the highest-probability token at each step
        preds = np.argmax(logits, axis=-1)

        # shift to ignore padded tokens (-100)
        mask = labels != -100
        preds = preds[mask]
        labels = labels[mask]

        return {
            "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        }
    
    def save_predictions(self, model, tokenizer, test_ds, save_name="base_eval"):
        """
        Generates per-example predictions and saves:
        1) predictions file
        2) metrics summary file
        """

        # directory
        out_dir = f"./Results/metrics/{save_name}"
        os.makedirs(out_dir, exist_ok=True)

        # output file paths
        pred_path = f"{out_dir}/predictions.jsonl"
        summary_path = f"{out_dir}/summary.json"

        results = []
        correct_count = 0

        print(f"[BaseModel] Generating predictions → {pred_path}")

        for row in tqdm(test_ds, desc="Generating outputs"):
            prompt = row["prompt"]
            target = row["target"]

            # generate model output
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            output_ids = model.generate(input_ids, max_new_tokens=50)[0]
            pred_text = tokenizer.decode(output_ids, skip_special_tokens=True)

            # extract only the generated part
            pred_only = pred_text[len(prompt):].strip()

            # binary correctness
            is_correct = int(pred_only.strip() == target.strip())
            correct_count += is_correct

            results.append({
                "prompt": prompt,
                "target": target,
                "prediction": pred_only,
                "correct": is_correct,
            })

        # save predictions file
        with open(pred_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        # summary metrics
        accuracy = correct_count / len(results)

        summary = {
            "accuracy": accuracy,
            "total_examples": len(results),
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"[BaseModel] Saved metrics → {summary_path}")
        print("RAW MODEL OUTPUT:\n", pred_text)

        return summary
