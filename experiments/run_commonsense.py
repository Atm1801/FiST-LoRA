"""
FiST-LoRA: CommonSense Experiments
====================================

LLaMA-2-7B on CommonSense170K, evaluated on 8 commonsense reasoning tasks.

Usage:
    python experiments/run_commonsense.py
    python experiments/run_commonsense.py --ranks 32 --methods fist_full lora_xs
"""

import os
import sys
import json
import time
import math
import argparse
import traceback
from pathlib import Path

import torch
import numpy as np

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from datasets import load_dataset

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fist_lora.fisher import compute_diagonal_fisher
from fist_lora.init import gradient_projected_R, zero_R
from fist_lora.model import (
    inject_fist_lora,
    count_trainable_params,
    count_total_params,
    collect_plain_svd,
    collect_fisher_svd,
)
from fist_lora.utils import (
    set_seed,
    get_device,
    compute_warmup_steps,
    print_trainable_summary,
)


MODEL_NAME = "meta-llama/Llama-2-7b-hf"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
ALPHA = 32.0
CALIBRATION_SAMPLES = 256
INIT_SCALE = 0.01
MAX_SEQ_LEN = 256
HEAD_KEYWORDS = ["lm_head"]

EVAL_TASKS = ["boolq", "piqa", "siqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]

METHOD_LRS = {
    "lora": 2e-5,
    "lora_xs": 1e-4,
    "fist_no_fisher": 1e-4,
    "fist_full": 1e-4,
    "lora_sb": 1e-4,
}

ALL_METHODS = ["lora", "lora_xs", "fist_no_fisher", "fist_full"]


def get_bnb_config():
    """4-bit quantization config for 7B models."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def load_commonsense170k(tokenizer, max_length=MAX_SEQ_LEN):
    """
    Load CommonSense170K training data.
    Expected format: instruction-following with input/output pairs.
    """
    try:
        dataset = load_dataset("json", data_files="data/commonsense_170k.json")["train"]
    except Exception:
        print("[WARNING] Could not load local commonsense_170k.json.")
        print("Please download from: https://github.com/AGI-Edgerunners/LLM-Adapters")
        print("Place at: data/commonsense_170k.json")
        raise

    def tokenize_fn(examples):
        # Format: instruction -> answer
        texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            inp = examples.get("input", [""] * len(examples["instruction"]))
            input_text = inp[0] if isinstance(inp, list) else inp
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            texts.append(prompt)

        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        encodings["labels"] = encodings["input_ids"].copy()
        return encodings

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized.set_format("torch")
    return tokenized


def warmup_lm_head(model, train_dataset, tokenizer, num_steps=100, batch_size=4, lr=1e-3):
    """Warmup LM head for calibration (causal LM variant)."""
    device = next(model.parameters()).device

    for param in model.parameters():
        param.requires_grad_(False)

    head_params = []
    for name, param in model.named_parameters():
        if any(kw in name for kw in HEAD_KEYWORDS):
            param.requires_grad_(True)
            head_params.append(param)

    if not head_params:
        print("[Warmup] No LM head found; skipping warmup.")
        return model

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")
    n_examples = min(num_steps * batch_size, len(train_dataset))
    loader = torch.utils.data.DataLoader(
        train_dataset.select(range(n_examples)),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    optim = torch.optim.AdamW(head_params, lr=lr)
    model.train()

    step = 0
    losses = []
    for batch in loader:
        if step >= num_steps:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        if loss is None:
            break
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
        step += 1

    if losses:
        print(f"[Warmup] LM head: {step} steps. Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    return model


def make_calibration_loader_causal(train_dataset, tokenizer, num_samples=256, batch_size=4):
    """Create calibration DataLoader for causal LM."""
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")
    n = min(num_samples, len(train_dataset))
    subset = train_dataset.select(range(n))
    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )


def evaluate_commonsense(model, tokenizer, model_path=None):
    """
    Evaluate on commonsense reasoning tasks using lm-eval-harness.

    Returns dict: {task_name: accuracy}.
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)

        task_map = {
            "boolq": "boolq",
            "piqa": "piqa",
            "siqa": "social_iqa",
            "hellaswag": "hellaswag",
            "winogrande": "winogrande",
            "arc_easy": "arc_easy",
            "arc_challenge": "arc_challenge",
            "openbookqa": "openbookqa",
        }

        results = {}
        for short_name, lm_eval_name in task_map.items():
            try:
                task_results = lm_eval.simple_evaluate(
                    model=lm,
                    tasks=[lm_eval_name],
                    batch_size=4,
                )
                acc = task_results["results"][lm_eval_name].get(
                    "acc,none", task_results["results"][lm_eval_name].get("acc_norm,none", 0)
                )
                results[short_name] = acc
                print(f"  {short_name}: {acc:.4f}")
            except Exception as e:
                print(f"  {short_name}: ERROR - {e}")
                results[short_name] = None

        return results

    except ImportError:
        print("[WARNING] lm-eval not installed. Install with: pip install lm-eval")
        print("Skipping evaluation.")
        return {}


def apply_adapter_causal(model, method, rank, svd_plain, svd_fisher, grad_R_plain, grad_R_fisher):
    """Apply adapter to a causal LM."""
    from peft import get_peft_model, LoraConfig, TaskType

    if method == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=ALPHA,
            lora_dropout=0.0,
            target_modules=TARGET_MODULES,
            bias="none",
        )
        model = get_peft_model(model, peft_config)

    elif method == "lora_xs":
        B_dict = {n: BA[0] for n, BA in svd_plain.items()}
        A_dict = {n: BA[2] for n, BA in svd_plain.items()}
        R_dict = {n: zero_R(rank) for n in svd_plain}
        model = inject_fist_lora(
            model, TARGET_MODULES, B_dict, A_dict, R_dict, ALPHA, rank,
            head_keywords=HEAD_KEYWORDS,
        )

    elif method == "fist_no_fisher":
        B_dict = {n: BA[0] for n, BA in svd_plain.items()}
        A_dict = {n: BA[2] for n, BA in svd_plain.items()}
        R_dict = {n: grad_R_plain.get(n, zero_R(rank)) for n in svd_plain}
        model = inject_fist_lora(
            model, TARGET_MODULES, B_dict, A_dict, R_dict, ALPHA, rank,
            head_keywords=HEAD_KEYWORDS,
        )

    elif method == "fist_full":
        B_dict = {n: BA[0] for n, BA in svd_fisher.items()}
        A_dict = {n: BA[2] for n, BA in svd_fisher.items()}
        R_dict = {n: grad_R_fisher.get(n, zero_R(rank)) for n in svd_fisher}
        model = inject_fist_lora(
            model, TARGET_MODULES, B_dict, A_dict, R_dict, ALPHA, rank,
            head_keywords=HEAD_KEYWORDS,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return model


def main():
    parser = argparse.ArgumentParser(description="FiST-LoRA CommonSense experiments")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--ranks", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--output_dir", type=str, default="results/commonsense")
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    methods = args.methods or ALL_METHODS
    device = get_device()
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.json")
    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize training data
    print("Loading CommonSense170K...")
    train_dataset = load_commonsense170k(tokenizer)
    print(f"Training samples: {len(train_dataset):,}")

    # Load base model for calibration (4-bit quantized)
    print("Loading base model for calibration...")
    bnb_config = get_bnb_config()
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    base_model.eval()

    # Warmup LM head
    print("Warming up LM head for calibration...")
    base_model = warmup_lm_head(base_model, train_dataset, tokenizer)

    # Calibration
    calibration_loader = make_calibration_loader_causal(train_dataset, tokenizer)

    print("Computing diagonal Fisher...")
    fisher_diags = compute_diagonal_fisher(
        base_model, calibration_loader, TARGET_MODULES, CALIBRATION_SAMPLES
    )

    # Pre-compute SVD and gradient R for all ranks
    svd_plain_by_rank = {}
    svd_fisher_by_rank = {}
    grad_R_plain_by_rank = {}
    grad_R_fisher_by_rank = {}

    for rank in args.ranks:
        print(f"\nComputing SVD and gradient R for rank={rank}...")
        svd_plain_by_rank[rank] = collect_plain_svd(base_model, TARGET_MODULES, rank)
        svd_fisher_by_rank[rank] = collect_fisher_svd(
            base_model, fisher_diags, TARGET_MODULES, rank
        )

        plain_ba = {n: (B, A) for n, (B, S, A) in svd_plain_by_rank[rank].items()}
        grad_R_plain_by_rank[rank] = gradient_projected_R(
            base_model, calibration_loader, plain_ba,
            TARGET_MODULES, CALIBRATION_SAMPLES, ALPHA, rank,
            init_scale=INIT_SCALE,
        )

        fisher_ba = {n: (B, A) for n, (B, S, A) in svd_fisher_by_rank[rank].items()}
        grad_R_fisher_by_rank[rank] = gradient_projected_R(
            base_model, calibration_loader, fisher_ba,
            TARGET_MODULES, CALIBRATION_SAMPLES, ALPHA, rank,
            init_scale=INIT_SCALE,
        )

    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Train and evaluate
    for rank in args.ranks:
        for seed in args.seeds:
            for method in methods:
                run_key = f"rank_{rank}_seed_{seed}"
                if run_key not in results:
                    results[run_key] = {}
                if method in results[run_key] and "error" not in results[run_key][method]:
                    print(f"\n[Resume] Skipping {run_key}/{method}")
                    continue

                print(f"\n{'=' * 60}")
                print(f"  Rank: {rank} | Seed: {seed} | Method: {method}")
                print(f"{'=' * 60}")

                set_seed(seed)
                start_time = time.time()

                # Load fresh model
                bnb_config = get_bnb_config()
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )

                try:
                    model = apply_adapter_causal(
                        model, method, rank,
                        svd_plain_by_rank[rank],
                        svd_fisher_by_rank[rank],
                        grad_R_plain_by_rank[rank],
                        grad_R_fisher_by_rank[rank],
                    )
                except Exception:
                    print(f"[ERROR] Adapter failed for {method}:")
                    traceback.print_exc()
                    results[run_key][method] = {"error": "adapter_construction_failed"}
                    continue

                trainable, total = print_trainable_summary(model, method)

                lr = METHOD_LRS.get(method, 1e-4)
                warmup_steps = compute_warmup_steps(
                    len(train_dataset), 4, 3,
                    gradient_accumulation_steps=16,
                )

                training_args = TrainingArguments(
                    output_dir=os.path.join(args.output_dir, f"{method}_r{rank}_s{seed}"),
                    num_train_epochs=3,
                    per_device_train_batch_size=4,
                    per_device_eval_batch_size=4,
                    gradient_accumulation_steps=16,
                    learning_rate=lr,
                    weight_decay=0.0,
                    warmup_steps=warmup_steps,
                    logging_steps=50,
                    save_strategy="no",
                    seed=seed,
                    bf16=True,
                    report_to=args.report_to,
                    gradient_checkpointing=True,
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    processing_class=tokenizer,
                    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
                )

                try:
                    trainer.train()
                except Exception:
                    print(f"[ERROR] Training failed for {method}:")
                    traceback.print_exc()
                    results[run_key][method] = {
                        "error": "training_failed",
                        "trainable_params": trainable,
                    }
                    del model, trainer
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    continue

                elapsed = time.time() - start_time

                # Evaluate
                eval_results = {}
                if not args.skip_eval:
                    print("Evaluating on commonsense tasks...")
                    eval_results = evaluate_commonsense(model, tokenizer)

                results[run_key][method] = {
                    "eval": eval_results,
                    "trainable_params": trainable,
                    "time_seconds": elapsed,
                    "seed": seed,
                    "rank": rank,
                }

                del model, trainer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

                # Save intermediate
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("COMMONSENSE RESULTS")
    print("=" * 80)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
