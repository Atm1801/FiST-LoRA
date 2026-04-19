"""
FiST-LoRA: Math Experiments
==============================

Mistral-7B on MetaMathQA, evaluated on GSM8K and MATH (exact match).

Usage:
    python experiments/run_math.py
    python experiments/run_math.py --ranks 32 64 --methods fist_full lora
"""

import os
import sys
import re
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


MODEL_NAME = "mistralai/Mistral-7B-v0.1"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
ALPHA = 32.0
CALIBRATION_SAMPLES = 256
INIT_SCALE = 0.01
MAX_SEQ_LEN = 512
HEAD_KEYWORDS = ["lm_head"]
MAX_TRAIN_SAMPLES = 50000

METHOD_LRS = {
    "lora": 2e-5,
    "lora_xs": 1e-4,
    "fist_no_fisher": 1e-4,
    "fist_full": 1e-4,
    "lora_sb": 1e-4,
}

ALL_METHODS = ["lora", "lora_xs", "fist_no_fisher", "fist_full"]


def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def load_metamathqa(tokenizer, max_samples=MAX_TRAIN_SAMPLES, max_length=MAX_SEQ_LEN):
    """Load and tokenize MetaMathQA dataset."""
    dataset = load_dataset("meta-math/MetaMathQA", split="train")

    if max_samples and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    def tokenize_fn(examples):
        texts = []
        for query, response in zip(examples["query"], examples["response"]):
            text = f"### Question:\n{query}\n\n### Answer:\n{response}"
            texts.append(text)

        encodings = tokenizer(
            texts, truncation=True, max_length=max_length, padding=False
        )
        encodings["labels"] = encodings["input_ids"].copy()
        return encodings

    tokenized = dataset.map(
        tokenize_fn, batched=True, remove_columns=dataset.column_names
    )
    tokenized.set_format("torch")
    return tokenized


def extract_answer(text):
    """Extract the final numeric answer from model output."""
    # Look for #### pattern (GSM8K format)
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")

    # Look for boxed answer (MATH format)
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()

    # Fall back to last number in text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def evaluate_gsm8k(model, tokenizer, num_samples=None):
    """Evaluate on GSM8K test set with exact match."""
    dataset = load_dataset("gsm8k", "main", split="test")
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    correct = 0
    total = 0
    device = next(model.parameters()).device

    model.eval()
    for example in dataset:
        question = example["question"]
        gold_answer = extract_answer(example["answer"])

        prompt = f"### Question:\n{question}\n\n### Answer:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_answer = extract_answer(generated)

        if pred_answer.strip() == gold_answer.strip():
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"  GSM8K: {correct}/{total} = {accuracy:.4f}")
    return accuracy


def evaluate_math(model, tokenizer, num_samples=None):
    """Evaluate on MATH test set with exact match."""
    try:
        dataset = load_dataset("hendrycks/competition_math", split="test")
    except Exception:
        print("  MATH dataset not available. Skipping.")
        return None

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    correct = 0
    total = 0
    device = next(model.parameters()).device

    model.eval()
    for example in dataset:
        question = example["problem"]
        gold_answer = extract_answer(example["solution"])

        prompt = f"### Question:\n{question}\n\n### Answer:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_answer = extract_answer(generated)

        if pred_answer.strip() == gold_answer.strip():
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"  MATH: {correct}/{total} = {accuracy:.4f}")
    return accuracy


def warmup_lm_head(model, train_dataset, tokenizer, num_steps=100, batch_size=4, lr=1e-3):
    """Warmup LM head for calibration."""
    device = next(model.parameters()).device

    for param in model.parameters():
        param.requires_grad_(False)

    head_params = []
    for name, param in model.named_parameters():
        if any(kw in name for kw in HEAD_KEYWORDS):
            param.requires_grad_(True)
            head_params.append(param)

    if not head_params:
        print("[Warmup] No LM head found; skipping.")
        return model

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")
    n_examples = min(num_steps * batch_size, len(train_dataset))
    loader = torch.utils.data.DataLoader(
        train_dataset.select(range(n_examples)),
        batch_size=batch_size, shuffle=True, collate_fn=collator,
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
    parser = argparse.ArgumentParser(description="FiST-LoRA Math experiments")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--ranks", nargs="+", type=int, default=[32, 64, 96])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--output_dir", type=str, default="results/math")
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=MAX_TRAIN_SAMPLES)
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

    print("Loading MetaMathQA...")
    train_dataset = load_metamathqa(tokenizer, max_samples=args.max_train_samples)
    print(f"Training samples: {len(train_dataset):,}")

    # Calibration
    print("Loading base model for calibration...")
    bnb_config = get_bnb_config()
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    base_model.eval()

    print("Warming up LM head...")
    base_model = warmup_lm_head(base_model, train_dataset, tokenizer)

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")
    n_cal = min(CALIBRATION_SAMPLES, len(train_dataset))
    calibration_loader = torch.utils.data.DataLoader(
        train_dataset.select(range(n_cal)),
        batch_size=4, shuffle=False, collate_fn=collator,
    )

    print("Computing diagonal Fisher...")
    fisher_diags = compute_diagonal_fisher(
        base_model, calibration_loader, TARGET_MODULES, CALIBRATION_SAMPLES
    )

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
                    len(train_dataset), 4, 2,
                    gradient_accumulation_steps=32,
                )

                training_args = TrainingArguments(
                    output_dir=os.path.join(args.output_dir, f"{method}_r{rank}_s{seed}"),
                    num_train_epochs=2,
                    per_device_train_batch_size=4,
                    per_device_eval_batch_size=4,
                    gradient_accumulation_steps=32,
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
                    results[run_key][method] = {"error": "training_failed", "trainable_params": trainable}
                    del model, trainer
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    continue

                elapsed = time.time() - start_time

                eval_results = {}
                if not args.skip_eval:
                    print("Evaluating on GSM8K...")
                    eval_results["gsm8k"] = evaluate_gsm8k(model, tokenizer)
                    print("Evaluating on MATH...")
                    eval_results["math"] = evaluate_math(model, tokenizer)

                results[run_key][method] = {
                    "eval": eval_results,
                    "trainable_params": trainable,
                    "time_seconds": elapsed,
                    "seed": seed,
                    "rank": rank,
                }

                del model, trainer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("MATH RESULTS")
    print("=" * 80)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
