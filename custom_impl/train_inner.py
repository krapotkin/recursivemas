#!/usr/bin/env python3
"""
train_inner.py — Inner-Loop Training (воспроизведение оригинала)

Обучает inner RecursiveLink adapter для каждой роли MAS (planner/refiner/solver)
на датасете RecursiveMAS/Sequential-Math из HuggingFace (как в оригинале).

Следует оригинальной реализации из experiments/original/train/train_inner.py:
- Adapter: ln_res_adapter (LayerNorm → MLP → residual → LayerNorm)
- Loss: cosine similarity (1.0) + MSE (0.0 по умолчанию)
- Optimizer: AdamW(betas=(0.9, 0.95))
- Scheduler: cosine with warmup
- Dataset: RecursiveMAS/Sequential-Math (HuggingFace, 1904 примера)
- Prompts: оригинальные из mas_prompt.py (build_math_planner/refiner/solver_prompt)

Поддерживаемые датасеты (как в оригинале):
- RecursiveMAS/Sequential-Math — основной (по умолчанию, как в оригинале)
- openai/gsm8k — GSM8K с форматированием \boxed{} (для отладки)
- Локальные JSON/JSONL файлы

Пример:
    # Planner
    python train_inner.py --mas_role planner --model_name_or_path Qwen/Qwen3-1.7B

    # Refiner (Critic)
    python train_inner.py --mas_role refiner --model_name_or_path meta-llama/Llama-3.2-1B-Instruct

    # Solver
    python train_inner.py --mas_role solver --model_name_or_path ./models/Qwen2.5-Math-1.5B-Instruct

    # С GSM8K (для отладки)
    python train_inner.py --mas_role solver --dataset_name openai/gsm8k --model_name_or_path ...

    # Resume training
    python train_inner.py --mas_role planner --model_name_or_path ... --load_dir ckpts/inner_planner
"""

import argparse
import json
import math
import os
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from datasets import load_dataset

# Fix for CUDA
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# Suppress datasets progress bars
try:
    import datasets as _datasets
    _datasets.disable_progress_bars()
except Exception:
    pass


# ============================================================
# Inner Adapter (ln_res_adapter — как в оригинале)
# ============================================================
class InnerAdapter(torch.nn.Module):
    """
    ln_res_adapter из оригинала:
    LayerNorm → W1 → GELU → W2 → +residual → LayerNorm
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.pre_ln = torch.nn.LayerNorm(hidden_size)
        self.proj1 = torch.nn.Linear(hidden_size, hidden_size)
        self.act = torch.nn.GELU()
        self.proj2 = torch.nn.Linear(hidden_size, hidden_size)
        self.post_ln = torch.nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_ln(x)
        out = self.proj2(self.act(self.proj1(h)))
        out = x + out  # residual
        out = self.post_ln(out)
        return out


# ============================================================
# Prompts (как в оригинале mas_prompt.py)
# ============================================================
SYSTEM_PROMPT = "You are a helpful assistant."

def build_math_planner_prompt(question: str) -> str:
    """Оригинальный промпт из mas_prompt.py — Sequential-Math."""
    return (
        "You are a planner agent in a multi-agent system.\n"
        "Give a plan for the question below.\n"
        "Question:\n"
        f"{question}\n"
        "Your response should be in the format of:\n"
        "Step 1: ...\n"
        "...\n"
        "Step n: ..."
    )

def build_math_refiner_prompt(question: str, planner_plan: str) -> str:
    """Оригинальный промпт из mas_prompt.py — Sequential-Math."""
    return (
        "You are a refiner agent in a multi-agent system.\n"
        "The question is:\n"
        "Question:\n"
        f"{question}\n"
        "The initial plan from the planner:\n"
        "Initial Plan:\n"
        f"{planner_plan}\n"
        "You should refine the initial plan and respond with pure plan only in the format of:\n"
        "Step 1: ...\n"
        "...\n"
        "Step n: ..."
    )

def build_math_solver_prompt(question: str, refined_plan: str, solver_pre_question: int = 0) -> str:
    """Оригинальный промпт из mas_prompt.py — Sequential-Math."""
    import re
    is_choice_question = bool(re.search(r"(?mi)^\s*[A-D]\s*[\.):\-]\s+", question))
    final_instruction = (
        "Solve the question and put the final choice inside \\boxed{}, for example \\boxed{A}."
        if is_choice_question
        else "Solve the question given information and put the final answer inside \\boxed{}, for example \\boxed{1}."
    )
    if solver_pre_question:
        # Question first, then refined plan
        return (
            "You are a solver agent in a multi-agent system.\n"
            "The question is:\n"
            "Question:\n"
            f"{question}\n"
            "Here is the refined plan:\n"
            "Refined Plan:\n"
            f"{refined_plan}\n"
            f"{final_instruction}"
        )
    else:
        # Refined plan first, then question (default, как в оригинале)
        return (
            "You are a solver agent in a multi-agent system.\n"
            "Here is the refined plan:\n"
            "Refined Plan:\n"
            f"{refined_plan}\n"
            "The question is:\n"
            "Question:\n"
            f"{question}\n"
            f"{final_instruction}"
        )


# ============================================================
# GSM8K helpers (как в оригинале data.py)
# ============================================================
import re

def _build_gsm8k_user_prompt(question: str) -> str:
    return (
        f"Question: {question}\n\n"
        "Reason step by step and output the final answer inside \\boxed{YOUR_FINAL_ANSWER}:\n"
    )

def _extract_gsm8k_final_answer(answer: str) -> str:
    match = re.search(r"####\s*(.+)$", answer, flags=re.DOTALL)
    if not match:
        return answer.strip()
    final = match.group(1).strip()
    return final if final else answer.strip()

def _format_gsm8k_answer(answer: str) -> str:
    final = _extract_gsm8k_final_answer(answer)
    reason_text = answer.rstrip()
    return f"{reason_text}\nFinal Answer: \\boxed{{{final}}}"


# ============================================================
# Dataset loading
# ============================================================
def load_dataset_split(dataset_name: str, dataset_split: str, dataset_json_field: Optional[str] = None):
    """
    Загружает датасет (как в оригинале _load_dataset_split).
    
    Поддерживает:
    - RecursiveMAS/Sequential-Math
    - openai/gsm8k
    - Локальные JSON/JSONL файлы
    - Любой HF датасет
    """
    key = dataset_name.strip().lower()
    if key in {"gsm8k", "openai/gsm8k"}:
        return load_dataset("openai/gsm8k", "main", split=dataset_split)

    if os.path.isfile(dataset_name):
        suffix = os.path.splitext(dataset_name)[1].lower()
        if suffix == ".json":
            if dataset_json_field:
                return load_dataset("json", data_files=dataset_name, field=dataset_json_field, split="train")
            return load_dataset("json", data_files=dataset_name, split="train")
        if suffix == ".jsonl":
            return load_dataset("json", data_files=dataset_name, split="train")
        raise ValueError(f"Unsupported local dataset file type: {dataset_name}")

    return load_dataset(dataset_name, split=dataset_split)


def load_sequential_math(split: str = "train", num_samples: int = -1):
    """Загружает RecursiveMAS/Sequential-Math из HuggingFace."""
    ds = load_dataset("RecursiveMAS/Sequential-Math", split=split)
    if num_samples > 0 and num_samples < len(ds):
        ds = ds.select(range(num_samples))
    return ds


# ============================================================
# Tokenization
# ============================================================
def build_tokenized_batch(
    tokenizer,
    question: str,
    assistant_text: str,
    user_prompt: str,
    max_length: int = 2048,
    system_prompt: str = SYSTEM_PROMPT,
    enable_thinking: bool = False,
) -> dict:
    """
    Токенизирует batch sample с teacher forcing.
    
    Формирует chat template: system + user prompt → assistant response.
    Возвращает input_ids, attention_mask, loss_mask, latent_mask.
    
    Как в оригинале: _tokenize_chat_example() с system prompt.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_text},
    ]
    
    # Build kwargs for apply_chat_template
    kwargs = {"tokenize": False, "add_generation_prompt": False}
    if enable_thinking:
        kwargs["enable_thinking"] = True
    
    full_text = tokenizer.apply_chat_template(messages, **kwargs)
    
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    
    input_ids = tokenized.input_ids[0]
    attention_mask = tokenized.attention_mask[0]
    
    # Находим границу assistant токенов
    assistant_tokens = tokenizer(assistant_text, add_special_tokens=False).input_ids
    assistant_start = len(input_ids) - len(assistant_tokens)
    
    # loss_mask: 1 для assistant токенов (где считаем loss)
    loss_mask = torch.zeros(len(input_ids), dtype=torch.float)
    loss_mask[assistant_start:] = 1.0
    
    # latent_mask: 1 для assistant токенов (где применяем adapter)
    latent_mask = loss_mask.clone()
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "latent_mask": latent_mask,
    }


def build_gsm8k_tokenized_batch(
    tokenizer,
    question: str,
    answer: str,
    max_length: int = 2048,
    enable_thinking: bool = False,
) -> dict:
    """
    Токенизирует GSM8K пример с форматированием \boxed{}.
    
    Как в оригинале: _tokenize_gsm8k_chat_example().
    """
    answer_text = _format_gsm8k_answer(answer)
    user_prompt = _build_gsm8k_user_prompt(question)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": answer_text},
    ]
    
    kwargs = {"tokenize": False, "add_generation_prompt": False}
    if enable_thinking:
        kwargs["enable_thinking"] = True
    
    full_text = tokenizer.apply_chat_template(messages, **kwargs)
    
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    
    input_ids = tokenized.input_ids[0]
    attention_mask = tokenized.attention_mask[0]
    
    assistant_tokens = tokenizer(answer_text, add_special_tokens=False).input_ids
    assistant_start = len(input_ids) - len(assistant_tokens)
    
    loss_mask = torch.zeros(len(input_ids), dtype=torch.float)
    loss_mask[assistant_start:] = 1.0
    
    # latent_mask: 0 после "Final Answer:" (как в оригинале с marker_candidates)
    latent_mask = loss_mask.clone()
    marker_text = "\nFinal Answer:"
    marker_ids = tokenizer(marker_text, add_special_tokens=False).input_ids
    for i in range(assistant_start, len(input_ids) - len(marker_ids) + 1):
        if list(input_ids[i:i+len(marker_ids)]) == marker_ids:
            latent_mask[i:] = 0.0
            break
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "latent_mask": latent_mask,
    }


def collate_fn(batch: list, tokenizer) -> dict:
    """Collate function для DataLoader (left padding, как в оригинале)."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    
    input_ids = []
    attention_masks = []
    loss_masks = []
    latent_masks = []
    
    for item in batch:
        pad_len = max_len - item["input_ids"].shape[0]
        input_ids.append(torch.cat([
            torch.full((pad_len,), tokenizer.pad_token_id),
            item["input_ids"],
        ]))
        attention_masks.append(torch.cat([
            torch.zeros(pad_len, dtype=torch.long),
            item["attention_mask"],
        ]))
        loss_masks.append(torch.cat([
            torch.zeros(pad_len, dtype=torch.float),
            item["loss_mask"],
        ]))
        latent_masks.append(torch.cat([
            torch.zeros(pad_len, dtype=torch.float),
            item["latent_mask"],
        ]))
    
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "loss_mask": torch.stack(loss_masks),
        "latent_mask": torch.stack(latent_masks),
    }


# ============================================================
# Training
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inner-Loop Training for RecursiveMAS")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--mas_role", type=str, required=True,
                        choices=["planner", "refiner", "solver"])
    parser.add_argument("--dataset_name", type=str, default="RecursiveMAS/Sequential-Math")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_json_field", type=str, default=None,
                        help="Top-level field for local json dataset (e.g., data).")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--adapter_lr", type=float, default=5e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        choices=["constant", "cosine"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps (как в оригинале)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--load_dir", type=str, default=None,
                        help="Resume training from checkpoint directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--adapter_dtype", type=str, default="auto",
                        help="Adapter dtype (auto = same as model)")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--adapter_cos_weight", type=float, default=1.0)
    parser.add_argument("--adapter_mse_weight", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda:2")
    # Original flags for compatibility
    parser.add_argument("--enable_thinking", type=int, default=0, choices=[0, 1],
                        help="Enable thinking mode for tokenizer (DeepSeek-style)")
    parser.add_argument("--solver_pre_question", type=int, default=0, choices=[0, 1],
                        help="Put question before refined plan in solver prompt")
    parser.add_argument("--mas_task", type=str, default="math", choices=["math", "code", "choice"])
    parser.add_argument("--mas_design", type=str, default="sequential",
                        choices=["sequential", "hie", "distill", "deliberation"])
    return parser.parse_args()


def resolve_dtype(dtype_str: str) -> Optional[torch.dtype]:
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return None


def build_optimizer(adapter, weight_decay: float, adapter_lr: float):
    """AdamW с группировкой параметров (как в оригинале)."""
    decay, no_decay = [], []
    for name, param in adapter.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    
    param_groups = []
    if decay:
        param_groups.append({"params": decay, "weight_decay": weight_decay, "lr": adapter_lr})
    if no_decay:
        param_groups.append({"params": no_decay, "weight_decay": 0.0, "lr": adapter_lr})
    if not param_groups:
        raise ValueError("No trainable adapter parameters found.")
    
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.95))


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch_dtype = resolve_dtype(args.dtype)
    adapter_dtype = torch_dtype if args.adapter_dtype == "auto" else resolve_dtype(args.adapter_dtype)
    
    print("=" * 60)
    print(f"INNER-LOOP TRAINING: {args.mas_role.upper()}")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"Role: {args.mas_role}")
    print(f"Dataset: {args.dataset_name} ({args.dataset_split})")
    print(f"Device: {device}")
    print(f"Dtype: {args.dtype}, Adapter dtype: {args.adapter_dtype}")
    print(f"Max steps: {args.max_steps}")
    print(f"LR: {args.adapter_lr}")
    print(f"Grad accum: {args.grad_accum_steps}")
    print(f"Loss: cosine({args.adapter_cos_weight}) + mse({args.adapter_mse_weight})")
    print(f"mas_task: {args.mas_task}, mas_design: {args.mas_design}")
    if args.load_dir:
        print(f"Resume from: {args.load_dir}")
    print()
    
    # 1. Загрузка токенизатора
    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Загрузка модели
    print("📦 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map={"": device.index},
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    d_model = model.config.hidden_size
    print(f"📐 d_model: {d_model}")
    
    # 3. Создание adapter
    adapter_config = {
        "adapter_type": "ln_res_adapter",
        "hidden_size": d_model,
        "model_name_or_path": args.model_name_or_path,
        "mas_role": args.mas_role,
    }
    
    print(f"🔗 Creating InnerAdapter (ln_res_adapter, {d_model})...")
    adapter = InnerAdapter(d_model).to(device)
    if adapter_dtype:
        adapter.to(adapter_dtype)
    
    # Resume from checkpoint
    if args.load_dir:
        config_path = os.path.join(args.load_dir, "adapter_config.json")
        if os.path.isfile(config_path):
            with open(config_path, "r") as f:
                saved_config = json.load(f)
            print(f"  📋 Loaded config: {saved_config.get('mas_role', 'unknown')}")
        
        adapter_path = os.path.join(args.load_dir, "adapter.pt")
        if os.path.isfile(adapter_path):
            adapter.load_state_dict(torch.load(adapter_path, map_location="cpu", weights_only=True))
            print(f"  ✅ Loaded adapter from {args.load_dir}")
    
    total_params = sum(p.numel() for p in adapter.parameters())
    model_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Adapter params: {total_params:,} ({total_params/model_params*100:.2f}%)")
    
    # 4. Загрузка датасета
    print(f"📚 Loading {args.dataset_name}...")
    is_gsm8k = args.dataset_name.strip().lower() in {"gsm8k", "openai/gsm8k"}
    
    if is_gsm8k:
        dataset = load_dataset_split(args.dataset_name, args.dataset_split, args.dataset_json_field)
    elif args.dataset_name.strip().lower() == "recursivemas/sequential-math":
        dataset = load_sequential_math(split=args.dataset_split, num_samples=args.num_samples)
    else:
        dataset = load_dataset_split(args.dataset_name, args.dataset_split, args.dataset_json_field)
    
    print(f"📊 Dataset size: {len(dataset)}")
    
    # 5. Токенизация датасета
    print("🔤 Tokenizing dataset...")
    tokenized_samples = []
    for i, sample in enumerate(dataset):
        question = sample["question"]
        
        if is_gsm8k:
            # GSM8K: question + answer → форматирование \boxed{}
            answer = sample.get("answer", "")
            if not answer:
                continue
            tokenized = build_gsm8k_tokenized_batch(
                tokenizer, question, answer, max_length=args.max_length,
                enable_thinking=bool(args.enable_thinking),
            )
        elif args.mas_role == "planner":
            user_prompt = build_math_planner_prompt(question)
            assistant_text = sample.get("plan", "")
            if not assistant_text:
                continue
            tokenized = build_tokenized_batch(
                tokenizer, question, assistant_text, user_prompt,
                max_length=args.max_length,
                enable_thinking=bool(args.enable_thinking),
            )
        elif args.mas_role == "refiner":
            user_prompt = build_math_refiner_prompt(question, sample.get("plan", ""))
            assistant_text = sample.get("refined_plan", "")
            if not assistant_text:
                continue
            tokenized = build_tokenized_batch(
                tokenizer, question, assistant_text, user_prompt,
                max_length=args.max_length,
                enable_thinking=bool(args.enable_thinking),
            )
        elif args.mas_role == "solver":
            user_prompt = build_math_solver_prompt(
                question, sample.get("refined_plan", ""),
                solver_pre_question=args.solver_pre_question,
            )
            assistant_text = sample.get("answer", "")
            if not assistant_text:
                continue
            tokenized = build_tokenized_batch(
                tokenizer, question, assistant_text, user_prompt,
                max_length=args.max_length,
                enable_thinking=bool(args.enable_thinking),
            )
        else:
            raise ValueError(f"Unknown role: {args.mas_role}")
        
        tokenized_samples.append(tokenized)
        
        if (i + 1) % 500 == 0:
            print(f"  Tokenized {i+1}/{len(dataset)}...")
    
    print(f"📊 Tokenized samples: {len(tokenized_samples)}")
    
    # 6. DataLoader
    dataloader = DataLoader(
        tokenized_samples,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )
    
    # 7. Оптимизатор и scheduler
    optimizer = build_optimizer(adapter, args.weight_decay, args.adapter_lr)
    
    num_update_steps = args.max_steps
    if args.lr_scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_update_steps,
            num_cycles=0.5,
        )
    else:
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
        )
    
    # 8. Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    embedding_layer = model.get_input_embeddings()
    global_step = 0
    start_time = time.time()
    
    log_stats = torch.zeros(4, device=device)  # loss, cos, mse, samples
    step_stats = torch.zeros(4, device=device)
    
    while global_step < args.max_steps:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            latent_mask = batch["latent_mask"].to(device)
            
            # Forward pass через ЗАМРОЖЕННУЮ модель
            with torch.no_grad():
                input_embeds = embedding_layer(input_ids)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden_states = outputs.hidden_states[-1]
            
            # Pairwise alignment: hidden[:, :-1] → target: input_embeds[:, 1:]
            hidden_prev = hidden_states[:, :-1, :]
            target_embeds = input_embeds[:, 1:, :]
            loss_mask_shifted = loss_mask[:, 1:]
            
            # Применяем adapter
            preds = adapter(hidden_prev)
            
            # Loss (float32 для стабильности)
            preds_f = preds.float()
            targets_f = target_embeds.float()
            
            cosine = 1.0 - torch.nn.functional.cosine_similarity(preds_f, targets_f, dim=-1)
            cosine_loss = (cosine * loss_mask_shifted).sum() / loss_mask_shifted.sum().clamp(min=1.0)
            
            mse = (preds_f - targets_f).pow(2).mean(dim=-1)
            mse_loss = (mse * loss_mask_shifted).sum() / loss_mask_shifted.sum().clamp(min=1.0)
            
            loss = args.adapter_cos_weight * cosine_loss + args.adapter_mse_weight * mse_loss
            
            # Backward с gradient accumulation (как в оригинале)
            loss = loss / args.grad_accum_steps
            loss.backward()
            
            if global_step % args.grad_accum_steps == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                
                if step_stats[3] > 0:
                    log_stats += step_stats
                step_stats.zero_()
            
            bs = input_ids.shape[0]
            step_stats[0] += loss.item() * bs * args.grad_accum_steps
            step_stats[1] += cosine_loss.item() * bs
            step_stats[2] += mse_loss.item() * bs
            step_stats[3] += bs
            
            if global_step % args.log_every == 0:
                denom = max(log_stats[3].item(), 1.0)
                avg = log_stats[:3] / denom
                elapsed = time.time() - start_time
                print(f"step={global_step} loss={avg[0]:.4f} cos={avg[1]:.4f} "
                      f"mse={avg[2]:.4f} time={elapsed:.0f}s", flush=True)
                log_stats.zero_()
            
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                if args.save_dir:
                    save_path = os.path.join(args.save_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(adapter.state_dict(), os.path.join(save_path, "adapter.pt"))
                    print(f"  💾 Saved checkpoint-{global_step}")
            
            if global_step >= args.max_steps:
                break
        
        if global_step >= args.max_steps:
            break
    
    # Финальный чекпоинт
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(adapter.state_dict(), os.path.join(args.save_dir, "adapter.pt"))
        with open(os.path.join(args.save_dir, "adapter_config.json"), "w") as f:
            json.dump({
                "adapter_type": "ln_res_adapter",
                "hidden_size": d_model,
                "model_name_or_path": args.model_name_or_path,
                "mas_role": args.mas_role,
                "dataset_name": args.dataset_name,
                "max_steps": global_step,
            }, f, indent=2)
        print(f"\n💾 Saved to {args.save_dir}/")
    
    elapsed = time.time() - start_time
    print(f"\n🎉 Training complete! steps={global_step}, time={elapsed:.0f}s")


if __name__ == "__main__":
    main()
