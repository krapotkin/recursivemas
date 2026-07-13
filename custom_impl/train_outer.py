#!/usr/bin/env python3
"""
train_outer.py — Outer-Loop Training (воспроизведение оригинала)

Обучает outer RecursiveLink адаптеры, соединяющие агенты MAS.
Полное воспроизведение оригинальной реализации из:
  experiments/original/train/outer/sequential.py + experiments/original/train/outer/common.py

Ключевые особенности (как в оригинале):
1. CrossModelAdapter: outer_ln_res_adapter (LayerNorm + hidden_dim=out_dim*2 + residual)
2. Recursive rounds: planner→refiner→solver цикл повторяется N раз (default=3)
3. Feedback: solver hidden states → outer_31 → planner для следующего round
4. Inner adapter gradient flow: градиенты проходят через frozen inner adapters
5. Loss: CE на solver output, supervise_final_only=1 (только последний round)
6. Dataset: RecursiveMAS/Sequential-Math
7. max_length=4096, max_latent_tokens=80

Пример:
    python train_outer.py \
        --agent1_model Qwen/Qwen3-1.7B \
        --agent2_model Qwen/Qwen3-1.7B \
        --agent3_model /home/hermes/workspace/models/models_llm/Qwen2.5-Math-1.5B-Instruct \
        --agent1_inner ckpts/planner_math/ \
        --agent2_inner ckpts/refiner_math/ \
        --agent3_inner ckpts/solver_math/ \
        --save_dir ckpts/outer_math \
        --num_recursive_rounds 3 \
        --max_length 4096 \
        --max_latent_tokens 80
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Suppress datasets progress bars
try:
    import datasets as _datasets
    _datasets.disable_progress_bars()
except Exception:
    pass


# ============================================================
# Prompts (оригинальные из mas_prompt.py)
# ============================================================
SYSTEM_PROMPT = "You are a helpful assistant."
PLANNER_SLOT = "<<LATENT_PLANNER_SLOT>>"
REFINED_SLOT = "<<LATENT_REFINED_SLOT>>"
FEEDBACK_SLOT = "<<LATENT_FEEDBACK_SLOT>>"


def build_math_planner_prompt(question: str) -> str:
    """Оригинальный промпт из mas_prompt.py."""
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


def build_math_planner_prompt_with_feedback_slot(question: str) -> str:
    """Оригинальный промпт из mas_prompt.py — recursive round 2+."""
    return (
        "You are a planner agent in a recursive multi-agent system.\n"
        "This is round 2.\n"
        "Question:\n"
        f"{question}\n"
        "Feedback signal from the previous solver round:\n"
        f"{FEEDBACK_SLOT}\n"
        "Use the feedback as a soft correction signal to improve the plan.\n"
        "If there is any conflict, prioritize the question constraints.\n"
        "Output only a concise plan in the format:\n"
        "Step 1: ...\n"
        "...\n"
        "Step n: ..."
    )


def build_math_refiner_prompt_with_slot(question: str) -> str:
    """Оригинальный промпт из mas_prompt.py — с planner slot."""
    return (
        "You are a refiner agent in a multi-agent system.\n"
        "The question is:\n"
        "Question:\n"
        f"{question}\n"
        "The initial plan from the planner:\n"
        "Initial Plan:\n"
        f"{PLANNER_SLOT}\n"
        "You should refine the initial plan and respond with pure plan only in the format of:\n"
        "Step 1: ...\n"
        "...\n"
        "Step n: ..."
    )


def build_math_solver_prompt_with_slots(question: str) -> str:
    """Оригинальный промпт из mas_prompt.py — с refined plan slot."""
    return (
        "You are a solver agent in a multi-agent system.\n"
        "Here is the refined plan:\n"
        "Refined Plan:\n"
        f"{REFINED_SLOT}\n"
        "The question is:\n"
        "Question:\n"
        f"{question}\n"
        "Solve the question given information and put the final answer inside \\boxed{}, for example \\boxed{1}."
    )


# ============================================================
# Inner Adapter (ln_res_adapter — как в оригинале model.py)
# ============================================================
class Adapter(nn.Module):
    """
    Inner adapter из model.py.
    Поддерживает все типы: linear_adapter, linear_res_adapter, adapter, res_adapter, ln_res_adapter.
    """
    def __init__(self, hidden_size: int, adapter_type: str = "ln_res_adapter"):
        super().__init__()
        self.adapter_type = adapter_type
        self.is_linear = adapter_type in {"linear_adapter", "linear_res_adapter"}
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU() if not self.is_linear else None
        self.proj2 = nn.Linear(hidden_size, hidden_size) if not self.is_linear else None
        self.use_residual = adapter_type in {"linear_res_adapter", "res_adapter", "ln_res_adapter"}
        self.pre_ln = nn.LayerNorm(hidden_size) if adapter_type == "ln_res_adapter" else None
        self.post_ln = nn.LayerNorm(hidden_size) if adapter_type == "ln_res_adapter" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_ln(x) if self.pre_ln is not None else x
        if self.is_linear:
            out = self.proj1(h)
        else:
            out = self.proj2(self.act(self.proj1(h)))
        if self.use_residual:
            out = x + out
        if self.post_ln is not None:
            out = self.post_ln(out)
        return out


# ============================================================
# CrossModelAdapter (как в оригинале model.py)
# ============================================================
class CrossModelAdapter(nn.Module):
    """
    Outer Link adapter из model.py.
    
    Типы: outer_linear_adapter, outer_linear_res_adapter, outer_adapter, 
          outer_res_adapter, outer_ln_adapter, outer_ln_res_adapter
    
    outer_ln_res_adapter (по умолчанию):
      LayerNorm(in) → Linear(in, out*2) → GELU → Linear(out*2, out) + Linear(in, out) → LayerNorm(out)
    """
    def __init__(self, in_dim: int, out_dim: int, adapter_type: str = "outer_ln_res_adapter"):
        super().__init__()
        self.adapter_type = adapter_type
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.is_linear = adapter_type in {"outer_linear_adapter", "outer_linear_res_adapter"}
        self.use_ln = adapter_type in {"outer_ln_adapter", "outer_ln_res_adapter"}
        self.use_residual = adapter_type in {"outer_linear_res_adapter", "outer_res_adapter", "outer_ln_res_adapter"}

        hidden_dim = out_dim * 2 if self.use_ln else out_dim
        self.proj1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU() if not self.is_linear else None
        self.proj2 = nn.Linear(hidden_dim, out_dim) if not self.is_linear else None

        self.ln_source = nn.LayerNorm(in_dim) if self.use_ln else None
        self.ln_target = nn.LayerNorm(out_dim) if self.use_ln else None
        self.residual_proj = nn.Linear(in_dim, out_dim) if self.use_residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_source(x) if self.ln_source is not None else x
        if self.is_linear:
            out = self.proj1(h)
        else:
            out = self.proj2(self.act(self.proj1(h)))
        if self.residual_proj is not None:
            out = out + self.residual_proj(x)
        if self.ln_target is not None:
            out = self.ln_target(out)
        return out


# ============================================================
# StagePack (как в оригинале outer/common.py)
# ============================================================
@dataclass
class StagePack:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    assistant_mask: torch.Tensor


# ============================================================
# Gradient flow helpers (как в оригинале outer/common.py)
# ============================================================
def run_inner_adapter_preserve_input_grad(adapter: nn.Module, x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """
    Применяет inner adapter БЕЗ torch.no_grad().
    Градиенты проходят ЧЕРЕЗ frozen inner adapter к upstream outer adapters.
    """
    adapter_param = next(adapter.parameters(), None)
    adapter_dtype = adapter_param.dtype if adapter_param is not None else x.dtype
    y = adapter(x.to(adapter_dtype))
    if y.dtype != out_dtype:
        y = y.to(out_dtype)
    return y


def run_outer_adapter(adapter: nn.Module, x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Применяет outer adapter."""
    adapter_param = next(adapter.parameters(), None)
    adapter_dtype = adapter_param.dtype if adapter_param is not None else x.dtype
    y = adapter(x.to(adapter_dtype))
    if y.dtype != out_dtype:
        y = y.to(out_dtype)
    return y


def trim_latent(latent: torch.Tensor, max_latent_tokens: int) -> torch.Tensor:
    """Ограничивает длину латентной последовательности."""
    if max_latent_tokens > 0 and latent.size(0) > max_latent_tokens:
        return latent[:max_latent_tokens]
    return latent


# ============================================================
# Chat template helpers (как в оригинале outer/common.py)
# ============================================================
def apply_chat_template(tokenizer, messages, tokenize, add_generation_prompt, enable_thinking=False):
    """apply_chat_template с fallback для enable_thinking и role alternation."""
    kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        kwargs.pop("enable_thinking", None)
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except Exception as inner_exc:
            err_text = str(inner_exc)
            if "Conversation roles must alternate" not in err_text:
                raise
    except Exception as exc:
        err_text = str(exc)
        if "Conversation roles must alternate" not in err_text:
            raise

    # Fallback: merge system + user
    normalized = list(messages)
    if (
        len(normalized) >= 2
        and isinstance(normalized[0], dict)
        and isinstance(normalized[1], dict)
        and normalized[0].get("role") == "system"
        and normalized[1].get("role") == "user"
    ):
        merged_user = dict(normalized[1])
        merged_user["content"] = (
            f"{normalized[0].get('content', '')}\n\n{normalized[1].get('content', '')}".strip()
        )
        normalized = [merged_user] + normalized[2:]
        return tokenizer.apply_chat_template(normalized, **kwargs)
    raise


def _normalize_template_text(tokenizer, value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "input_ids" in value:
            return _normalize_template_text(tokenizer, value["input_ids"])
        raise ValueError("chat template output mapping missing input_ids")
    if hasattr(value, "tolist"):
        return _normalize_template_text(tokenizer, value.tolist())
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if not value:
            return ""
        if isinstance(value[0], list):
            return _normalize_template_text(tokenizer, value[0])
        if isinstance(value[0], str):
            return "".join(value)
        return tokenizer.decode([int(x) for x in value], skip_special_tokens=False)
    raise ValueError(f"Unsupported chat template output type: {type(value)}")


def _normalize_template_ids(tokenizer, value, max_length=None) -> list:
    if isinstance(value, str):
        fallback_max_len = max_length or 32768
        return tokenizer(value, truncation=True, max_length=int(fallback_max_len), padding=False, add_special_tokens=False)["input_ids"]
    if isinstance(value, dict):
        return _normalize_template_ids(tokenizer, value.get("input_ids", []), max_length=max_length)
    if hasattr(value, "tolist"):
        return _normalize_template_ids(tokenizer, value.tolist(), max_length=max_length)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if not value:
            return []
        if isinstance(value[0], list):
            return _normalize_template_ids(tokenizer, value[0], max_length=max_length)
        return [int(x) for x in value]
    raise ValueError(f"Unsupported chat template output type: {type(value)}")


def render_chat_text(tokenizer, user_prompt: str, assistant_text, enable_thinking: bool) -> str:
    """Рендерит chat template в текст (для slot splitting)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if assistant_text is None:
        rendered = apply_chat_template(tokenizer, messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        return _normalize_template_text(tokenizer, rendered)
    messages.append({"role": "assistant", "content": assistant_text})
    rendered = apply_chat_template(tokenizer, messages, tokenize=False, add_generation_prompt=False, enable_thinking=enable_thinking)
    return _normalize_template_text(tokenizer, rendered)


def render_chat_ids(tokenizer, user_prompt: str, assistant_text, enable_thinking: bool, max_length=None) -> list:
    """Рендерит chat template в input_ids."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if assistant_text is None:
        rendered = apply_chat_template(tokenizer, messages, tokenize=True, add_generation_prompt=True, enable_thinking=enable_thinking)
        return _normalize_template_ids(tokenizer, rendered, max_length=max_length)
    messages.append({"role": "assistant", "content": assistant_text})
    rendered = apply_chat_template(tokenizer, messages, tokenize=True, add_generation_prompt=False, enable_thinking=enable_thinking)
    return _normalize_template_ids(tokenizer, rendered, max_length=max_length)


def split_rendered_text_by_slot(rendered_text: str, slot_text: str) -> Tuple[str, str]:
    """Split rendered text по slot marker."""
    pos = rendered_text.find(slot_text)
    if pos < 0:
        raise RuntimeError(f"Failed to locate slot marker {slot_text!r} in rendered chat text.")
    return rendered_text[:pos], rendered_text[pos + len(slot_text):]


def text_to_ids(tokenizer, text: str) -> list:
    if not text:
        return []
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def ids_to_embeds(embedding_layer, token_ids, device, dtype) -> torch.Tensor:
    hidden_size = embedding_layer.weight.size(-1)
    if not token_ids:
        return torch.empty((0, hidden_size), dtype=dtype, device=device)
    token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    embeds = embedding_layer(token_tensor)[0]
    if embeds.dtype != dtype:
        embeds = embeds.to(dtype)
    return embeds


# ============================================================
# Stage building (как в оригинале outer/common.py)
# ============================================================
def build_stage_with_slot(
    tokenizer,
    embedding_layer,
    user_prompt_with_slot: str,
    assistant_text: str,
    slot_text: str,
    slot_embeds: torch.Tensor,
    enable_thinking: bool,
    device: torch.device,
    embed_dtype: torch.dtype,
    max_length: int,
) -> StagePack:
    """
    Создаёт tokenized input с slot injection.
    Полное воспроизведение оригинала: render_chat_text → split by slot → inject embeds.
    """
    prompt_rendered = render_chat_text(tokenizer, user_prompt_with_slot, assistant_text=None, enable_thinking=enable_thinking)
    full_rendered = render_chat_text(tokenizer, user_prompt_with_slot, assistant_text=assistant_text, enable_thinking=enable_thinking)

    prompt_prefix_text, prompt_suffix_text = split_rendered_text_by_slot(prompt_rendered, slot_text)
    full_prefix_text, full_suffix_text = split_rendered_text_by_slot(full_rendered, slot_text)

    prompt_prefix_ids = text_to_ids(tokenizer, prompt_prefix_text)
    prompt_suffix_ids = text_to_ids(tokenizer, prompt_suffix_text)
    full_prefix_ids = text_to_ids(tokenizer, full_prefix_text)
    full_suffix_ids = text_to_ids(tokenizer, full_suffix_text)

    slot_len = int(slot_embeds.size(0))
    prompt_len = len(prompt_prefix_ids) + slot_len + len(prompt_suffix_ids)

    token_ids = full_prefix_ids + ([-100] * slot_len) + full_suffix_ids
    truncate_left = 0
    if len(token_ids) > max_length:
        truncate_left = len(token_ids) - max_length
        token_ids = token_ids[truncate_left:]
        prompt_len = max(prompt_len - truncate_left, 0)

    labels = torch.full((len(token_ids),), -100, dtype=torch.long, device=device)
    for idx in range(prompt_len, len(token_ids)):
        tid = token_ids[idx]
        if tid >= 0:
            labels[idx] = tid
    assistant_mask = labels.ne(-100)

    prefix_embeds = ids_to_embeds(embedding_layer, full_prefix_ids, device=device, dtype=embed_dtype)
    suffix_embeds = ids_to_embeds(embedding_layer, full_suffix_ids, device=device, dtype=embed_dtype)

    if slot_embeds.dtype != embed_dtype:
        slot_embeds = slot_embeds.to(embed_dtype)

    seq_embeds = torch.cat([prefix_embeds, slot_embeds, suffix_embeds], dim=0)
    if truncate_left > 0:
        seq_embeds = seq_embeds[truncate_left:]
    attention_mask = torch.ones((seq_embeds.size(0),), dtype=torch.long, device=device)

    return StagePack(
        inputs_embeds=seq_embeds.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        labels=labels.unsqueeze(0),
        assistant_mask=assistant_mask,
    )


def build_planner_teacher_forced_inputs(
    tokenizer,
    question: str,
    plan: str,
    enable_thinking: bool,
    device: torch.device,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Создаёт input для Planner (первый round, teacher forcing)."""
    user_prompt = build_math_planner_prompt(question)
    prompt_ids = render_chat_ids(tokenizer, user_prompt, assistant_text=None, enable_thinking=enable_thinking, max_length=max_length)
    full_ids = render_chat_ids(tokenizer, user_prompt, assistant_text=plan, enable_thinking=enable_thinking, max_length=max_length)

    assistant_token_count = max(len(full_ids) - len(prompt_ids), 0)
    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]
        assistant_kept = min(assistant_token_count, len(full_ids))
        prompt_len = len(full_ids) - assistant_kept
    else:
        prompt_len = min(len(prompt_ids), len(full_ids))

    assistant_mask = torch.zeros((len(full_ids),), dtype=torch.bool, device=device)
    if prompt_len < len(full_ids):
        assistant_mask[prompt_len:] = True

    input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask, assistant_mask


def compute_solver_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """CE loss для solver output (как в оригинале)."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# ============================================================
# Dataset loading (как в оригинале outer/common.py)
# ============================================================
def load_outer_training_dataset(dataset_name: str, dataset_split: str, dataset_json_field=None):
    """Загружает датасет для outer training."""
    if os.path.isfile(dataset_name):
        suffix = os.path.splitext(dataset_name)[1].lower()
        if suffix not in {".json", ".jsonl"}:
            raise ValueError(f"Unsupported local dataset file type: {dataset_name}")
        kwargs = {"data_files": dataset_name}
        if suffix == ".json" and dataset_json_field is not None:
            kwargs["field"] = dataset_json_field
        return load_dataset("json", **kwargs, split="train")
    return load_dataset(dataset_name, split=dataset_split)


# ============================================================
# Checkpoint saving (как в оригинале outer/sequential.py)
# ============================================================
def save_recursive_outer_checkpoint(
    save_dir: str,
    step: Optional[int],
    outer_12: CrossModelAdapter,
    outer_23: CrossModelAdapter,
    outer_31: CrossModelAdapter,
    args,
) -> None:
    """Сохраняет checkpoint в формате оригинала."""
    output_dir = os.path.join(save_dir, f"checkpoint-{step}") if step is not None else save_dir
    os.makedirs(output_dir, exist_ok=True)

    torch.save(outer_12.state_dict(), os.path.join(output_dir, "outer_12.pt"))
    torch.save(outer_23.state_dict(), os.path.join(output_dir, "outer_23.pt"))
    torch.save(outer_31.state_dict(), os.path.join(output_dir, "outer_31.pt"))

    cfg = {
        "outer_12_type": outer_12.adapter_type,
        "outer_23_type": outer_23.adapter_type,
        "outer_31_type": outer_31.adapter_type,
        "outer_12_in_dim": outer_12.in_dim,
        "outer_12_out_dim": outer_12.out_dim,
        "outer_23_in_dim": outer_23.in_dim,
        "outer_23_out_dim": outer_23.out_dim,
        "outer_31_in_dim": outer_31.in_dim,
        "outer_31_out_dim": outer_31.out_dim,
        "mas_shape": "chain",
        "agent1_model_name_or_path": args.agent1_model,
        "agent2_model_name_or_path": args.agent2_model,
        "agent3_model_name_or_path": args.agent3_model,
        "agent1_inner_aligner_path": args.agent1_inner,
        "agent2_inner_aligner_path": args.agent2_inner,
        "agent3_inner_aligner_path": args.agent3_inner,
        "enable_thinking": args.enable_thinking,
        "mas_task": "math",
        "num_recursive_rounds": args.num_recursive_rounds,
        "supervise_final_only": args.supervise_final_only,
        "non_last_loss_weight": args.non_last_loss_weight,
    }
    with open(os.path.join(output_dir, "outer_adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    # outerlink_config.json (как в оригинале)
    manifest = {
        "format_version": 1,
        "paradigm": "sequential",
        "legacy_config_filename": "outer_adapter_config.json",
        "adapters": [
            {"legacy_key": "outer_12", "filename": "outer_12.pt", "adapter_type": outer_12.adapter_type, "in_dim": outer_12.in_dim, "out_dim": outer_12.out_dim},
            {"legacy_key": "outer_23", "filename": "outer_23.pt", "adapter_type": outer_23.adapter_type, "in_dim": outer_23.in_dim, "out_dim": outer_23.out_dim},
            {"legacy_key": "outer_31", "filename": "outer_31.pt", "adapter_type": outer_31.adapter_type, "in_dim": outer_31.in_dim, "out_dim": outer_31.out_dim},
        ],
    }
    with open(os.path.join(output_dir, "outerlink_config.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


# ============================================================
# Inner adapter loading (как в оригинале outer/common.py)
# ============================================================
def load_inner_adapter(
    adapter_path: str,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
    fallback_adapter_type: str = "ln_res_adapter",
) -> Adapter:
    """Загружает inner adapter из checkpoint."""
    if os.path.isdir(adapter_path):
        state_path = os.path.join(adapter_path, "adapter.pt")
        config_path = os.path.join(adapter_path, "adapter_config.json")
    else:
        state_path = adapter_path
        config_path = os.path.join(os.path.dirname(adapter_path), "adapter_config.json")

    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Inner aligner weights not found: {state_path}")

    adapter_type = fallback_adapter_type
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        adapter_type = cfg.get("adapter_type", adapter_type)

    adapter = Adapter(hidden_size=hidden_size, adapter_type=adapter_type)
    adapter.load_state_dict(torch.load(state_path, map_location="cpu"), strict=True)
    adapter.to(device=device, dtype=dtype)
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad = False
    return adapter


# ============================================================
# Model loading (как в оригинале outer/common.py)
# ============================================================
def load_model_and_tokenizer(
    model_name_or_path: str,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
    agent_name: str,
    gradient_checkpointing: bool = False,
):
    """Загружает модель и токенизатор."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(f"{agent_name} tokenizer has no apply_chat_template")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise RuntimeError(f"{agent_name} tokenizer has no pad/eos token")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                model.config.use_cache = False
        except Exception as exc:
            print(f"[warn] failed to enable gradient checkpointing for {agent_name}: {exc}")

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


def activate_gc_runtime(model: nn.Module, model_name: str) -> None:
    """
    Включает gradient checkpointing через model.train() + dropout=0.
    Как в оригинале: активирует checkpointing branches, отключает dropout.
    """
    model.train()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    dropout_count = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
            dropout_count += 1
    if dropout_count > 0:
        print(f"[info] {model_name}: disabled {dropout_count} dropout layers for gradient checkpointing")


# ============================================================
# Argument parsing (как в оригинале outer/sequential.py)
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Outer-Loop Training for RecursiveMAS (sequential)")
    parser.add_argument("--agent1_model", type=str, required=True, help="Planner model path")
    parser.add_argument("--agent2_model", type=str, required=True, help="Refiner model path")
    parser.add_argument("--agent3_model", type=str, required=True, help="Solver model path")

    parser.add_argument("--agent1_inner", type=str, required=True, help="Planner inner adapter path")
    parser.add_argument("--agent2_inner", type=str, required=True, help="Refiner inner adapter path")
    parser.add_argument("--agent3_inner", type=str, required=True, help="Solver inner adapter path")
    parser.add_argument("--inner_adapter_type_fallback", type=str, default="ln_res_adapter")

    parser.add_argument("--dataset_name", type=str, default="RecursiveMAS/Sequential-Math",
                        help="Training dataset. Default: RecursiveMAS/Sequential-Math (как в оригинале). "
                             "Поддерживает: RecursiveMAS/Sequential-Math, openai/gsm8k, локальные JSON/JSONL файлы.")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_json_field", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mas_task", type=str, default="math", choices=["math", "code"])
    parser.add_argument("--solver_pre_question", type=int, default=0)
    parser.add_argument("--enable_thinking", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gradient_checkpointing", type=int, default=1, choices=[0, 1])

    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_latent_tokens", type=int, default=80)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--outer_lr", type=float, default=5e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--num_recursive_rounds", type=int, default=3, help="Number of recursive rounds")
    parser.add_argument("--supervise_final_only", type=int, default=1, choices=[0, 1],
                        help="If 1, optimize only the final-round CE loss (default, как в оригинале)")
    parser.add_argument("--non_last_loss_weight", type=float, default=0.1,
                        help="When supervise_final_only=0, add this weight * mean(non-last round losses)")

    parser.add_argument("--outer_adapter_type", type=str, default="outer_ln_res_adapter",
                        choices=["outer_linear_adapter", "outer_linear_res_adapter", "outer_adapter",
                                 "outer_res_adapter", "outer_ln_adapter", "outer_ln_res_adapter"])
    parser.add_argument("--outer_12_type", type=str, default=None)
    parser.add_argument("--outer_23_type", type=str, default=None)
    parser.add_argument("--outer_31_type", type=str, default=None)

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--outer_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda:2")

    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--load_dir", type=str, default=None, help="Resume from checkpoint dir")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")

    return parser.parse_args()


# ============================================================
# Main training loop (как в оригинале outer/sequential.py)
# ============================================================
def resolve_dtype(dtype_str: str) -> Optional[torch.dtype]:
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return None


def main():
    args = parse_args()
    device = torch.device(args.device)
    model_dtype = resolve_dtype(args.dtype)
    outer_dtype = resolve_dtype(args.outer_dtype)

    if device.type == "cpu" and model_dtype in {torch.float16, torch.bfloat16}:
        print("[warn] CPU + fp16/bf16 is unstable. Falling back model dtype to float32.")
        model_dtype = torch.float32
    if device.type == "cpu" and outer_dtype in {torch.float16, torch.bfloat16}:
        print("[warn] CPU + fp16/bf16 is unstable. Falling back outer dtype to float32.")
        outer_dtype = torch.float32

    torch.manual_seed(args.seed)
    enable_thinking = bool(args.enable_thinking)

    # ── Load models ──────────────────────────────────────────
    print("=" * 60)
    print("OUTER-LOOP TRAINING: Sequential Style")
    print("=" * 60)
    print(f"Agent 1 (Planner):  {args.agent1_model}")
    print(f"Agent 2 (Refiner):  {args.agent2_model}")
    print(f"Agent 3 (Solver):   {args.agent3_model}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Recursive rounds: {args.num_recursive_rounds}")
    print(f"max_length: {args.max_length}, max_latent_tokens: {args.max_latent_tokens}")
    print(f"supervise_final_only: {args.supervise_final_only}")
    print(f"Outer adapter type: {args.outer_adapter_type}")
    print(f"Max steps: {args.max_steps}")
    print(f"LR: {args.outer_lr}")
    print()

    planner_model, planner_tok = load_model_and_tokenizer(
        args.agent1_model, device=device, dtype=model_dtype,
        trust_remote_code=args.trust_remote_code, agent_name="planner",
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    refiner_model, refiner_tok = load_model_and_tokenizer(
        args.agent2_model, device=device, dtype=model_dtype,
        trust_remote_code=args.trust_remote_code, agent_name="refiner",
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    solver_model, solver_tok = load_model_and_tokenizer(
        args.agent3_model, device=device, dtype=model_dtype,
        trust_remote_code=args.trust_remote_code, agent_name="solver",
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )

    if bool(args.gradient_checkpointing):
        activate_gc_runtime(planner_model, "planner")
        activate_gc_runtime(refiner_model, "refiner")
        activate_gc_runtime(solver_model, "solver")

    planner_embed = planner_model.get_input_embeddings()
    refiner_embed = refiner_model.get_input_embeddings()
    solver_embed = solver_model.get_input_embeddings()

    planner_hidden = planner_embed.weight.size(-1)
    refiner_hidden = refiner_embed.weight.size(-1)
    solver_hidden = solver_embed.weight.size(-1)

    print(f"Hidden sizes: planner={planner_hidden}, refiner={refiner_hidden}, solver={solver_hidden}")

    # ── Load inner adapters ──────────────────────────────────
    print("\n🔗 Loading inner adapters...")
    inner_1 = load_inner_adapter(args.agent1_inner, hidden_size=planner_hidden, device=device, dtype=model_dtype, fallback_adapter_type=args.inner_adapter_type_fallback)
    inner_2 = load_inner_adapter(args.agent2_inner, hidden_size=refiner_hidden, device=device, dtype=model_dtype, fallback_adapter_type=args.inner_adapter_type_fallback)
    inner_3 = load_inner_adapter(args.agent3_inner, hidden_size=solver_hidden, device=device, dtype=model_dtype, fallback_adapter_type=args.inner_adapter_type_fallback)

    # ── Create outer adapters ────────────────────────────────
    outer_12_type = args.outer_12_type or args.outer_adapter_type
    outer_23_type = args.outer_23_type or args.outer_adapter_type
    outer_31_type = args.outer_31_type or args.outer_adapter_type

    print(f"\n🔗 Creating outer adapters (type: {outer_12_type})...")
    outer_12 = CrossModelAdapter(planner_hidden, refiner_hidden, outer_12_type).to(device=device, dtype=outer_dtype)
    outer_23 = CrossModelAdapter(refiner_hidden, solver_hidden, outer_23_type).to(device=device, dtype=outer_dtype)
    outer_31 = CrossModelAdapter(solver_hidden, planner_hidden, outer_31_type).to(device=device, dtype=outer_dtype)
    outer_12.train()
    outer_23.train()
    outer_31.train()

    params = list(outer_12.parameters()) + list(outer_23.parameters()) + list(outer_31.parameters())
    trainable = sum(p.numel() for p in params)
    print(f"  Outer 12 (Planner→Refiner): {planner_hidden} → {refiner_hidden}")
    print(f"  Outer 23 (Refiner→Solver):  {refiner_hidden} → {solver_hidden}")
    print(f"  Outer 31 (Solver→Planner):  {solver_hidden} → {planner_hidden}")
    print(f"  Trainable params: {trainable:,}")

    optimizer = torch.optim.AdamW(params, lr=args.outer_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    # ── Load dataset ─────────────────────────────────────────
    print(f"\nLoading {args.dataset_name}...")
    dataset = load_outer_training_dataset(args.dataset_name, args.dataset_split, args.dataset_json_field)
    needed_cols = {"question", "plan", "refined_plan", "answer"}
    missing = needed_cols.difference(set(dataset.column_names))
    if missing:
        raise ValueError(f"Dataset missing required fields: {sorted(missing)}")
    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)
    if args.num_samples > 0:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    print(f"Dataset size: {len(dataset)}")

    rows = [
        {
            "question": sample.get("question", ""),
            "plan": sample.get("plan", ""),
            "refined_plan": sample.get("refined_plan", ""),
            "answer": sample.get("answer", ""),
            "type": sample.get("type", "complete"),
            "fn_name": sample.get("fn_name", None),
        }
        for sample in dataset
    ]

    dataloader = DataLoader(rows, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: x)
    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty. Increase dataset size or reduce batch_size.")

    # ── Scheduler ────────────────────────────────────────────
    steps_per_epoch = len(dataloader)
    max_train_steps = args.max_steps if args.max_steps > 0 else args.num_train_epochs * steps_per_epoch

    if args.lr_scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps,
            num_training_steps=max_train_steps, num_cycles=0.5,
        )
    else:
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    # ── Resume from checkpoint ───────────────────────────────
    global_step = 0
    if args.load_dir:
        ckpt_12 = os.path.join(args.load_dir, "outer_12.pt")
        if os.path.isfile(ckpt_12):
            outer_12.load_state_dict(torch.load(ckpt_12, map_location="cpu"))
            outer_23.load_state_dict(torch.load(os.path.join(args.load_dir, "outer_23.pt"), map_location="cpu"))
            outer_31.load_state_dict(torch.load(os.path.join(args.load_dir, "outer_31.pt"), map_location="cpu"))
            # Extract step from dir name
            dir_name = os.path.basename(args.load_dir)
            if dir_name.startswith("checkpoint-"):
                global_step = int(dir_name.split("-")[1])
            print(f"🔄 Resumed from {args.load_dir} at step={global_step}")

    # ── Training loop (как в оригинале outer/sequential.py) ──
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("STARTING OUTER-LOOP TRAINING")
    print(f"Recursive rounds: {args.num_recursive_rounds}")
    print(f"Max length: {args.max_length}")
    print(f"Max latent tokens: {args.max_latent_tokens}")
    print(f"Supervise final only: {args.supervise_final_only}")
    print(f"Starting from step: {global_step}")
    print("=" * 60)

    start_time = time.time()
    skipped_count = 0
    _accum_counter = 0  # gradient accumulation counter

    log_loss = 0.0
    log_r_first = 0.0
    log_r_last = 0.0
    log_count = 0

    while global_step < max_train_steps:
        for batch in dataloader:
            if global_step >= max_train_steps:
                break

            sample_r_first_losses: List[float] = []
            sample_r_last_losses: List[float] = []
            valid_count = 0
            batch_loss_sum = 0.0

            # Zero grads only at start of accumulation cycle
            if _accum_counter == 0:
                optimizer.zero_grad(set_to_none=True)

            for sample in batch:
                q = str(sample["question"]).strip()
                p = str(sample["plan"]).strip()
                rp = str(sample["refined_plan"]).strip()
                ans = str(sample["answer"]).strip()
                task_type = str(sample.get("type", "complete")).strip().lower() or "complete"
                fn_name = sample.get("fn_name", None)

                if not q or not p or not rp or not ans:
                    skipped_count += 1
                    continue

                try:
                    feedback_to_planner = None
                    round_losses: List[torch.Tensor] = []

                    for round_idx in range(args.num_recursive_rounds):
                        # ── Stage 1: Planner ──────────────────
                        if round_idx == 0:
                            planner_input_ids, planner_attention_mask, planner_assist_mask = (
                                build_planner_teacher_forced_inputs(
                                    planner_tok, question=q, plan=p,
                                    enable_thinking=enable_thinking, device=device,
                                    max_length=args.max_length,
                                )
                            )
                            with torch.no_grad():
                                planner_out = planner_model(
                                    input_ids=planner_input_ids,
                                    attention_mask=planner_attention_mask,
                                    output_hidden_states=True,
                                    use_cache=False,
                                    return_dict=True,
                                )
                            planner_hidden = planner_out.hidden_states[-1][0][planner_assist_mask]
                        else:
                            if feedback_to_planner is None or feedback_to_planner.size(0) == 0:
                                round_losses = []
                                break

                            planner_user_with_slot = build_math_planner_prompt_with_feedback_slot(q)
                            planner_pack = build_stage_with_slot(
                                tokenizer=planner_tok,
                                embedding_layer=planner_embed,
                                user_prompt_with_slot=planner_user_with_slot,
                                assistant_text=p,
                                slot_text=FEEDBACK_SLOT,
                                slot_embeds=feedback_to_planner,
                                enable_thinking=enable_thinking,
                                device=device,
                                embed_dtype=planner_embed.weight.dtype,
                                max_length=args.max_length,
                            )
                            planner_out = planner_model(
                                inputs_embeds=planner_pack.inputs_embeds,
                                attention_mask=planner_pack.attention_mask,
                                output_hidden_states=True,
                                use_cache=False,
                                return_dict=True,
                            )
                            planner_hidden = planner_out.hidden_states[-1][0][planner_pack.assistant_mask]

                        if planner_hidden.size(0) == 0:
                            round_losses = []
                            break

                        planner_inner = run_inner_adapter_preserve_input_grad(inner_1, planner_hidden, out_dtype=model_dtype)
                        planner_to_refiner = run_outer_adapter(outer_12, planner_inner, out_dtype=refiner_embed.weight.dtype)
                        planner_to_refiner = trim_latent(planner_to_refiner, args.max_latent_tokens)

                        # ── Stage 2: Refiner ──────────────────
                        refiner_user_with_slot = build_math_refiner_prompt_with_slot(q)
                        refiner_pack = build_stage_with_slot(
                            tokenizer=refiner_tok,
                            embedding_layer=refiner_embed,
                            user_prompt_with_slot=refiner_user_with_slot,
                            assistant_text=rp,
                            slot_text=PLANNER_SLOT,
                            slot_embeds=planner_to_refiner,
                            enable_thinking=enable_thinking,
                            device=device,
                            embed_dtype=refiner_embed.weight.dtype,
                            max_length=args.max_length,
                        )
                        refiner_out = refiner_model(
                            inputs_embeds=refiner_pack.inputs_embeds,
                            attention_mask=refiner_pack.attention_mask,
                            output_hidden_states=True,
                            use_cache=False,
                            return_dict=True,
                        )
                        refiner_hidden = refiner_out.hidden_states[-1][0][refiner_pack.assistant_mask]
                        if refiner_hidden.size(0) == 0:
                            round_losses = []
                            break

                        refiner_inner = run_inner_adapter_preserve_input_grad(inner_2, refiner_hidden, out_dtype=model_dtype)
                        refiner_to_solver = run_outer_adapter(outer_23, refiner_inner, out_dtype=solver_embed.weight.dtype)
                        refiner_to_solver = trim_latent(refiner_to_solver, args.max_latent_tokens)

                        # ── Stage 3: Solver ───────────────────
                        solver_user_with_slot = build_math_solver_prompt_with_slots(q)
                        solver_pack = build_stage_with_slot(
                            tokenizer=solver_tok,
                            embedding_layer=solver_embed,
                            user_prompt_with_slot=solver_user_with_slot,
                            assistant_text=ans,
                            slot_text=REFINED_SLOT,
                            slot_embeds=refiner_to_solver,
                            enable_thinking=enable_thinking,
                            device=device,
                            embed_dtype=solver_embed.weight.dtype,
                            max_length=args.max_length,
                        )
                        need_feedback = round_idx < args.num_recursive_rounds - 1
                        solver_out = solver_model(
                            inputs_embeds=solver_pack.inputs_embeds,
                            attention_mask=solver_pack.attention_mask,
                            output_hidden_states=need_feedback,
                            use_cache=False,
                            return_dict=True,
                        )
                        loss_round = compute_solver_ce_loss(solver_out.logits, solver_pack.labels)
                        if torch.isnan(loss_round) or torch.isinf(loss_round):
                            round_losses = []
                            break
                        round_losses.append(loss_round)

                        # Feedback for next round (solver → planner)
                        if need_feedback:
                            solver_hidden = solver_out.hidden_states[-1][0][solver_pack.assistant_mask]
                            if solver_hidden.size(0) == 0:
                                round_losses = []
                                break
                            solver_inner = run_inner_adapter_preserve_input_grad(inner_3, solver_hidden, out_dtype=model_dtype)
                            feedback_to_planner = run_outer_adapter(outer_31, solver_inner, out_dtype=planner_embed.weight.dtype)
                            feedback_to_planner = trim_latent(feedback_to_planner, args.max_latent_tokens)

                    if not round_losses:
                        skipped_count += 1
                        continue

                    # Loss computation (как в оригинале)
                    if args.supervise_final_only:
                        loss = round_losses[-1]
                    else:
                        if len(round_losses) > 1:
                            non_last_mean = torch.stack(round_losses[:-1]).mean()
                            loss = round_losses[-1] + args.non_last_loss_weight * non_last_mean
                        else:
                            loss = round_losses[-1]

                    # Gradient accumulation: divide loss by grad_accum_steps
                    (loss / (max(args.batch_size, 1) * args.grad_accum_steps)).backward()
                    valid_count += 1
                    batch_loss_sum += float(loss.item())
                    sample_r_first_losses.append(float(round_losses[0].item()))
                    sample_r_last_losses.append(float(round_losses[-1].item()))

                except RuntimeError as exc:
                    exc_msg = str(exc).lower()
                    if "sequence_too_long" in exc_msg:
                        skipped_count += 1
                        continue
                    raise

            if valid_count == 0:
                continue

            # Gradient scaling for skipped samples (как в оригинале)
            if valid_count != args.batch_size:
                grad_scale = args.batch_size / valid_count
                for param in params:
                    if param.grad is not None:
                        param.grad.mul_(grad_scale)

            # Gradient accumulation: only step every grad_accum_steps batches
            _accum_counter += 1
            if _accum_counter >= args.grad_accum_steps:
                _accum_counter = 0
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # Logging — only after optimizer step
                loss_batch = batch_loss_sum / valid_count
                log_loss += float(loss_batch)
                log_r_first += sum(sample_r_first_losses) / max(len(sample_r_first_losses), 1)
                log_r_last += sum(sample_r_last_losses) / max(len(sample_r_last_losses), 1)
                log_count += 1

                if global_step % args.log_every == 0:
                    avg_loss = log_loss / max(log_count, 1)
                    avg_r_first = log_r_first / max(log_count, 1)
                    avg_r_last = log_r_last / max(log_count, 1)
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    print(f"step={global_step} loss={avg_loss:.4f} r_first={avg_r_first:.4f} r_last={avg_r_last:.4f} lr={lr:.6f} time={elapsed:.0f}s skipped={skipped_count}", flush=True)
                    log_loss = 0.0
                    log_r_first = 0.0
                    log_r_last = 0.0
                    log_count = 0

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_recursive_outer_checkpoint(args.save_dir, global_step, outer_12, outer_23, outer_31, args)

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    # Final checkpoint
    save_recursive_outer_checkpoint(args.save_dir, None, outer_12, outer_23, outer_31, args)
    elapsed = time.time() - start_time
    print(f"\n🎉 Outer training complete! steps={global_step}, time={elapsed:.0f}s, skipped={skipped_count}")


if __name__ == "__main__":
    main()
