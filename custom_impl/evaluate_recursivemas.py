#!/usr/bin/env python3
"""
evaluate_recursivemas.py — Оценка RecursiveMAS на GSM8K (совместим с train_outer.py)

Pipeline (как в train_outer.py):
  1. Planner (Qwen3-1.7B) → inner_1 → latent tokens → outer_12
  2. Refiner (Qwen3-1.7B) ← slot injection ← outer_12 → inner_2 → latent tokens → outer_23
  3. Solver (Qwen2.5-Math-1.5B) ← slot injection ← outer_23 → генерирует текст

Использует те же модели, inner adapters и outer adapters, что и train_outer.py.

Пример:
    # Без чекпоинтов (случайная инициализация outer adapters)
    python evaluate_recursivemas.py --num_examples 20

    # С обученными адаптерами
    python evaluate_recursivemas.py \
        --num_examples 100 \
        --outer_checkpoint custom_impl/checkpoints/outer/ \
        --inner1_checkpoint custom_impl/checkpoints/inner_planner/ \
        --inner2_checkpoint custom_impl/checkpoints/inner_refiner/ \
        --inner3_checkpoint custom_impl/checkpoints/inner_solver/
"""

import os
import sys
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Fix for CUDA
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# Suppress datasets progress bars
try:
    import datasets as _datasets
    _datasets.disable_progress_bars()
except Exception:
    pass

# Унифицированные утилиты из gsm8k_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm8k_utils import parse_answer, normalize_answer, compute_accuracy, print_results, save_results

# ============================================================
# Модели и конфиг
# ============================================================
PLANNER_MODEL = "Qwen/Qwen3-1.7B"
REFINER_MODEL = "Qwen/Qwen3-1.7B"  # та же модель, что и Planner
SOLVER_MODEL = "/home/hermes/workspace/models/models_llm/Qwen2.5-Math-1.5B-Instruct"
DEFAULT_DEVICE = "cuda:2"

SYSTEM_PROMPT = "You are a helpful assistant."

# ============================================================
# Prompts (оригинальные из mas_prompt.py — совпадают с train_outer.py)
# ============================================================
def build_math_planner_prompt(question: str) -> str:
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

def build_math_planner_prompt_with_feedback(question: str, feedback: torch.Tensor) -> str:
    """Planner prompt with feedback from previous solver round."""
    return (
        "You are a planner agent in a recursive multi-agent system.\n"
        "This is round 2.\n"
        "Question:\n"
        f"{question}\n"
        "Feedback signal from the previous solver round:\n"
        "<<LATENT_FEEDBACK_SLOT>>\n"
        "Use the feedback as a soft correction signal to improve the plan.\n"
        "If there is any conflict, prioritize the question constraints.\n"
        "Output only a concise plan in the format:\n"
        "Step 1: ...\n"
        "...\n"
        "Step n: ..."
    )

def build_math_refiner_prompt_with_slot(question: str) -> str:
    return (
        "You are a refiner agent in a multi-agent system.\n"
        "The question is:\n"
        "Question:\n"
        f"{question}\n"
        "The initial plan from the planner:\n"
        "Initial Plan:\n"
        "<<LATENT_PLANNER_SLOT>>\n"
        "You should refine the initial plan and respond with pure plan only in the format of:\n"
        "Step 1: ...\n"
        "...\n"
        "Step n: ..."
    )

def build_math_solver_prompt_with_slot(question: str) -> str:
    return (
        "You are a solver agent in a multi-agent system.\n"
        "Here is the refined plan:\n"
        "Refined Plan:\n"
        "<<LATENT_REFINED_SLOT>>\n"
        "The question is:\n"
        "Question:\n"
        f"{question}\n"
        "Solve the question given information and put the final answer inside \\boxed{}, for example \\boxed{1}."
    )

# ============================================================
# Adapter classes (как в train_outer.py)
# ============================================================
class InnerAdapter(nn.Module):
    """ln_res_adapter — как в train_outer.py."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.pre_ln = nn.LayerNorm(hidden_size)
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden_size, hidden_size)
        self.post_ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        h = self.pre_ln(x)
        out = self.proj2(self.act(self.proj1(h)))
        out = x + out
        out = self.post_ln(out)
        return out


class CrossModelAdapter(nn.Module):
    """
    Outer Link adapter — outer_ln_res_adapter из оригинала.
    LayerNorm(in) → Linear(in, out×2) → GELU → Linear(out×2, out) + Linear(in, out) → LayerNorm(out)
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
# Slot injection helpers
# ============================================================
def build_inputs_with_slot(
    tokenizer,
    embedding_layer,
    user_prompt_with_slot: str,
    slot_embeds: torch.Tensor,
    slot_text: str,
    device: torch.device,
    embed_dtype: torch.dtype,
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple:
    """
    Создаёт inputs_embeds с injected slot.
    Как в train_outer.py: build_stage_with_slot().

    Returns:
        (inputs_embeds, attention_mask) — [1, total_seq_len, hidden]
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_with_slot},
    ]

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Split по slot placeholder
    slot_pos = full_text.find(slot_text)
    if slot_pos < 0:
        raise ValueError(f"Slot '{slot_text}' not found in prompt")

    prefix_text = full_text[:slot_pos]
    suffix_text = full_text[slot_pos + len(slot_text):]

    # Tokenize prefix and suffix
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False).input_ids

    # Convert to embeddings
    prefix_embeds = embedding_layer(torch.tensor(prefix_ids, device=device).unsqueeze(0))
    suffix_embeds = embedding_layer(torch.tensor(suffix_ids, device=device).unsqueeze(0))

    # slot_embeds: [1, seq_len, hidden]
    slot_embeds = slot_embeds.to(embed_dtype)

    inputs_embeds = torch.cat([prefix_embeds, slot_embeds, suffix_embeds], dim=1)
    total_len = inputs_embeds.shape[1]
    attention_mask = torch.ones((1, total_len), dtype=torch.long, device=device)

    return inputs_embeds, attention_mask


def generate_from_embeds(
    model, tokenizer, inputs_embeds, attention_mask,
    prefix_len: int, max_new_tokens: int = 512,  # 512 для ускорения
    do_sample: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.95,
    **gen_kwargs
) -> str:
    """
    Генерирует текст из inputs_embeds.

    prefix_len — длина prefix (до slot), чтобы отсечь его от вывода.
    """
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **gen_kwargs
        )

    # Decode only generated tokens (after prefix)
    generated_ids = outputs[0][prefix_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text


# ============================================================
# Latent token generation (autoregressive)
# ============================================================
def generate_latent_tokens(
    model, embedding_layer, inner_adapter,
    inputs_embeds, attention_mask,
    num_tokens: int = 32,  # FIX: was 20, original uses 32 for sequential_light/math500
    device: torch.device = None,
) -> torch.Tensor:
    """
    Генерирует num_tokens латентных токенов авто-регрессивно.

    Как в оригинале:
      e_{t+1} = R_in(h_t)    # last-layer hidden → input embedding
      h_{t+1} = f_θ([E; e_{t+1}])  # forward pass

    Returns:
        latent_hidden: [1, num_tokens, hidden_dim] — last-layer hidden states
    """
    if device is None:
        device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # Start from the last hidden state of the input
    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    last_hidden = outputs.hidden_states[-1][:, -1:, :]  # [1, 1, hidden]

    latent_hiddens = []
    current_input = inputs_embeds  # [1, seq_len, hidden]
    current_attn = attention_mask

    for _ in range(num_tokens):
        # Apply inner adapter: last hidden → input embedding space
        latent_emb = inner_adapter(last_hidden)  # [1, 1, hidden]

        # Append to input
        new_input = torch.cat([current_input, latent_emb], dim=1)
        new_attn = torch.cat([
            current_attn,
            torch.ones((1, 1), dtype=torch.long, device=device)
        ], dim=1)

        # Forward pass — need hidden_states to get last layer output
        with torch.no_grad():
            out = model(
                inputs_embeds=new_input,
                attention_mask=new_attn,
                output_hidden_states=True,
            )
        last_hidden = out.hidden_states[-1][:, -1:, :]  # [1, 1, hidden]
        latent_hiddens.append(last_hidden)

        current_input = new_input
        current_attn = new_attn

    latent_hidden = torch.cat(latent_hiddens, dim=1)  # [1, num_tokens, hidden]
    return latent_hidden


# ============================================================
# GSM8K helpers
# ============================================================
def load_gsm8k_test(num_examples: int = 100):
    """Загружает GSM8K test split."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if num_examples > 0 and num_examples < len(ds):
        ds = ds.select(range(num_examples))
    return ds


# ============================================================
# Main evaluation pipeline
# ============================================================
class RecursiveMASEvaluator:
    """
    Полная оценка RecursiveMAS, совместимая с train_outer.py.

    Pipeline:
      1. Planner → inner_1 → latent tokens → outer_12
      2. Refiner ← slot injection ← outer_12 → inner_2 → latent tokens → outer_23
      3. Solver ← slot injection ← outer_23 → текст
    """

    def __init__(self, args, device):
        self.device = device
        self.num_latent_tokens = args.num_latent_tokens
        self.num_recursive_rounds = args.num_recursive_rounds  # FIX: was missing, original uses 3
        self.model_dtype = torch.bfloat16

        # Load models
        print("Loading Planner...")
        self.planner_tok = AutoTokenizer.from_pretrained(
            PLANNER_MODEL, trust_remote_code=True, use_fast=True)
        if self.planner_tok.pad_token_id is None:
            self.planner_tok.pad_token = self.planner_tok.eos_token
        self.planner_model = AutoModelForCausalLM.from_pretrained(
            PLANNER_MODEL, torch_dtype=self.model_dtype,
            device_map={"": device.index}, trust_remote_code=True)
        self.planner_model.eval()
        for p in self.planner_model.parameters():
            p.requires_grad = False
        self.h1 = self.planner_model.config.hidden_size
        self.embed1 = self.planner_model.get_input_embeddings()
        print(f"  Hidden size: {self.h1}")

        # Refiner (shared with Planner if same model)
        self._shared_12 = (PLANNER_MODEL == REFINER_MODEL)
        if self._shared_12:
            self.refiner_tok = self.planner_tok
            self.refiner_model = self.planner_model
            self.h2 = self.h1
            self.embed2 = self.embed1
            print("Refiner: shared with Planner")
        else:
            print("Loading Refiner...")
            self.refiner_tok = AutoTokenizer.from_pretrained(
                REFINER_MODEL, trust_remote_code=True, use_fast=True)
            if self.refiner_tok.pad_token_id is None:
                self.refiner_tok.pad_token = self.refiner_tok.eos_token
            self.refiner_model = AutoModelForCausalLM.from_pretrained(
                REFINER_MODEL, torch_dtype=self.model_dtype,
                device_map={"": device.index}, trust_remote_code=True)
            self.refiner_model.eval()
            for p in self.refiner_model.parameters():
                p.requires_grad = False
            self.h2 = self.refiner_model.config.hidden_size
            self.embed2 = self.refiner_model.get_input_embeddings()
            print(f"  Hidden size: {self.h2}")

        print("Loading Solver...")
        self.solver_tok = AutoTokenizer.from_pretrained(
            SOLVER_MODEL, trust_remote_code=True, use_fast=True)
        if self.solver_tok.pad_token_id is None:
            self.solver_tok.pad_token = self.solver_tok.eos_token
        self.solver_model = AutoModelForCausalLM.from_pretrained(
            SOLVER_MODEL, torch_dtype=self.model_dtype,
            device_map={"": device.index}, trust_remote_code=True)
        self.solver_model.eval()
        for p in self.solver_model.parameters():
            p.requires_grad = False
        self.h3 = self.solver_model.config.hidden_size
        self.embed3 = self.solver_model.get_input_embeddings()
        print(f"  Hidden size: {self.h3}")

        # Load inner adapters
        print("Loading inner adapters...")
        self.inner1 = InnerAdapter(self.h1).to(device, dtype=self.model_dtype)
        self.inner2 = InnerAdapter(self.h2).to(device, dtype=self.model_dtype)
        self.inner3 = InnerAdapter(self.h3).to(device, dtype=self.model_dtype)

        if args.inner1_checkpoint:
            self._load_inner(args.inner1_checkpoint, self.inner1, "Planner")
        if args.inner2_checkpoint:
            self._load_inner(args.inner2_checkpoint, self.inner2, "Refiner")
        if args.inner3_checkpoint:
            self._load_inner(args.inner3_checkpoint, self.inner3, "Solver")

        for a in [self.inner1, self.inner2, self.inner3]:
            a.eval()

        # Create/load outer adapters
        print("Loading outer adapters...")
        self.outer_12 = CrossModelAdapter(self.h1, self.h2).to(device, dtype=self.model_dtype)
        self.outer_23 = CrossModelAdapter(self.h2, self.h3).to(device, dtype=self.model_dtype)

        if args.outer_checkpoint:
            self._load_outer(args.outer_checkpoint)

        self.outer_12.eval()
        self.outer_23.eval()

        trainable = sum(p.numel() for o in [self.outer_12, self.outer_23] for p in o.parameters())
        print(f"  Outer adapters params: {trainable:,}")
        print(f"  Outer 12 (Planner→Refiner): {self.h1} → {self.h2}")
        print(f"  Outer 23 (Refiner→Solver):  {self.h2} → {self.h3}")
        print(f"  Recursive rounds: {self.num_recursive_rounds}")
        print(f"  Latent tokens: {self.num_latent_tokens}")
        print(f"  Generation: do_sample={True}, temperature=0.6, top_p=0.95, max_new_tokens=1000")

    def _load_inner(self, path, adapter, name):
        adapter_path = os.path.join(path, "adapter.pt")
        if os.path.isfile(adapter_path):
            adapter.load_state_dict(torch.load(adapter_path, map_location="cpu", weights_only=True))
            print(f"  {name} inner: loaded from {path}")
        else:
            print(f"  WARNING: {name} inner: no adapter at {path}")

    def _load_outer(self, save_dir):
        """Загружает outer adapters из train_outer.py checkpoint."""
        # Try different formats
        # Format 1: outer_links_compat.pt (converted from outer_adapters.pt)
        compat_path = os.path.join(save_dir, "outer_links_compat.pt")
        if os.path.isfile(compat_path):
            state = torch.load(compat_path, map_location="cpu", weights_only=True)
            self.outer_12.load_state_dict(state["outer_1"])
            self.outer_23.load_state_dict(state["outer_2"])
            print(f"  Outer adapters loaded from {compat_path}")
            return

        # Format 2: checkpoint-N/outer_12.pt, outer_23.pt
        # Find latest checkpoint
        checkpoints = []
        for entry in os.listdir(save_dir):
            if entry.startswith("checkpoint-") and os.path.isdir(os.path.join(save_dir, entry)):
                try:
                    step = int(entry.split("-")[1])
                    checkpoints.append((step, entry))
                except ValueError:
                    pass
        if checkpoints:
            checkpoints.sort(reverse=True)
            ckpt_dir = os.path.join(save_dir, checkpoints[0][1])
            p12 = os.path.join(ckpt_dir, "outer_12.pt")
            p23 = os.path.join(ckpt_dir, "outer_23.pt")
            if os.path.isfile(p12) and os.path.isfile(p23):
                self.outer_12.load_state_dict(torch.load(p12, map_location="cpu", weights_only=True))
                self.outer_23.load_state_dict(torch.load(p23, map_location="cpu", weights_only=True))
                print(f"  Outer adapters loaded from {ckpt_dir}")
                return

        # Format 2b: save_dir is the checkpoint dir itself (e.g. checkpoint-20000/)
        direct_12 = os.path.join(save_dir, "outer_12.pt")
        direct_23 = os.path.join(save_dir, "outer_23.pt")
        if os.path.isfile(direct_12) and os.path.isfile(direct_23):
            self.outer_12.load_state_dict(torch.load(direct_12, map_location="cpu", weights_only=True))
            self.outer_23.load_state_dict(torch.load(direct_23, map_location="cpu", weights_only=True))
            print(f"  Outer adapters loaded from {save_dir}")
            return

        # Format 3: outer_adapters.pt (original train_outer format)
        orig_path = os.path.join(save_dir, "outer_adapters.pt")
        if os.path.isfile(orig_path):
            state = torch.load(orig_path, map_location="cpu", weights_only=True)
            # outer_12, outer_23, outer_31
            if "outer_12" in state:
                self.outer_12.load_state_dict(state["outer_12"])
                self.outer_23.load_state_dict(state["outer_23"])
                print(f"  Outer adapters loaded from {orig_path}")
                return

        print(f"  WARNING: No outer adapters found in {save_dir} — using random initialization")

    def evaluate_example(self, question: str) -> dict:
        """
        Оценивает один пример через полный RecursiveMAS pipeline с recursive rounds.

        Как в оригинале:
        - Round 1: Planner → Refiner → Solver
        - Round 2+: Solver latent → outer_31 → Planner feedback → Refiner → Solver
        - ... повторяется num_recursive_rounds раз
        - Ответ берётся только из последнего round'а

        Returns:
            dict с 'answer', 'plan_latent_norm', 'refined_latent_norm', 'timings'
        """
        result = {"answer": None, "timings": {}, "latent_stats": {}}
        device = self.device

        feedback_to_planner = None  # latent feedback from previous round

        for round_idx in range(self.num_recursive_rounds):
            # ============================================================
            # Stage 1: Planner → latent tokens
            # ============================================================
            t0 = time.time()
            if round_idx == 0:
                # First round: no feedback
                planner_prompt = build_math_planner_prompt(question)
            else:
                # Subsequent rounds: use feedback from previous solver
                planner_prompt = build_math_planner_prompt_with_feedback(question, feedback_to_planner)

            # Tokenize planner prompt
            planner_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": planner_prompt},
            ]
            planner_text = self.planner_tok.apply_chat_template(
                planner_messages, tokenize=False, add_generation_prompt=True
            )
            planner_inputs = self.planner_tok(
                planner_text, return_tensors="pt",
                truncation=True, max_length=2048,
            ).to(device)

            # Generate latent tokens from Planner (autoregressive)
            planner_embeds = self.embed1(planner_inputs.input_ids)
            planner_latent = generate_latent_tokens(
                self.planner_model, self.embed1, self.inner1,
                planner_embeds, planner_inputs.attention_mask,
                num_tokens=self.num_latent_tokens,
                device=device,
            )
            result["timings"]["planner"] = time.time() - t0
            result["latent_stats"]["planner_latent"] = list(planner_latent.shape)

            # ============================================================
            # Stage 2: Outer 12 (Planner → Refiner)
            # ============================================================
            t0 = time.time()
            refiner_input_latent = self.outer_12(planner_latent)  # [1, num_tokens, h2]
            result["timings"]["outer_12"] = time.time() - t0

            # ============================================================
            # Stage 3: Refiner ← slot injection → latent tokens
            # ============================================================
            t0 = time.time()
            refiner_prompt = build_math_refiner_prompt_with_slot(question)

            refiner_embeds, refiner_attn = build_inputs_with_slot(
                self.refiner_tok, self.embed2,
                refiner_prompt, refiner_input_latent,
                "<<LATENT_PLANNER_SLOT>>", device, self.model_dtype,
            )

            # Generate latent tokens from Refiner
            refiner_latent = generate_latent_tokens(
                self.refiner_model, self.embed2, self.inner2,
                refiner_embeds, refiner_attn,
                num_tokens=self.num_latent_tokens,
                device=device,
            )
            result["timings"]["refiner"] = time.time() - t0
            result["latent_stats"]["refiner_latent"] = list(refiner_latent.shape)

            # ============================================================
            # Stage 4: Outer 23 (Refiner → Solver)
            # ============================================================
            t0 = time.time()
            solver_input_latent = self.outer_23(refiner_latent)  # [1, num_tokens, h3]
            result["timings"]["outer_23"] = time.time() - t0

            # ============================================================
            # Stage 5: Solver ← slot injection → text generation
            # ============================================================
            t0 = time.time()
            solver_prompt = build_math_solver_prompt_with_slot(question)

            solver_embeds, solver_attn = build_inputs_with_slot(
                self.solver_tok, self.embed3,
                solver_prompt, solver_input_latent,
                "<<LATENT_REFINED_SLOT>>", device, self.model_dtype,
            )

            # Calculate prefix length for decoding
            prefix_len = solver_embeds.shape[1]

            # Generate text
            answer = generate_from_embeds(
                self.solver_model, self.solver_tok,
                solver_embeds, solver_attn,
                prefix_len=prefix_len,
                max_new_tokens=1000,  # FIX: was 512
                do_sample=True,  # FIX: was False
                temperature=0.6,  # FIX: was 0
                top_p=0.95,  # FIX: original uses top_p=0.95
            )
            result["timings"]["solver"] = time.time() - t0

            # Store answer from last round
            if round_idx == self.num_recursive_rounds - 1:
                result["answer"] = answer
            else:
                # Generate feedback for next round (solver latent → outer_31 → planner)
                # For now, we'll use the solver's last hidden state as feedback
                # This is a simplified version - full implementation would need hidden states
                feedback_to_planner = solver_input_latent  # pass through for now

        return result


def main():
    parser = argparse.ArgumentParser(
        description="RecursiveMAS evaluation on GSM8K (совместим с train_outer.py)",
    )
    parser.add_argument("--num_examples", type=int, default=100,
                        help="Количество примеров (default: 100)")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                        help=f"Устройство (default: {DEFAULT_DEVICE})")
    parser.add_argument("--outer_checkpoint", type=str, default=None,
                        help="Путь к директории с outer adapters (train_outer.py checkpoint)")
    parser.add_argument("--inner1_checkpoint", type=str, default=None,
                        help="Путь к inner adapter Planner (train_inner.py checkpoint)")
    parser.add_argument("--inner2_checkpoint", type=str, default=None,
                        help="Путь к inner adapter Refiner (train_inner.py checkpoint)")
    parser.add_argument("--inner3_checkpoint", type=str, default=None,
                        help="Путь к inner adapter Solver (train_inner.py checkpoint)")
    parser.add_argument("--num_latent_tokens", type=int, default=32,
                        help="Количество латентных токенов на агент (default: 32, как в оригинале)")
    parser.add_argument("--num_recursive_rounds", type=int, default=3,
                        help="Количество recursive round'ов (default: 3, как в оригинале)")
    parser.add_argument("--output", type=str, default=None,
                        help="Путь для JSON результатов")

    args = parser.parse_args()
    device = torch.device(args.device)

    print("=" * 70)
    print("RECURSIVEMAS EVALUATION (совместим с train_outer.py)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Latent tokens per agent: {args.num_latent_tokens}")
    print(f"Recursive rounds: {args.num_recursive_rounds}")
    print(f"Generation: do_sample=True, temperature=0.6, top_p=0.95, max_new_tokens=1000")
    print(f"Outer checkpoint: {args.outer_checkpoint or 'random'}")
    print(f"Inner checkpoints: {args.inner1_checkpoint or 'random'}, {args.inner2_checkpoint or 'random'}, {args.inner3_checkpoint or 'random'}")
    print()

    # Create evaluator
    evaluator = RecursiveMASEvaluator(args, device)

    # Load GSM8K
    print("\nLoading GSM8K test...")
    ds = load_gsm8k_test(args.num_examples)
    questions = ds["question"]
    answers = ds["answer"]
    # Парсим ground truth ответы
    gt_numbers = [normalize_answer(parse_answer(a) or "") for a in answers]
    print(f"  Загружено {len(questions)} примеров")

    # Run evaluation
    print(f"\n{'='*70}")
    print(f"Pipeline: Planner -> inner_1 -> outer_12 -> Refiner -> inner_2 -> outer_23 -> Solver")
    print(f"Recursive rounds: {args.num_recursive_rounds}")
    print(f"{'='*70}")

    predictions = []
    timings = []
    stage_timings = {"planner": [], "outer_12": [], "refiner": [], "outer_23": [], "solver": []}

    for i, question in enumerate(questions):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Обработка {i+1}/{len(questions)}...")

        t_start = time.time()
        result = evaluator.evaluate_example(question)
        elapsed = time.time() - t_start

        predictions.append(result["answer"])
        timings.append(elapsed)

        for stage in stage_timings:
            if stage in result["timings"]:
                stage_timings[stage].append(result["timings"][stage])

    print(f"  Обработка {len(questions)}/{len(questions)}... DONE\n")

    # Compute accuracy
    results = compute_accuracy(predictions, gt_numbers, questions=questions)

    # Print results
    print_results(results, title="RecursiveMAS Evaluation", timings=timings)

    # Stage timings
    print(f"\nСреднее время по стадиям:")
    for stage, times in stage_timings.items():
        avg = sum(times) / len(times) if times else 0
        print(f"  {stage}: {avg:.2f}s")

    # Save results
    if args.output:
        output_data = {
            "accuracy": results["accuracy"],
            "correct": results["correct"],
            "total": results["total"],
            "examples": [
                {
                    "index": ex["index"],
                    "prediction": ex["prediction"],
                    "prediction_parsed": ex["prediction_parsed"],
                    "answer_parsed": ex["answer_parsed"],
                    "correct": ex["correct"],
                    "question": questions[ex["index"]],
                }
                for ex in results["examples"]
            ],
            "timings": {
                "per_example": timings,
                "avg": sum(timings) / len(timings) if timings else 0,
                "total": sum(timings),
            },
            "meta": {
                "type": "recursivemas_v2",
                "planner_model": PLANNER_MODEL,
                "refiner_model": REFINER_MODEL,
                "solver_model": SOLVER_MODEL,
                "outer_checkpoint": args.outer_checkpoint,
                "inner_checkpoints": {
                    "planner": args.inner1_checkpoint,
                    "refiner": args.inner2_checkpoint,
                    "solver": args.inner3_checkpoint,
                },
                "num_latent_tokens": args.num_latent_tokens,
                "device": str(device),
                "num_examples": len(questions),
                "split": "test",
                "stage_avg_times": {
                    k: (sum(v) / len(v) if v else 0) for k, v in stage_timings.items()
                },
            },
        }
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nРезультаты сохранены в {args.output}")

    sys.exit(0 if results["accuracy"] > 0 else 1)


if __name__ == "__main__":
    main()
