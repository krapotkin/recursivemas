#!/usr/bin/env python3
"""
train_outer_link_gsm8k.py — Обучение Outer Link адаптеров на GSM8K.

Архитектура:
- Planner/Critic: Qwen3-1.7B (2048 hidden) — frozen
- Solver: Qwen2.5-Math-1.5B-Instruct (1536 hidden) — frozen
- Outer Link 1: Planner → Critic (2048 → 2048) — trainable
- Outer Link 2: Critic → Solver (2048 → 1536) — trainable

Pipeline:
  1. Planner: question + plan → hidden states (frozen, teacher forcing)
  2. Outer1: planner_hidden → critic_hidden (2048→2048)
  3. Critic: question + slot(critic_hidden) → hidden states (frozen)
  4. Outer2: critic_hidden → solver_hidden (2048→1536)
  5. Solver: question + slot(solver_hidden) → logits
  6. CE Loss: logits vs answer tokens (все токены!)
  7. Backward → optimize Outer1 + Outer2

Data:
  - Teacher data (plan, refined_plan) pre-generated → JSON
  - GSM8K train split

Hyperparameters:
  - batch_size: 2
  - lr: 5e-4, betas=(0.9, 0.95)
  - epochs: 5
  - warmup: 10 steps
  - gradient clipping: 1.0
  - cosine scheduler

Пример:
    python train_outer_link_gsm8k.py --teacher_data ~/workspace/tmp/recursivemas/teacher_data/gsm8k_train_5000.json
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Модели
PLANNER_MODEL = "Qwen/Qwen3-1.7B"
SOLVER_MODEL = "./models/Qwen2.5-Math-1.5B-Instruct"
DEFAULT_DEVICE = "cuda:2"

# Hidden sizes
PLANNER_HIDDEN = 2048
SOLVER_HIDDEN = 1536

# System prompts
PLANNER_SYSTEM = (
    "You are a math problem planner. "
    "Given a math word problem, decompose it into clear, step-by-step instructions "
    "that a solver can follow to arrive at the correct answer."
)

CRITIC_SYSTEM = (
    "You are a math problem critic. "
    "Review and improve the solution plan."
)

SOLVER_SYSTEM = (
    "You are a math problem solver. "
    "Solve the math problem step by step. "
    "Put your final numerical answer after ####, for example: #### 42"
)


# ============================================================
# Outer Adapter (outer_ln_res_adapter — как в оригинале)
# ============================================================
class OuterAdapter(nn.Module):
    """
    LayerNorm(in) → proj1(in→2*out) → GELU → proj2(2*out→out) → +residual_proj(in→out) → LayerNorm(out)
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.ln_source = nn.LayerNorm(in_dim)
        hidden_dim = out_dim * 2  # expansion 2x!
        self.proj1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden_dim, out_dim)
        self.residual_proj = nn.Linear(in_dim, out_dim)
        self.ln_target = nn.LayerNorm(out_dim)

    def forward(self, x):
        h = self.ln_source(x)
        out = self.proj2(self.act(self.proj1(h)))
        out = out + self.residual_proj(x)  # residual
        out = self.ln_target(out)
        return out


# ============================================================
# Dataset
# ============================================================
class GSM8KTeacherDataset(Dataset):
    """Датасет с pre-generated teacher data."""

    def __init__(self, data_path):
        print(f"Загрузка teacher data: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        print(f"  Загружено {len(self.data)} примеров")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Простой collate — возвращает список словарей."""
    return batch


# ============================================================
# Training Pipeline
# ============================================================
class TrainingPipeline:
    """Pipeline для обучения Outer Link адаптеров."""

    def __init__(self, device):
        self.device = device

        # Загрузка моделей
        print("\nЗагрузка моделей...")

        # Planner/Critic
        self.planner_tokenizer = AutoTokenizer.from_pretrained(
            PLANNER_MODEL, trust_remote_code=True
        )
        self.planner_model = AutoModelForCausalLM.from_pretrained(
            PLANNER_MODEL,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        self.planner_model.eval()
        for p in self.planner_model.parameters():
            p.requires_grad = False
        print(f"  Planner/Critic: {PLANNER_MODEL} (hidden={PLANNER_HIDDEN})")

        # Solver
        self.solver_tokenizer = AutoTokenizer.from_pretrained(
            SOLVER_MODEL, trust_remote_code=True
        )
        self.solver_model = AutoModelForCausalLM.from_pretrained(
            SOLVER_MODEL,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        self.solver_model.eval()
        for p in self.solver_model.parameters():
            p.requires_grad = False
        print(f"  Solver: {SOLVER_MODEL} (hidden={SOLVER_HIDDEN})")

        # Outer Link адаптеры
        print("\nСоздание Outer Link адаптеров...")
        self.outer_1 = OuterAdapter(PLANNER_HIDDEN, PLANNER_HIDDEN).to(device)  # Planner → Critic
        self.outer_2 = OuterAdapter(PLANNER_HIDDEN, SOLVER_HIDDEN).to(device)   # Critic → Solver

        total_params = sum(p.numel() for p in self.outer_1.parameters()) + \
                       sum(p.numel() for p in self.outer_2.parameters())
        print(f"  Outer1 (Planner→Critic): {PLANNER_HIDDEN} → {PLANNER_HIDDEN}")
        print(f"  Outer2 (Critic→Solver):  {PLANNER_HIDDEN} → {SOLVER_HIDDEN}")
        print(f"  Trainable params: {total_params:,}")

    def _get_planner_hidden(self, question, plan):
        """Planner: question + plan → hidden states (teacher forcing)."""
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": plan},
        ]
        input_text = self.planner_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.planner_tokenizer(
            input_text, return_tensors="pt",
            truncation=True, max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.planner_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
            )

        # Извлекаем hidden states assistant токенов
        assistant_tokens = self.planner_tokenizer(plan, add_special_tokens=False).input_ids
        assistant_start = inputs.input_ids.shape[1] - len(assistant_tokens)
        hidden = outputs.hidden_states[-1][0, assistant_start:, :]  # [seq_len, 2048]
        return hidden

    def _get_critic_hidden(self, question, refined_plan, slot_embeds):
        """Critic: question + slot(slot_embeds) → hidden states."""
        messages = [
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": f"Problem:\n{question}\n\nSolution Plan:\n<<SLOT>>"},
        ]
        input_text = self.planner_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.planner_tokenizer(
            input_text, return_tensors="pt",
            truncation=True, max_length=512,
        ).to(self.device)

        # Slot injection
        input_embeds = self.planner_model.get_input_embeddings()(inputs.input_ids)
        slot_pos = inputs.input_ids.shape[1]
        seq_embeds = torch.cat([input_embeds, slot_embeds], dim=1)

        with torch.no_grad():
            outputs = self.planner_model(
                inputs_embeds=seq_embeds,
                attention_mask=torch.ones(seq_embeds.shape[:2], dtype=torch.long, device=self.device),
                output_hidden_states=True,
            )

        # Берём последние токены (ответ critic'а)
        hidden = outputs.hidden_states[-1][0, slot_pos:, :]  # [seq_len, 2048]
        return hidden

    def _solver_forward(self, question, slot_embeds, answer):
        """Solver: question + slot(slot_embeds) → CE loss vs answer."""
        # Prompt для Solver'а
        messages = [
            {"role": "system", "content": SOLVER_SYSTEM},
            {"role": "user", "content": f"Problem:\n{question}\n\nPlan:\n<<SLOT>>"},
        ]
        input_text = self.solver_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.solver_tokenizer(
            input_text, return_tensors="pt",
            truncation=True, max_length=512,
        ).to(self.device)

        # Slot injection
        input_embeds = self.solver_model.get_input_embeddings()(inputs.input_ids)
        slot_pos = inputs.input_ids.shape[1]
        seq_embeds = torch.cat([input_embeds, slot_embeds], dim=1)

        # Labels: -100 для prompt, реальные IDs для answer
        # Токенизируем answer
        answer_tokens = self.solver_tokenizer(answer, add_special_tokens=False).input_ids
        answer_tensor = torch.tensor(answer_tokens, dtype=torch.long, device=self.device).unsqueeze(0)

        # Создаём labels
        total_len = seq_embeds.shape[1]
        labels = torch.full((1, total_len), -100, dtype=torch.long, device=self.device)
        # Answer начинается после slot
        answer_start = slot_pos + slot_embeds.shape[1]
        if answer_start + len(answer_tokens) <= total_len:
            labels[0, answer_start:answer_start + len(answer_tokens)] = answer_tensor[0]
        else:
            # Если answer не помещается, берём сколько влезет
            remaining = total_len - answer_start
            if remaining > 0:
                labels[0, answer_start:answer_start + remaining] = answer_tensor[0, :remaining]

        # Forward
        outputs = self.solver_model(
            inputs_embeds=seq_embeds,
            attention_mask=torch.ones(seq_embeds.shape[:2], dtype=torch.long, device=self.device),
            labels=labels,
        )
        return outputs.loss

    def forward(self, question, plan, refined_plan, answer):
        """Полный forward pass: Planner → Outer1 → Critic → Outer2 → Solver."""

        # 1. Planner → hidden
        planner_hidden = self._get_planner_hidden(question, plan)  # [seq_len, 2048]

        # 2. Outer1: Planner → Critic
        planner_to_critic = self.outer_1(planner_hidden.unsqueeze(0))  # [1, seq_len, 2048]

        # 3. Critic ← slot → hidden
        critic_hidden = self._get_critic_hidden(question, refined_plan, planner_to_critic)

        # 4. Outer2: Critic → Solver
        critic_to_solver = self.outer_2(critic_hidden.unsqueeze(0))  # [1, seq_len, 1536]

        # 5. Solver ← slot → CE loss
        loss = self._solver_forward(question, critic_to_solver, answer)

        return loss


# ============================================================
# Validation
# ============================================================
def validate(pipeline, checkpoint_path, num_examples=100, device="cuda:2"):
    """Валидация через evaluate_recursivemas.py."""
    # Сохраняем текущие адаптеры
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        "outer_1": pipeline.outer_1.state_dict(),
        "outer_2": pipeline.outer_2.state_dict(),
    }, checkpoint_path)

    # Запускаем evaluate_recursivemas.py
    import subprocess
    result = subprocess.run(
        [
            sys.executable,
            "evaluate_recursivemas.py",
            "--num_examples", str(num_examples),
            "--device", device,
            "--checkpoint", checkpoint_path,
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )

    # Парсим accuracy из вывода
    output = result.stdout + result.stderr
    for line in output.split("\n"):
        if "Accuracy:" in line:
            parts = line.split("Accuracy:")[1].strip()
            acc_str = parts.split("%")[0].strip()
            return float(acc_str)

    return None


# ============================================================
# Training
# ============================================================
def train(teacher_data_path, epochs, batch_size, lr, num_examples_val, output_dir):
    """Основной цикл обучения."""

    device = DEFAULT_DEVICE
    checkpoint_path = os.path.join(output_dir, "latest.pt")
    best_checkpoint = os.path.join(output_dir, "best.pt")

    # Pipeline
    pipeline = TrainingPipeline(device)

    # Датасет
    dataset = GSM8KTeacherDataset(teacher_data_path)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Оптимизатор
    all_params = list(pipeline.outer_1.parameters()) + list(pipeline.outer_2.parameters())
    optimizer = torch.optim.AdamW(
        all_params, lr=lr, weight_decay=0.0, betas=(0.9, 0.95)
    )

    # Scheduler
    num_steps = epochs * len(dataloader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=num_steps,
        num_cycles=0.5,
    )

    print(f"\n{'='*70}")
    print(f"НАЧАЛО ОБУЧЕНИЯ")
    print(f"{'='*70}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR: {lr}")
    print(f"  Samples: {len(dataset)}")
    print(f"  Steps per epoch: {len(dataloader)}")
    print(f"  Total steps: {num_steps}")
    print(f"{'='*70}\n")

    best_accuracy = 0.0
    global_step = 0

    for epoch in range(epochs):
        pipeline.outer_1.train()
        pipeline.outer_2.train()

        epoch_losses = []
        epoch_start = time.time()

        for step, batch in enumerate(dataloader):
            batch_loss = 0.0

            # Process each sample in batch individually (разная длина)
            for sample in batch:
                question = sample["question"]
                plan = sample["plan"]
                refined_plan = sample["refined_plan"]
                answer = sample["answer"]

                loss = pipeline.forward(question, plan, refined_plan, answer)
                loss.backward()
                batch_loss += loss.item()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            avg_batch_loss = batch_loss / len(batch)
            epoch_losses.append(avg_batch_loss)

            if global_step % 20 == 0:
                elapsed = time.time() - epoch_start
                print(f"  Epoch {epoch+1}/{epochs}, Step {global_step}, "
                      f"Loss: {avg_batch_loss:.4f}, Time: {elapsed:.0f}s")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_time = time.time() - epoch_start

        print(f"\n  📊 Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, Time = {epoch_time:.0f}s")

        # Валидация
        print(f"\n  🔍 Валидация (100 примеров)...")
        accuracy = validate(pipeline, checkpoint_path, num_examples=num_examples_val, device=device)

        if accuracy is not None:
            print(f"  ✅ Accuracy: {accuracy:.1f}%")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save({
                    "outer_1": pipeline.outer_1.state_dict(),
                    "outer_2": pipeline.outer_2.state_dict(),
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "accuracy": accuracy,
                }, best_checkpoint)
                print(f"  🏆 New best: {accuracy:.1f}% (saved to {best_checkpoint})")
        else:
            print(f"  ⚠️ Не удалось распарсить accuracy")

        # Сохраняем чекпоинт эпохи
        epoch_checkpoint = os.path.join(output_dir, f"epoch_{epoch+1}.pt")
        torch.save({
            "outer_1": pipeline.outer_1.state_dict(),
            "outer_2": pipeline.outer_2.state_dict(),
            "epoch": epoch + 1,
            "loss": avg_loss,
        }, epoch_checkpoint)

    # Финал
    print(f"\n{'='*70}")
    print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"{'='*70}")
    print(f"  Best accuracy: {best_accuracy:.1f}%")
    print(f"  Best checkpoint: {best_checkpoint}")
    print(f"{'='*70}")

    return best_accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Train Outer Link adapters on GSM8K",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--teacher_data",
        type=str,
        required=True,
        help="Путь к pre-generated teacher data (JSON)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Количество эпох (default: 5)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (default: 2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--num_examples_val",
        type=int,
        default=100,
        help="Примеров для валидации (default: 100)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Директория для чекпоинтов (default: ~/workspace/tmp/recursivemas/training/)",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.expanduser("~/workspace/tmp/recursivemas/training/")

    os.makedirs(args.output_dir, exist_ok=True)

    # Сохраняем конфиг
    config = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    train(
        teacher_data_path=args.teacher_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_examples_val=args.num_examples_val,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
