#!/usr/bin/env python3
"""
⚠️ УСТАРЕЛ — используйте custom_impl/train_outer.py

Этот скрипт использует GSM8K (не Sequential-Math) и кастомный подход к teacher forcing.
Для воспроизведения оригинальной реализации используйте:
    python custom_impl/train_outer.py --agent1_model ... --agent2_model ... --agent3_model ...

RecursiveMAS — Outer Link Training (Путь Б: Рабочий прототип для GSM8K)

Архитектура:
- Planner: Qwen3-1.7B (2048 hidden)
- Critic:  Qwen3-1.7B (2048 hidden)
- Solver:  Qwen2.5-Math-1.5B (1536 hidden)

Outer Link 1 (Planner → Critic): 2048 → 2048
Outer Link 2 (Critic → Solver):   2048 → 1536
Outer Link 3 (Solver → Planner):  1536 → 2048

Особенности:
✅ Slot injection через placeholder <<LATENT_SLOT>>
✅ Teacher forcing (используем сгенерированный текст для hidden states)
✅ Chat templates (apply_chat_template)
✅ CE loss на ВСЕХ токенах ответа
✅ Поддержка двух моделей с разными hidden sizes
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import copy

# Фикс для PyTorch CUDA зависимостей
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# ============================================================
# Конфигурация
# ============================================================
CONFIG = {
    "planner_model": "Qwen/Qwen3-1.7B",
    "critic_model": "Qwen/Qwen3-1.7B",
    "solver_model": "./models/Qwen2.5-Math-1.5B-Instruct",
    "hidden_sizes": {
        "planner": 2048,
        "critic": 2048,
        "solver": 1536,
    },
    "training": {
        "epochs": 3,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "max_steps": 100,
        "gradient_accumulation": 4,
        "max_question_length": 128,
        "max_answer_length": 256,
        "num_beams": 1,
        "max_new_tokens": 256,
    },
    "adapter": {
        "hidden_dim": 512,
    },
    "device": "cuda:2",  # RTX 4060 Ti 16GB
}

# ============================================================
# Outer Link Adapter (с поддержкой разных размеров)
# ============================================================
class OuterLinkAdapter(nn.Module):
    """
    Адаптер для связи между разными моделями.
    Поддерживает dimension mapping (W3 для resize).
    
    R_out(h) = W3·h + W2·GELU(W1·h)
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Non-linear branch
        self.W1 = nn.Linear(in_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, out_dim)
        
        # Linear branch (dimension mapping)
        self.W3 = nn.Linear(in_dim, out_dim, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация: W3 ≈ identity (если возможно), W2 ≈ 0"""
        nn.init.kaiming_uniform_(self.W1.weight, a=0)  # GELU
        nn.init.zeros_(self.W2.weight)
        
        # W3: если in_dim == out_dim → identity, иначе → zeros
        if self.in_dim == self.out_dim:
            nn.init.eye_(self.W3.weight)
        else:
            nn.init.zeros_(self.W3.weight)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [batch, seq_len, in_dim] — hidden states от модели-источника
        Returns:
            [batch, seq_len, out_dim] — mapped hidden states
        """
        linear_branch = self.W3(h)
        non_linear_branch = self.W2(torch.nn.functional.gelu(self.W1(h)))
        return linear_branch + non_linear_branch


# ============================================================
# GSM8K Dataset
# ============================================================
class GSM8KDataset(Dataset):
    """
    Датасет GSM8K с teacher forcing.
    
    Для каждого примера:
    1. Генерируем Plan (Planner)
    2. Генерируем Refined Plan (Critic)
    3. Используем Answer (ground truth)
    """
    def __init__(self, questions: list, answers: list):
        self.questions = questions
        self.answers = answers
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return {
            "question": self.questions[idx],
            "answer": self.answers[idx],
        }


# ============================================================
# Slot Injection (как в оригинале)
# ============================================================
def build_stage_with_slot(
    tokenizer,
    user_prompt: str,
    assistant_text: str = None,  # teacher forcing
    slot_placeholder: str = "<<LATENT_SLOT>>",
    max_length: int = 512,
):
    """
    Создаёт input с placeholder'ом для slot injection.
    
    Возвращает:
    - prefix_ids: токены ДО placeholder'а
    - suffix_ids: токены ПОСЛЕ placeholder'а
    - assistant_mask: маска для assistant токенов (teacher forcing)
    - labels: labels для CE loss
    """
    # 1. Формируем prompt с placeholder'ом
    if assistant_text:
        # Teacher forcing: assistant_text — это ground truth
        messages = [
            {"role": "user", "content": user_prompt.replace(slot_placeholder, "")},
            {"role": "assistant", "content": assistant_text},
        ]
    else:
        # Inference: только user prompt
        messages = [
            {"role": "user", "content": user_prompt},
        ]
    
    # 2. Применяем chat template
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not assistant_text,
    )
    
    # 3. Токенизируем
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    
    input_ids = tokenized.input_ids[0]
    attention_mask = tokenized.attention_mask[0]
    
    # 4. Находим границы assistant токенов (для teacher forcing)
    # В chat template assistant начинается после специального токена
    assistant_start = len(input_ids)
    if assistant_text:
        # Токенизируем только assistant текст для определения длины
        assistant_tokens = tokenizer(assistant_text, add_special_tokens=False).input_ids
        assistant_start = len(input_ids) - len(assistant_tokens)
    
    # 5. Создаём маску assistant токенов
    assistant_mask = torch.zeros(len(input_ids), dtype=torch.bool)
    if assistant_text:
        assistant_mask[assistant_start:] = True
    
    # 6. Создаём labels для CE loss (-100 для prompt, реальные IDs для assistant)
    labels = torch.full((len(input_ids),), -100, dtype=torch.long)
    if assistant_text:
        labels[assistant_start:] = input_ids[assistant_start:]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "assistant_mask": assistant_mask,
        "labels": labels,
        "full_text": full_text,
    }


def inject_slot_embeds(
    model,
    input_ids: torch.Tensor,
    slot_embeds: torch.Tensor,
    slot_position: int,
) -> torch.Tensor:
    """
    Вставляет slot embeddings в указанную позицию.
    
    Args:
        model: модель с get_input_embeddings()
        input_ids: [batch, seq_len]
        slot_embeds: [batch, slot_len, hidden_dim]
        slot_position: позиция вставки (после каких токенов)
    
    Returns:
        inputs_embeds: [batch, seq_len + slot_len, hidden_dim]
    """
    embedding_layer = model.get_input_embeddings()
    input_embeds = embedding_layer(input_ids)
    
    # Вставляем slot после slot_position токенов
    prefix_embeds = input_embeds[:, :slot_position, :]
    suffix_embeds = input_embeds[:, slot_position:, :]
    
    seq_embeds = torch.cat([prefix_embeds, slot_embeds, suffix_embeds], dim=1)
    return seq_embeds


# ============================================================
# RecursiveMAS Pipeline
# ============================================================
class RecursiveMASPipeline(nn.Module):
    """
    Полный пайплайн RecursiveMAS с 3 агентами и outer link адаптерами.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # Загрузка моделей
        print("Загрузка моделей...")
        
        # Planner
        self.planner_tokenizer = AutoTokenizer.from_pretrained(config["planner_model"])
        self.planner_model = AutoModelForCausalLM.from_pretrained(
            config["planner_model"],
            torch_dtype=torch.float16,
            device_map="cuda:2",
        )
        self.planner_model.eval()
        for param in self.planner_model.parameters():
            param.requires_grad = False
        
        # Critic
        self.critic_tokenizer = AutoTokenizer.from_pretrained(config["critic_model"])
        self.critic_model = AutoModelForCausalLM.from_pretrained(
            config["critic_model"],
            torch_dtype=torch.float16,
            device_map="cuda:2",
        )
        self.critic_model.eval()
        for param in self.critic_model.parameters():
            param.requires_grad = False
        
        # Solver
        self.solver_tokenizer = AutoTokenizer.from_pretrained(config["solver_model"])
        self.solver_model = AutoModelForCausalLM.from_pretrained(
            config["solver_model"],
            torch_dtype=torch.float16,
            device_map="cuda:2",
        )
        self.solver_model.eval()
        for param in self.solver_model.parameters():
            param.requires_grad = False
        
        # Outer Link адаптеры
        hidden_dim = config["adapter"]["hidden_dim"]
        h_planner = config["hidden_sizes"]["planner"]
        h_critic = config["hidden_sizes"]["critic"]
        h_solver = config["hidden_sizes"]["solver"]
        
        self.outer_1 = OuterLinkAdapter(h_planner, h_critic, hidden_dim)  # Planner → Critic
        self.outer_2 = OuterLinkAdapter(h_critic, h_solver, hidden_dim)   # Critic → Solver
        self.outer_3 = OuterLinkAdapter(h_solver, h_planner, hidden_dim)  # Solver → Planner
        
        print(f"Outer Link 1 (Planner→Critic): {h_planner} → {h_critic}")
        print(f"Outer Link 2 (Critic→Solver):  {h_critic} → {h_solver}")
        print(f"Outer Link 3 (Solver→Planner): {h_solver} → {h_planner}")
    
    def generate_plan(self, question: str) -> str:
        """Генерирует план с помощью Planner (учитель)"""
        messages = [
            {"role": "system", "content": "You are a planner. Decompose the problem into steps."},
            {"role": "user", "content": question},
        ]
        
        input_text = self.planner_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.planner_tokenizer(input_text, return_tensors="pt").to("cuda:2")
        
        with torch.no_grad():
            outputs = self.planner_model.generate(
                **inputs,
                max_new_tokens=self.config["training"]["max_new_tokens"],
                num_beams=self.config["training"]["num_beams"],
                do_sample=False,
            )
        
        # Декодируем только новую часть
        plan = self.planner_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return plan
    
    def generate_refined_plan(self, question: str, plan: str) -> str:
        """Генерирует улучшенный план с помощью Critic (учитель)"""
        messages = [
            {"role": "system", "content": "You are a critic. Review and improve the plan."},
            {"role": "user", "content": f"Question: {question}\n\nInitial Plan:\n{plan}"},
        ]
        
        input_text = self.critic_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.critic_tokenizer(input_text, return_tensors="pt").to("cuda:2")
        
        with torch.no_grad():
            outputs = self.critic_model.generate(
                **inputs,
                max_new_tokens=self.config["training"]["max_new_tokens"],
                num_beams=self.config["training"]["num_beams"],
                do_sample=False,
            )
        
        refined_plan = self.critic_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return refined_plan
    
    def forward(
        self,
        question: str,
        plan: str,
        refined_plan: str,
        answer: str,
    ) -> dict:
        """
        Forward pass с teacher forcing.
        
        Args:
            question: вопрос
            plan: план от Planner (teacher)
            refined_plan: улучшенный план от Critic (teacher)
            answer: правильный ответ (ground truth)
        
        Returns:
            dict с loss и hidden states
        """
        device = "cuda:2"
        
        # ============================================================
        # Шаг 1: Planner → hidden states (teacher forcing)
        # ============================================================
        planner_pack = build_stage_with_slot(
            tokenizer=self.planner_tokenizer,
            user_prompt=question,
            assistant_text=plan,
            max_length=self.config["training"]["max_question_length"] + 128,
        )
        
        planner_input_ids = planner_pack["input_ids"].unsqueeze(0).to(device)
        planner_attn_mask = planner_pack["attention_mask"].unsqueeze(0).to(device)
        planner_assist_mask = planner_pack["assistant_mask"].to(device)
        
        with torch.no_grad():
            planner_outputs = self.planner_model(
                input_ids=planner_input_ids,
                attention_mask=planner_attn_mask,
                output_hidden_states=True,
            )
        
        # Извлекаем hidden states ТОЛЬКО для assistant токенов
        planner_hidden = planner_outputs.hidden_states[-1][0][planner_assist_mask]  # [seq_len, 2048]
        
        # ============================================================
        # Шаг 2: Critic ← Outer Link 1 → hidden states (teacher forcing)
        # ============================================================
        # Маппим planner hidden → critic space
        planner_to_critic = self.outer_1(planner_hidden.unsqueeze(0))  # [1, seq_len, 2048]
        
        # Создаём input для Critic с slot injection
        critic_user_prompt = f"Question: {question}\n\nInitial Plan:\n<<LATENT_SLOT>>"
        critic_pack = build_stage_with_slot(
            tokenizer=self.critic_tokenizer,
            user_prompt=critic_user_prompt,
            assistant_text=refined_plan,
            max_length=self.config["training"]["max_question_length"] + 256,
        )
        
        critic_input_ids = critic_pack["input_ids"].unsqueeze(0).to(device)
        critic_attn_mask = critic_pack["attention_mask"].unsqueeze(0).to(device)
        critic_assist_mask = critic_pack["assistant_mask"].to(device)
        
        # Находим позицию вставки slot (после user prompt)
        slot_position = len(critic_input_ids[0]) - len(
            self.critic_tokenizer(refined_plan, add_special_tokens=False).input_ids
        )
        
        # Вставляем slot embeds
        critic_embeds = inject_slot_embeds(
            model=self.critic_model,
            input_ids=critic_input_ids,
            slot_embeds=planner_to_critic,
            slot_position=slot_position,
        )
        
        # Forward через Critic
        with torch.no_grad():
            critic_outputs = self.critic_model(
                inputs_embeds=critic_embeds,
                attention_mask=torch.ones(critic_embeds.shape[:2], dtype=torch.long, device=device),
                output_hidden_states=True,
            )
        
        # Извлекаем hidden states для assistant токенов
        # Нужно скорректировать mask из-за вставленного slot
        critic_hidden = critic_outputs.hidden_states[-1][0][critic_assist_mask]  # [seq_len, 2048]
        
        # ============================================================
        # Шаг 3: Solver ← Outer Link 2 → hidden states (teacher forcing)
        # ============================================================
        # Маппим critic hidden → solver space (2048 → 1536!)
        critic_to_solver = self.outer_2(critic_hidden.unsqueeze(0))  # [1, seq_len, 1536]
        
        # Создаём input для Solver с slot injection
        solver_user_prompt = f"Question: {question}\n\nPlan:\n<<LATENT_SLOT>>"
        solver_pack = build_stage_with_slot(
            tokenizer=self.solver_tokenizer,
            user_prompt=solver_user_prompt,
            assistant_text=answer,
            max_length=self.config["training"]["max_question_length"] + 256,
        )
        
        solver_input_ids = solver_pack["input_ids"].unsqueeze(0).to(device)
        solver_attn_mask = solver_pack["attention_mask"].unsqueeze(0).to(device)
        solver_assist_mask = solver_pack["assistant_mask"].to(device)
        solver_labels = solver_pack["labels"].unsqueeze(0).to(device)
        
        # Находим позицию вставки slot
        slot_position = len(solver_input_ids[0]) - len(
            self.solver_tokenizer(answer, add_special_tokens=False).input_ids
        )
        
        # Вставляем slot embeds
        solver_embeds = inject_slot_embeds(
            model=self.solver_model,
            input_ids=solver_input_ids,
            slot_embeds=critic_to_solver,
            slot_position=slot_position,
        )
        
        # Forward через Solver
        solver_outputs = self.solver_model(
            inputs_embeds=solver_embeds,
            attention_mask=torch.ones(solver_embeds.shape[:2], dtype=torch.long, device=device),
            labels=solver_labels,
        )
        
        # CE loss на ВСЕХ токенах ответа
        loss = solver_outputs.loss
        
        return {
            "loss": loss,
            "planner_hidden": planner_hidden,
            "critic_hidden": critic_hidden,
            "solver_logits": solver_outputs.logits,
        }
    
    def inference(self, question: str) -> str:
        """
        Inference без teacher forcing.
        Planner → Critic → Solver с генерацией.
        """
        device = "cuda:2"
        
        # Шаг 1: Planner генерирует план
        plan = self.generate_plan(question)
        
        # Шаг 2: Critic улучшает план (через outer link)
        # ... (аналогично forward, но с генерацией)
        
        # Шаг 3: Solver решает задачу
        # ... (аналогично forward, но с генерацией)
        
        return "TODO: inference implementation"


# ============================================================
# Training Loop
# ============================================================
def train():
    """Основной цикл обучения"""
    device = torch.device(CONFIG["device"])
    
    # Загрузка GSM8K
    print("Загрузка GSM8K...")
    dataset = load_dataset("gsm8k", "main")
    
    # Берём первые 100 примеров для быстрого теста
    questions = dataset["train"]["question"][:100]
    answers = dataset["train"]["answer"][:100]
    
    # Создание pipeline
    print("Создание RecursiveMAS pipeline...")
    pipeline = RecursiveMASPipeline(CONFIG).to(device)
    
    # Оптимизатор (только для адаптеров)
    optimizer = torch.optim.AdamW(
        pipeline.parameters(),
        lr=CONFIG["training"]["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    
    # DataLoader
    gsm8k_dataset = GSM8KDataset(questions, answers)
    dataloader = DataLoader(gsm8k_dataset, batch_size=CONFIG["training"]["batch_size"], shuffle=True)
    
    # Training loop
    print("Начало обучения...")
    global_step = 0
    for epoch in range(CONFIG["training"]["epochs"]):
        pipeline.train()
        total_loss = 0
        
        for batch in dataloader:
            if global_step >= CONFIG["training"]["max_steps"]:
                break
            
            question = batch["question"][0]
            answer = batch["answer"][0]
            
            # Генерация teacher forcing данных
            plan = pipeline.generate_plan(question)
            refined_plan = pipeline.generate_refined_plan(question, plan)
            
            # Forward pass
            outputs = pipeline(question, plan, refined_plan, answer)
            loss = outputs["loss"]
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(pipeline.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            global_step += 1
            
            if global_step % 10 == 0:
                print(f"Step {global_step}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / max(global_step, 1)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
    
    # Сохранение checkpoint'а
    checkpoint_path = "/home/hermes/workspace/projects/recursivemas/custom_impl/checkpoints/outer_link_gsm8k.pt"
    torch.save({
        "outer_1": pipeline.outer_1.state_dict(),
        "outer_2": pipeline.outer_2.state_dict(),
        "outer_3": pipeline.outer_3.state_dict(),
        "config": CONFIG,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    train()
