"""
Практическое задание: Обучение Outer Link (исправленная версия)

Следует оригинальной реализации из train/outer/sequential.py.

Исправления относительно v1:
1. ✅ Outer Link: expansion 2x (hidden_dim = out_dim * 2)
2. ✅ Outer Link: LayerNorm (source + target)
3. ✅ Optimizer: AdamW(betas=(0.9, 0.95))
4. ✅ Gradient clipping: max_norm=1.0
5. ✅ Scheduler: cosine with warmup
6. ✅ Data format: {question, plan, refined_plan, answer}
7. ✅ Latent injection: через slot в prompt
8. ✅ Inner adapter: ln_res_adapter (загружаем из v2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import os
import random
import math
import json

# ==================== Конфигурация ====================
CONFIG = {
    "model_name": "Qwen/Qwen3-1.7B",
    "device": "cuda:2",
    "batch_size": 2,
    "epochs": 2,
    "lr": 5e-4,
    "warmup_steps": 10,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "recursion_rounds": 2,
    "latent_steps": 5,  # уменьшено для скорости (оригинал: 80)
    "max_length": 512,
    "num_samples": 50,
    "save_path": "custom_impl/checkpoints/outer_link_sequential_v2",
}


# ==================== Inner Adapter (ln_res_adapter) ====================
class InnerAdapter(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.pre_ln = nn.LayerNorm(hidden_size)
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden_size, hidden_size)
        self.post_ln = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        h = self.pre_ln(x)
        out = self.proj2(self.act(self.proj1(h)))
        out = self.post_ln(x + out)
        return out


# ==================== Outer Adapter (outer_ln_res_adapter) ====================
class OuterAdapter(nn.Module):
    """
    outer_ln_res_adapter из оригинала:
    LayerNorm(in) → proj1(in→2*out) → GELU → proj2(2*out→out) → +residual_proj(in→out) → LayerNorm(out)
    """
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
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


# ==================== Агент ====================
class Agent(nn.Module):
    def __init__(self, model, tokenizer, role, d_model):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.role = role
        self.inner_adapter = InnerAdapter(d_model)
        
        # Role prompts (упрощённые)
        role_prompts = {
            "planner": "You are a Planner. Decompose the problem and create a solution plan.\nQuestion: ",
            "refiner": "You are a Refiner. Review the plan and suggest improvements.\nQuestion: ",
            "solver": "You are a Solver. Execute the plan and produce the final answer.\nQuestion: ",
        }
        self.role_prompt = role_prompts[role]
        
        role_tokens = tokenizer(self.role_prompt, return_tensors="pt")
        self.register_buffer("role_input_ids", role_tokens["input_ids"])
        self.register_buffer("role_attention_mask", role_tokens["attention_mask"])
    
    def forward(self, question_ids, question_mask, slot_embeds=None, compute_logits=False):
        """
        Forward pass. Модель заморожена (requires_grad=False),
        но градиенты проходят через slot_embeds → inputs_embeds → hidden_states.
        """
        batch_size = question_ids.shape[0]
        
        # Role prompt
        role_ids = self.role_input_ids.expand(batch_size, -1)
        role_mask = self.role_attention_mask.expand(batch_size, -1)
        
        # Concatenate: role + question
        combined_ids = torch.cat([role_ids, question_ids], dim=1)
        combined_mask = torch.cat([role_mask, question_mask], dim=1)
        
        # Получаем embeddings
        input_embeds = self.model.get_input_embeddings()(combined_ids)
        
        # Вставляем slot embeds (GRADIENTS FLOW HERE!)
        if slot_embeds is not None:
            slot_pos = role_ids.shape[1]
            input_embeds = torch.cat([
                input_embeds[:, :slot_pos, :],
                slot_embeds,
                input_embeds[:, slot_pos:, :]
            ], dim=1)
            combined_mask = torch.cat([
                combined_mask[:, :slot_pos],
                torch.ones(batch_size, slot_embeds.shape[1], device=combined_mask.device),
                combined_mask[:, slot_pos:]
            ], dim=1)
        
        # Forward pass — модель заморожена, но градиенты проходят через inputs_embeds
        # Ключ: НЕ используем no_grad, чтобы autograd мог проследить от slot_embeds к hidden_states
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=combined_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        
        return outputs


# ==================== Датасет ====================
class MathDataset(Dataset):
    def __init__(self, tokenizer, num_samples=50):
        self.tokenizer = tokenizer
        self.samples = self._generate_samples(num_samples)
    
    def _generate_samples(self, n):
        samples = []
        for i in range(n):
            task_type = random.choice(['linear', 'quadratic', 'arithmetic'])
            
            if task_type == 'linear':
                a, b, c = random.randint(1, 10), random.randint(-10, 10), random.randint(-20, 20)
                x = -(b + c) / a if a != 0 else 0
                question = f"Solve for x: {a}x + {b} = {c}"
                plan = f"Step 1: Rearrange to {a}x = {c-b}. Step 2: Divide by {a}. Result: x = {x:.1f}"
                refined_plan = f"The plan is correct. {a}x = {c-b}, so x = ({c-b})/{a} = {x:.1f}"
                answer = f"{x:.1f}"
            elif task_type == 'quadratic':
                a, b, c = random.randint(1, 5), random.randint(-10, 10), random.randint(-20, 20)
                d = b*b - 4*a*c
                if d >= 0:
                    x1 = (-b + math.sqrt(d)) / (2*a)
                    question = f"Solve: {a}x^2 + {b}x + {c} = 0"
                    plan = f"Use quadratic formula: x = (-{b} ± sqrt({d})) / {2*a}"
                    refined_plan = f"Discriminant D={d} > 0. Two real roots: x₁={x1:.1f}"
                    answer = f"{x1:.1f}"
                else:
                    question = f"Is x^2 + {b}x + {c} = 0 solvable?"
                    plan = f"Check discriminant: D = {b}^2 - 4*1*{c} = {d}"
                    refined_plan = f"D={d} < 0, no real solutions"
                    answer = "no"
            else:
                a, b = random.randint(1, 100), random.randint(1, 100)
                op = random.choice(['+', '-', '*'])
                result = eval(f"{a} {op} {b}")
                question = f"Calculate: {a} {op} {b}"
                plan = f"Compute {a} {op} {b} = {result}"
                refined_plan = f"The calculation is straightforward: {a} {op} {b} = {result}"
                answer = f"{result}"
            
            samples.append({
                "question": question,
                "plan": plan,
                "refined_plan": refined_plan,
                "answer": answer,
            })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ==================== Training ====================
def train_outer_link():
    print("=" * 60)
    print("OUTER LINK TRAINING (v2 — исправленная версия)")
    print("=" * 60)
    
    device = torch.device(CONFIG["device"])
    print(f"\n📍 Device: {device}")
    print(f"🔄 Recursion rounds: {CONFIG['recursion_rounds']}")
    print(f"📊 Samples: {CONFIG['num_samples']}")
    print(f"📏 Batch size: {CONFIG['batch_size']}")
    
    # 1. Загружаем модель ОДИН РАЗ
    print("\n🔄 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.bfloat16,
        device_map={"" : device.index},
        trust_remote_code=True,
    )
    model.eval()
    model.requires_grad_(False)
    
    d_model = model.config.hidden_size
    print(f"📐 d_model: {d_model}")
    
    # 2. Создаём 3 агента
    print("\n🤖 Creating 3 agents (Planner, Refiner, Solver)...")
    planner = Agent(model, tokenizer, "planner", d_model).to(device).to(torch.bfloat16)
    refiner = Agent(model, tokenizer, "refiner", d_model).to(device).to(torch.bfloat16)
    solver = Agent(model, tokenizer, "solver", d_model).to(device).to(torch.bfloat16)
    
    # 3. Загружаем Inner Adapters v2
    inner_path = "custom_impl/checkpoints/inner_link_qwen3_1_7b_adapter.pt"
    if os.path.exists(inner_path):
        print(f"\n📂 Loading Inner Adapters v2 from {inner_path}")
        checkpoint = torch.load(inner_path, map_location=device, weights_only=True)
        planner.inner_adapter.load_state_dict(checkpoint)
        refiner.inner_adapter.load_state_dict(checkpoint)
        solver.inner_adapter.load_state_dict(checkpoint)
        planner.inner_adapter.requires_grad_(False)
        refiner.inner_adapter.requires_grad_(False)
        solver.inner_adapter.requires_grad_(False)
        print("✅ Inner Adapters loaded and frozen")
    else:
        print(f"\n⚠️ Inner Adapters v2 not found at {inner_path}")
        print("   Running without pre-trained inner adapters...")
    
    # 4. Создаём Outer Adapters (outer_ln_res_adapter)
    print("\n🔗 Creating Outer Adapters (outer_ln_res_adapter)...")
    outer_12 = OuterAdapter(d_model, d_model).to(device).to(torch.bfloat16)  # planner → refiner
    outer_23 = OuterAdapter(d_model, d_model).to(device).to(torch.bfloat16)  # refiner → solver
    outer_31 = OuterAdapter(d_model, d_model).to(device).to(torch.bfloat16)  # solver → planner
    
    outer_params = sum(p.numel() for p in list(outer_12.parameters()) + 
                       list(outer_23.parameters()) + list(outer_31.parameters()))
    print(f"📊 Outer Adapter params: {outer_params:,}")
    print(f"📊 Per adapter: {outer_params//3:,}")
    
    # 5. Оптимизатор (betas=(0.9, 0.95))
    print("\n⚙️ Setting up optimizer...")
    
    # Создаём prediction head (lm_head заморожен)
    vocab_size = model.config.vocab_size
    prediction_head = nn.Linear(d_model, vocab_size, bias=False).to(device).to(torch.bfloat16)
    print(f"🎯 Prediction head: {d_model} → {vocab_size}")
    
    all_params = (list(outer_12.parameters()) + list(outer_23.parameters()) + 
                  list(outer_31.parameters()) + list(prediction_head.parameters()))
    optimizer = torch.optim.AdamW(all_params, lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"], betas=(0.9, 0.95))
    
    # 6. Scheduler
    num_update_steps = CONFIG["epochs"] * CONFIG["num_samples"] // CONFIG["batch_size"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG["warmup_steps"],
        num_training_steps=num_update_steps,
        num_cycles=0.5,
    )
    
    # 7. Датасет
    print("\n📚 Preparing dataset...")
    dataset = MathDataset(tokenizer, CONFIG["num_samples"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True,
                            collate_fn=lambda x: x)
    
    # 8. Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print("Pipeline: Planner → Outer(12) → Refiner → Outer(23) → Solver")
    print("          Solver → Outer(31) → Planner (feedback loop)")
    print()
    
    embedding_layer = model.get_input_embeddings()
    
    global_step = 0
    
    for epoch in range(CONFIG["epochs"]):
        epoch_losses = []
        
        for step, batch in enumerate(dataloader):
            questions = [s["question"] for s in batch]
            plans = [s["plan"] for s in batch]
            refined_plans = [s["refined_plan"] for s in batch]
            answers = [s["answer"] for s in batch]
            
            # Токенизируем
            q_tokens = tokenizer(questions, max_length=CONFIG["max_length"]//4, truncation=True, padding=True, return_tensors="pt")
            p_tokens = tokenizer(plans, max_length=CONFIG["max_length"]//4, truncation=True, padding=True, return_tensors="pt")
            rp_tokens = tokenizer(refined_plans, max_length=CONFIG["max_length"]//4, truncation=True, padding=True, return_tensors="pt")
            a_tokens = tokenizer(answers, max_length=CONFIG["max_length"]//4, truncation=True, padding=True, return_tensors="pt")
            
            q_ids = q_tokens["input_ids"].to(device)
            q_mask = q_tokens["attention_mask"].to(device)
            p_ids = p_tokens["input_ids"].to(device)
            rp_ids = rp_tokens["input_ids"].to(device)
            a_ids = a_tokens["input_ids"].to(device)
            
            # Unrolled recursion
            feedback_embeds = None
            round_losses = []
            
            for round_idx in range(CONFIG["recursion_rounds"]):
                # === PLANNER ===
                planner_outputs = planner(q_ids, q_mask, slot_embeds=feedback_embeds)
                planner_hidden = planner_outputs.hidden_states[-1][:, -CONFIG["latent_steps"]:, :]
                planner_inner = planner.inner_adapter(planner_hidden)
                planner_to_refiner = outer_12(planner_inner)
                
                # === REFINER ===
                refiner_outputs = refiner(q_ids, q_mask, slot_embeds=planner_to_refiner)
                refiner_hidden = refiner_outputs.hidden_states[-1][:, -CONFIG["latent_steps"]:, :]
                refiner_inner = refiner.inner_adapter(refiner_hidden)
                refiner_to_solver = outer_23(refiner_inner)
                
                # === SOLVER ===
                solver_outputs = solver(q_ids, q_mask, slot_embeds=refiner_to_solver)
                solver_hidden = solver_outputs.hidden_states[-1][:, -CONFIG["latent_steps"]:, :]
                solver_inner = solver.inner_adapter(solver_hidden)
                
                # Loss: prediction_head(solver_inner) vs answer
                solver_logits = prediction_head(solver_inner[:, -1, :])  # (batch, vocab)
                target = a_ids[:, 0]  # первый токен ответа
                loss = F.cross_entropy(solver_logits, target)
                
                round_losses.append(loss)
                
                # Feedback для следующего round'а
                if round_idx < CONFIG["recursion_rounds"] - 1:
                    feedback_embeds = outer_31(solver_inner)
            
            # Supervise final round only (как в оригинале)
            total_loss = round_losses[-1]
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=CONFIG["max_grad_norm"])
            
            optimizer.step()
            scheduler.step()
            global_step += 1
            
            epoch_losses.append(total_loss.item())
            
            if step % 5 == 0:
                print(f"  Epoch {epoch+1}/{CONFIG['epochs']}, Step {step}, "
                      f"Loss: {total_loss.item():.4f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\n  📊 Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
    
    # 9. Сохраняем
    os.makedirs(os.path.dirname(CONFIG["save_path"]), exist_ok=True)
    
    torch.save(outer_12.state_dict(), CONFIG["save_path"] + "_outer_12.pt")
    torch.save(outer_23.state_dict(), CONFIG["save_path"] + "_outer_23.pt")
    torch.save(outer_31.state_dict(), CONFIG["save_path"] + "_outer_31.pt")
    
    # Config
    config_dict = {
        "outer_adapter_type": "outer_ln_res_adapter",
        "in_dim": d_model,
        "out_dim": d_model,
        "recursion_rounds": CONFIG["recursion_rounds"],
        "latent_steps": CONFIG["latent_steps"],
    }
    with open(CONFIG["save_path"] + "_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n✅ Saved Outer Adapters to {CONFIG['save_path']}_*.pt")
    print(f"🎉 Outer Link training complete! (global_step={global_step})")


if __name__ == "__main__":
    train_outer_link()
