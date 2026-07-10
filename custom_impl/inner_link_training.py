"""
⚠️ УСТАРЕЛ — используйте custom_impl/train_inner.py

Этот скрипт использует синтетические данные (не Sequential-Math) и устаревший формат.
Для воспроизведения оригинальной реализации используйте:
    python custom_impl/train_inner.py --mas_role planner --model_name_or_path Qwen/Qwen3-1.7B

Практическое задание: Обучение Inner Link (исправленная версия)

Следует оригинальной реализации из train/model.py и train/train_inner.py.

Исправления относительно v1:
1. ✅ Target: pairwise alignment (hidden[:, :-1] → embeds[:, 1:])
2. ✅ Loss: cosine + 0.1 * MSE
3. ✅ LayerNorm: pre + post (ln_res_adapter)
4. ✅ Optimizer: AdamW(betas=(0.9, 0.95))
5. ✅ Gradient clipping: max_norm=1.0
6. ✅ Scheduler: cosine with warmup
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import json
import os
import random
import math

# ==================== Конфигурация ====================
CONFIG = {
    "model_name": "Qwen/Qwen3-1.7B",
    "device": "cuda:2",
    "batch_size": 4,
    "epochs": 3,
    "lr": 5e-4,
    "warmup_steps": 10,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "max_question_len": 128,
    "max_answer_len": 256,
    "num_samples": 100,
    "save_path": "custom_impl/checkpoints/inner_link_qwen3_1_7b_v2",
}


# ==================== Inner RecursiveLink (ln_res_adapter) ====================
class InnerAdapter(nn.Module):
    """
    ln_res_adapter из оригинала:
    LayerNorm → W1 → GELU → W2 → +residual → LayerNorm
    """
    
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
        out = x + out  # residual
        out = self.post_ln(out)
        return out


# ==================== Датасет ====================
class MathDataset(Dataset):
    def __init__(self, tokenizer, num_samples=100):
        self.tokenizer = tokenizer
        self.samples = self._generate_samples(num_samples)
    
    def _generate_samples(self, n):
        samples = []
        for i in range(n):
            task_type = random.choice(['linear', 'quadratic', 'arithmetic', 'geometry'])
            
            if task_type == 'linear':
                a, b, c = random.randint(1, 10), random.randint(-10, 10), random.randint(-20, 20)
                x = -(b + c) / a if a != 0 else 0
                question = f"Solve for x: {a}x + {b} = {c}"
                answer = f"x = {x:.2f}"
            elif task_type == 'quadratic':
                a, b, c = random.randint(1, 5), random.randint(-10, 10), random.randint(-20, 20)
                d = b*b - 4*a*c
                if d >= 0:
                    x1 = (-b + math.sqrt(d)) / (2*a)
                    x2 = (-b - math.sqrt(d)) / (2*a)
                    question = f"Solve: {a}x² + {b}x + {c} = 0"
                    answer = f"x₁ = {x1:.2f}, x₂ = {x2:.2f}"
                else:
                    question = f"Is x² + {b}x + {c} = 0 solvable in reals?"
                    answer = "No, discriminant is negative"
            elif task_type == 'arithmetic':
                a, b = random.randint(1, 100), random.randint(1, 100)
                op = random.choice(['+', '-', '*'])
                result = eval(f"{a} {op} {b}")
                question = f"Calculate: {a} {op} {b}"
                answer = f"{result}"
            else:
                r = random.randint(1, 10)
                area = math.pi * r * r
                question = f"Find the area of a circle with radius {r}"
                answer = f"Area = {area:.2f}"
            
            samples.append({"question": question, "answer": answer})
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Объединяем вопрос и ответ в одну последовательность
        # Как в оригинале: модель видит вопрос, потом ответ
        full_text = f"{sample['question']} {sample['answer']}"
        
        tokens = self.tokenizer(
            full_text,
            max_length=CONFIG["max_question_len"] + CONFIG["max_answer_len"],
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Находим границу между вопросом и ответом
        question_tokens = self.tokenizer(
            sample["question"],
            truncation=True,
            return_tensors="pt"
        )
        question_len = question_tokens["input_ids"].shape[1]
        
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "question_len": question_len,  # граница question → answer
        }


# ==================== Training ====================
def train_inner_link():
    print("=" * 60)
    print("INNER LINK TRAINING (v2 — исправленная версия)")
    print("=" * 60)
    
    device = torch.device(CONFIG["device"])
    print(f"\n📍 Device: {device}")
    print(f"📦 Model: {CONFIG['model_name']}")
    print(f"📊 Samples: {CONFIG['num_samples']}")
    print(f"🔄 Epochs: {CONFIG['epochs']}")
    print(f"📏 Batch size: {CONFIG['batch_size']}")
    
    # 1. Загружаем модель
    print("\n🔄 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_name"],
        trust_remote_code=True,
        use_fast=True,
    )
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
    
    # 2. Создаём Inner Adapter (ln_res_adapter)
    print(f"\n🔗 Creating Inner Adapter (ln_res_adapter, {d_model})...")
    adapter = InnerAdapter(d_model).to(device).to(torch.bfloat16)
    
    total_params = sum(p.numel() for p in adapter.parameters())
    model_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Adapter params: {total_params:,}")
    print(f"📊 Model params: {model_params:,}")
    print(f"📊 Ratio: {total_params/model_params*100:.2f}%")
    
    # 3. Оптимизатор (betas=(0.9, 0.95) как в оригинале)
    print("\n⚙️ Setting up optimizer...")
    decay, no_decay = [], []
    for name, param in adapter.named_parameters():
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    
    param_groups = [
        {"params": decay, "weight_decay": CONFIG["weight_decay"], "lr": CONFIG["lr"]},
    ]
    if no_decay:
        param_groups.append({"params": no_decay, "weight_decay": 0.0, "lr": CONFIG["lr"]})
    
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
    
    # 4. Датасет
    print("\n📚 Preparing dataset...")
    dataset = MathDataset(tokenizer, CONFIG["num_samples"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    
    # 5. Scheduler (cosine with warmup)
    num_update_steps = CONFIG["epochs"] * len(dataloader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG["warmup_steps"],
        num_training_steps=num_update_steps,
        num_cycles=0.5,
    )
    
    # 6. Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print("Loss: cosine + 0.1 * MSE (pairwise alignment)")
    print()
    
    embedding_layer = model.get_input_embeddings()
    
    global_step = 0
    
    for epoch in range(CONFIG["epochs"]):
        epoch_losses = []
        epoch_cos = []
        epoch_mse = []
        
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            question_lens = batch["question_len"]  # scalar per sample
            
            # Forward pass через ЗАМРОЖЕННУЮ модель
            with torch.no_grad():
                # Получаем input embeddings
                input_embeds = embedding_layer(input_ids)  # (batch, seq_len, d_model)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, d_model)
            
            # === Pairwise alignment (как в оригинале) ===
            # hidden[:, :-1] → target: input_embeds[:, 1:]
            # Это учит адаптер предсказывать следующий embedding из текущего hidden state
            
            # Маска: только токены ответа (после вопроса)
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            
            # Создаём маску: 1 для токенов ответа, 0 для токенов вопроса
            answer_mask = torch.zeros(batch_size, seq_len - 1, device=device)
            for i in range(batch_size):
                q_len = question_lens[i].item()
                # Токены ответа: от q_len до seq_len-1
                # Для pairwise: hidden[t] → embed[t+1], значит t от q_len-1 до seq_len-2
                if q_len > 1:
                    answer_mask[i, q_len-1:seq_len-1] = 1.0
            
            # Применяем адаптер к hidden states (кроме последнего токена)
            hidden_prev = hidden_states[:, :-1, :]  # (batch, seq_len-1, d_model)
            target_embeds = input_embeds[:, 1:, :]   # (batch, seq_len-1, d_model)
            
            # Проходим через адаптер
            preds = adapter(hidden_prev)  # (batch, seq_len-1, d_model)
            
            # Loss computation (как в оригинале: float32 для стабильности)
            preds_float = preds.float()
            targets_float = target_embeds.float()
            
            # Cosine loss
            cosine = 1.0 - torch.nn.functional.cosine_similarity(preds_float, targets_float, dim=-1)
            cosine_loss = (cosine * answer_mask).sum() / answer_mask.sum().clamp(min=1.0)
            
            # MSE loss
            mse = (preds_float - targets_float).pow(2).mean(dim=-1)
            mse_loss = (mse * answer_mask).sum() / answer_mask.sum().clamp(min=1.0)
            
            # Total loss: cosine + 0.1 * mse (как в оригинале)
            loss = 1.0 * cosine_loss + 0.1 * mse_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=CONFIG["max_grad_norm"])
            
            optimizer.step()
            scheduler.step()
            global_step += 1
            
            epoch_losses.append(loss.item())
            epoch_cos.append(cosine_loss.item())
            epoch_mse.append(mse_loss.item())
            
            if step % 10 == 0:
                print(f"  Epoch {epoch+1}/{CONFIG['epochs']}, Step {step}, "
                      f"Loss: {loss.item():.4f}, Cos: {cosine_loss.item():.4f}, MSE: {mse_loss.item():.4f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_cos = sum(epoch_cos) / len(epoch_cos)
        avg_mse = sum(epoch_mse) / len(epoch_mse)
        print(f"\n  📊 Epoch {epoch+1}: Loss={avg_loss:.4f}, Cos={avg_cos:.4f}, MSE={avg_mse:.4f}")
    
    # 7. Сохраняем
    os.makedirs(os.path.dirname(CONFIG["save_path"]), exist_ok=True)
    
    # Сохраняем adapter weights
    torch.save(adapter.state_dict(), CONFIG["save_path"] + "_adapter.pt")
    
    # Сохраняем config
    config_dict = {
        "adapter_type": "ln_res_adapter",
        "hidden_size": d_model,
        "config": CONFIG,
    }
    with open(CONFIG["save_path"] + "_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n✅ Saved adapter to {CONFIG['save_path']}_adapter.pt")
    print(f"✅ Saved config to {CONFIG['save_path']}_config.json")
    print(f"🎉 Training complete! (global_step={global_step})")
    
    return adapter


if __name__ == "__main__":
    train_inner_link()
