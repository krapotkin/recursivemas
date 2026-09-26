#!/usr/bin/env bash
# ============================================================
# Полный pipeline RecursiveMAS на Math500
# Последовательно: eval → train_inner → train_outer → eval
# ============================================================
set -euo pipefail

SCRIPT_DIR="$HOME/workspace/projects/recursivemas/custom_impl"
PROJECT_DIR="$HOME/workspace/projects/recursivemas"
VENV="$HOME/workspace/venvs/recursivemas/cuda_12_x"
CHECKPOINT_DIR="$HOME/workspace/tmp/recursivemas/checkpoints"
LOG_DIR="$HOME/workspace/tmp/recursivemas/logs"
DATA_DIR="$HOME/workspace/data/recursivemas/results"
DATASET="math500"
export CUDA_VISIBLE_DEVICES=1

# Модели — Planner и Refiner одна модель (экономия VRAM)
PLANNER_MODEL="$HOME/workspace/models/models_llm/Qwen3-1.7B"
REFINER_MODEL="$HOME/workspace/models/models_llm/Qwen3-1.7B"  # та же модель, что и Planner
SOLVER_MODEL="$HOME/workspace/models/models_llm/Qwen2.5-Math-1.5B-Instruct"

# Пути к чекпоинтам
INNER_PLANNER="$CHECKPOINT_DIR/inner_planner"
INNER_REFINER="$CHECKPOINT_DIR/inner_refiner"
INNER_SOLVER="$CHECKPOINT_DIR/inner_solver"
OUTER_DIR="$CHECKPOINT_DIR/outer"

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR" "$DATA_DIR"

cd "$SCRIPT_DIR"

###############################################################################
# Фаза 0: Быстрая проверка GPU
###############################################################################
echo "=== [0/5] Проверка GPU ==="
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

###############################################################################
# Фаза 1: Single Model eval (Planner и Solver)
###############################################################################
echo ""
echo "████████████████████████████████████████████████████████████████████████████"
echo "  [1/5] Single Model: Planner ($PLANNER_MODEL) → Math500"
echo "████████████████████████████████████████████████████████████████████████████"
source "$VENV/bin/activate"
#python3 evaluate_single_model.py \
#  --model "$PLANNER_MODEL" \
#  --dataset "$DATASET" \
#  --num_examples -1 \
#  --device cuda \
#  --output "$DATA_DIR/single_planner_math500.json" \
#  2>&1 | tee "$LOG_DIR/single_planner_math500.log"
echo "Результат: $(grep 'Accuracy' "$LOG_DIR/single_planner_math500.log" | tail -1)"

echo ""
echo "████████████████████████████████████████████████████████████████████████████"
echo "  [1/5] Single Model: Solver ($SOLVER_MODEL) → Math500"
echo "████████████████████████████████████████████████████████████████████████████"
source "$VENV/bin/activate"
#python3 evaluate_single_model.py \
#  --model "$SOLVER_MODEL" \
#  --dataset "$DATASET" \
#  --num_examples -1 \
#  --device cuda \
#  --output "$DATA_DIR/single_solver_math500.json" \
#  2>&1 | tee "$LOG_DIR/single_solver_math500.log"
echo "Результат: $(grep 'Accuracy' "$LOG_DIR/single_solver_math500.log" | tail -1)"

###############################################################################
# Фаза 2: Text MAS eval (Planner → Critic → Solver)
###############################################################################
echo ""
echo "████████████████████████████████████████████████████████████████████████████"
echo "  [2/5] Text MAS (Planner → Critic → Solver) → Math500"
echo "████████████████████████████████████████████████████████████████████████████"
source "$VENV/bin/activate"
#python3 evaluate_text_mas.py \
#  --dataset "$DATASET" \
#  --num_examples -1 \
#  --device cuda \
#  --output "$DATA_DIR/text_mas_math500.json" \
#  2>&1 | tee "$LOG_DIR/text_mas_math500.log"
echo "Результат: $(grep 'Accuracy' "$LOG_DIR/text_mas_math500.log" | tail -1)"

###############################################################################
# Фаза 3: Inner-loop обучение трёх агентов
###############################################################################
echo ""
echo "████████████████████████████████████████████████████████████████████████████"
echo "  [3/5] Inner-loop: Planner ($PLANNER_MODEL)"
echo "████████████████████████████████████████████████████████████████████████████"
source "$VENV/bin/activate"
#python3 train_inner.py \
#  --mas_role planner \
#  --model_name_or_path "$PLANNER_MODEL" \
#  --dataset_name RecursiveMAS/Sequential-Math \
#  --batch_size 8 \
#  --max_steps 20000 \
#  --adapter_lr 2e-4 \
#  --lr_scheduler_type cosine \
#  --warmup_steps 100 \
#  --max_length 2048 \
#  --save_dir "$INNER_PLANNER" \
#  --save_steps 0 \
#  2>&1 | tee "$LOG_DIR/train_inner_planner.log"
echo "Planner inner adapters обучены: $INNER_PLANNER"

echo ""
echo "████████████████████████████████████████████████████████████████████████████"
echo "  [3/5] Inner-loop: Refiner ($REFINER_MODEL)"
echo "████████████████████████████████████████████████████████████████████████████"
source "$VENV/bin/activate"
#python3 train_inner.py \
#  --mas_role refiner \
#  --model_name_or_path "$REFINER_MODEL" \
#  --dataset_name RecursiveMAS/Sequential-Math \
#  --batch_size 8 \
#  --max_steps 20000 \
#  --adapter_lr 2e-4 \
#  --lr_scheduler_type cosine \
#  --warmup_steps 100 \
#  --max_length 2048 \
#  --save_dir "$INNER_REFINER" \
#  --save_steps 0 \
#  2>&1 | tee "$LOG_DIR/train_inner_refiner.log"
echo "Refiner inner adapters обучены: $INNER_REFINER"

echo ""
echo "████████████████████████████████████████████████████████████████████████████"
echo "  [3/5] Inner-loop: Solver ($SOLVER_MODEL)"
echo "████████████████████████████████████████████████████████████████████████████"
source "$VENV/bin/activate"
#python3 train_inner.py \
#  --mas_role solver \
#  --model_name_or_path "$SOLVER_MODEL" \
#  --dataset_name RecursiveMAS/Sequential-Math \
#  --batch_size 8 \
#  --max_steps 20000 \
#  --adapter_lr 2e-4 \
#  --lr_scheduler_type cosine \
#  --warmup_steps 100 \
#  --max_length 2048 \
#  --save_dir "$INNER_SOLVER" \
#  --save_steps 0 \
#  2>&1 | tee "$LOG_DIR/train_inner_solver.log"
echo "Solver inner adapters обучены: $INNER_SOLVER"

###############################################################################
# Фаза 4: Outer-loop обучение с aligned init
###############################################################################
echo ""
echo "████████████████████████████████████████████████████████████████████████████"
echo "  [4/5] Outer-loop: Planner → Refiner → Solver (aligned init)"
echo "████████████████████████████████████████████████████████████████████████████"
source "$VENV/bin/activate"
python3 train_outer.py \
  --agent1_model "$PLANNER_MODEL" \
  --agent2_model "$REFINER_MODEL" \
  --agent3_model "$SOLVER_MODEL" \
  --agent1_inner "$INNER_PLANNER" \
  --agent2_inner "$INNER_REFINER" \
  --agent3_inner "$INNER_SOLVER" \
  --dataset_name RecursiveMAS/Sequential-Math \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --num_train_epochs 1 \
  --max_steps 5000 \
  --outer_lr 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_steps 10 \
  --supervise_final_only 1 \
  --non_last_loss_weight 0.1 \
  --num_recursive_rounds 3 \
  --max_latent_tokens 80 \
  --align_outer 1 \
  --align_samples 300 \
  --save_dir "$OUTER_DIR" \
  --save_steps 1000 \
  --device cuda \
  2>&1 | tee "$LOG_DIR/train_outer_aligned.log"
echo "Outer adapters обучены: $OUTER_DIR"

###############################################################################
# Фаза 5: RecursiveMAS eval с обученными адаптерами
###############################################################################
echo ""
echo "████████████████████████████████████████████████████████████████████████████"
echo "  [5/5] RecursiveMAS eval (с обученными адаптерами) → Math500"
echo "████████████████████████████████████████████████████████████████████████████"
source "$VENV/bin/activate"
python3 evaluate_recursivemas.py \
  --dataset "$DATASET" \
  --num_examples -1 \
  --device cuda \
  --outer_checkpoint "$OUTER_DIR" \
  --inner1_checkpoint "$INNER_PLANNER" \
  --inner2_checkpoint "$INNER_REFINER" \
  --inner3_checkpoint "$INNER_SOLVER" \
  --num_latent_tokens 32 \
  --num_recursive_rounds 3 \
  --output "$DATA_DIR/recursivemas_math500.json" \
  2>&1 | tee "$LOG_DIR/recursivemas_math500.log"
echo "Результат: $(grep 'Accuracy' "$LOG_DIR/recursivemas_math500.log" | tail -1)"

###############################################################################
# Итоговый отчёт
###############################################################################
echo ""
echo "████████████████████████████████████████████████████████████████████████████"
echo "  ИТОГОВЫЕ РЕЗУЛЬТАТЫ на Math500"
echo "████████████████████████████████████████████████████████████████████████████"
echo ""
for log in "$LOG_DIR"/single_planner_math500.log "$LOG_DIR"/single_solver_math500.log \
           "$LOG_DIR"/text_mas_math500.log "$LOG_DIR"/recursivemas_math500.log; do
  base=$(basename "$log" .log)
  acc=$(grep -oP 'Accuracy:\s+\d+\.\d+%' "$log" | tail -1 || grep -oP 'accuracy=\d+\.\d+%' "$log" | tail -1 || echo "N/A")
  echo "  ${base}: $acc"
done
echo ""
echo "Все результаты сохранены в: $DATA_DIR"
echo "Логи: $LOG_DIR"
