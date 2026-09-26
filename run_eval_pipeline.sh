#!/usr/bin/env bash
# Automated evaluation pipeline for RecursiveMAS
# Runs evaluations sequentially with logging

set -euo pipefail

cd "$(dirname "$0")"
source .venv

RESULTS_DIR="$HOME/workspace/data/recursivemas/results"
LOGS_DIR="$HOME/workspace/tmp/recursivemas"

echo "=== RecursiveMAS Evaluation Pipeline ==="
echo "Started: $(date)"
echo ""

# Step 1: Qwen2.5-Math-1.5B-Instruct baseline
echo "=== Step 1: Qwen2.5-Math-1.5B-Instruct baseline ==="
python3 -u custom_impl/evaluate_single_model.py \
    --model "$HOME/workspace/models/models_llm/Qwen2.5-Math-1.5B-Instruct" \
    --num_examples 1319 \
    --device cuda:2 \
    --output "${RESULTS_DIR}/eval_single_qwen25_math_15b_full.json" \
    2>&1 | tee "${LOGS_DIR}/eval_single_qwen25_math_15b_full.log"

echo ""
echo "Step 1 complete: $(date)"
echo ""

# Step 2: Qwen3-1.7B baseline
echo "=== Step 2: Qwen3-1.7B baseline ==="
python3 -u custom_impl/evaluate_single_model.py \
    --model "$HOME/workspace/models/models_llm/Qwen3-1.7B" \
    --num_examples 1319 \
    --device cuda:2 \
    --output "${RESULTS_DIR}/eval_single_qwen3_17b_full.json" \
    2>&1 | tee "${LOGS_DIR}/eval_single_qwen3_17b_full.log"

echo ""
echo "Step 2 complete: $(date)"
echo ""

# Step 3: Text MAS evaluation
echo "=== Step 3: Text MAS evaluation ==="
python3 -u custom_impl/evaluate_text_mas.py \
    --num_examples 1319 \
    --device cuda:2 \
    --output "${RESULTS_DIR}/eval_text_mas_full.json" \
    2>&1 | tee "${LOGS_DIR}/eval_text_mas_full.log"

echo ""
echo "Step 3 complete: $(date)"
echo ""

echo "=== All evaluations complete ==="
echo "Finished: $(date)"
