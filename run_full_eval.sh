#!/usr/bin/env bash
# Sequential evaluation runner — runs all three evaluations in order
# Usage: bash run_full_eval.sh

set -euo pipefail

cd "$(dirname "$0")"
source .venv

echo "=== Step 1: Qwen2.5-Math-1.5B-Instruct (1319 examples) ==="
#python3 -u custom_impl/evaluate_single_model.py \
#  --model "$HOME/workspace/models/models_llm/Qwen2.5-Math-1.5B-Instruct" \
#  --num_examples 1319 \
#  --device cuda:2 \
#  --output ~/workspace/data/recursivemas/results/eval_single_qwen25_math_15b_full.json \
#  2>&1 | tee ~/workspace/tmp/recursivemas/eval_single_qwen25_math_15b_full.log

echo ""
echo "=== Step 2: Qwen3-1.7B (1319 examples) ==="
#python3 -u custom_impl/evaluate_single_model.py \
#  --model "$HOME/workspace/models/models_llm/Qwen3-1.7B" \
#  --num_examples 1319 \
#  --device cuda:2 \
#  --output ~/workspace/data/recursivemas/results/eval_single_qwen3_17b_full.json \
#  2>&1 | tee ~/workspace/tmp/recursivemas/eval_single_qwen3_17b_full.log

echo ""
echo "=== Step 3: Text MAS (1319 examples) ==="
#python3 -u custom_impl/evaluate_text_mas.py \
#  --num_examples 1319 \
#  --device cuda:2 \
#  --output ~/workspace/data/recursivemas/results/eval_text_mas_full.json \
#  2>&1 | tee ~/workspace/tmp/recursivemas/eval_text_mas_full.log

echo ""
echo "=== All evaluations complete ==="
