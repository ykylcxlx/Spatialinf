#!/usr/bin/env bash
# Execute the memory QA generation pipeline for a predefined episode directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPISODE_DIR="/data5/zhuangyunhao/outputs/video/train_15_TissueBox_20251210_125749"
OUTPUT_DIR="/data5/zhuangyunhao/outputs/memory"

python "${SCRIPT_DIR}/generate_memory_qas.py" \
  "${EPISODE_DIR}" \
  --agent-prefix "agent_1" \
  --min-objects 2 \
  --max-objects 4 \
  --questions-per-size 10 \
  --cooccur-questions 12 \
  --cooccur-min-overlap 1 \
  --cooccur-max-options 4 \
  --earliest-questions 8 \
  --first-frame-questions 12 \
  --seed 2025 \
  --output-dir "${OUTPUT_DIR}" \
  --output-name "memory_order_questions.json"
