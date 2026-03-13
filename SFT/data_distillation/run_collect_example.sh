#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/root/Search-R1"
SCRIPT_DIR="${ROOT_DIR}/SFT/data_distillation"

# Input / output
INPUT_DIR="${ROOT_DIR}/data/nq_hotpotqa_train"
OUTPUT_DIR="${ROOT_DIR}/data/nq_hotpotqa_train/strong_rollouts_qwen"
SPLITS="train,test"

# Retrieval
SEARCH_URL="http://127.0.0.1:8000/retrieve"
RETRIEVER_TOPK=3
RETRIEVAL_TIMEOUT=60
RETRIEVAL_CACHE_SIZE=10000

# Rollout
BATCH_SIZE=16
MAX_TURNS=2
LIMIT=""
OFFSET=0
WRITE_PARQUET="true"

# LLM
LLM_API_BASE="https://your-qwen-endpoint/v1"
LLM_API_PATH="/chat/completions"
LLM_MODEL="your-qwen-model-name"
LLM_API_KEY="${LLM_API_KEY:-YOUR_KEY}"
LLM_TIMEOUT=120
LLM_CONCURRENCY=8
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.95
LLM_MAX_TOKENS=512
LLM_PRESENCE_PENALTY=0.0
LLM_FREQUENCY_PENALTY=0.0
LLM_SEED=""
LLM_EXTRA_BODY=""

CMD=(
  python "${SCRIPT_DIR}/collect_strong_rollouts.py"
  --input_dir "${INPUT_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --splits "${SPLITS}"
  --batch_size "${BATCH_SIZE}"
  --max_turns "${MAX_TURNS}"
  --search_url "${SEARCH_URL}"
  --retriever_topk "${RETRIEVER_TOPK}"
  --retrieval_timeout "${RETRIEVAL_TIMEOUT}"
  --retrieval_cache_size "${RETRIEVAL_CACHE_SIZE}"
  --llm_api_base "${LLM_API_BASE}"
  --llm_api_path "${LLM_API_PATH}"
  --llm_model "${LLM_MODEL}"
  --llm_api_key "${LLM_API_KEY}"
  --llm_timeout "${LLM_TIMEOUT}"
  --llm_concurrency "${LLM_CONCURRENCY}"
  --llm_temperature "${LLM_TEMPERATURE}"
  --llm_top_p "${LLM_TOP_P}"
  --llm_max_tokens "${LLM_MAX_TOKENS}"
  --llm_presence_penalty "${LLM_PRESENCE_PENALTY}"
  --llm_frequency_penalty "${LLM_FREQUENCY_PENALTY}"
)

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

if [[ -n "${OFFSET}" ]]; then
  CMD+=(--offset "${OFFSET}")
fi

if [[ "${WRITE_PARQUET}" == "true" ]]; then
  CMD+=(--write_parquet)
fi

if [[ -n "${LLM_SEED}" ]]; then
  CMD+=(--llm_seed "${LLM_SEED}")
fi

if [[ -n "${LLM_EXTRA_BODY}" ]]; then
  CMD+=(--llm_extra_body "${LLM_EXTRA_BODY}")
fi

printf 'Running command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
