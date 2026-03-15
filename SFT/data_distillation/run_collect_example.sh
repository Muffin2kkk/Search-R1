#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/root/Search-R1"
SCRIPT_DIR="${ROOT_DIR}/SFT/data_distillation"

# Input / output
INPUT_DIR="${ROOT_DIR}/data/nq_hotpotqa_tiny/distillation"
OUTPUT_DIR="${ROOT_DIR}/data/nq_hotpotqa_tiny/distillation"
SPLITS="train"

# Retrieval
SEARCH_URL="http://127.0.0.1:8000/retrieve"
RETRIEVER_TOPK=3
RETRIEVAL_TIMEOUT=60
RETRIEVAL_CACHE_SIZE=10000
TOKENIZER_PATH="/root/autodl-tmp/Qwen2.5-3B-Instruct"
MAX_PROMPT_LENGTH=6144
MAX_OBS_LENGTH=500

# Rollout
BATCH_SIZE=16
MAX_TURNS=3
LIMIT=""
OFFSET=0
WRITE_PARQUET="false"

# LLM
LLM_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_API_PATH="/chat/completions"
LLM_MODEL="qwen3.5-plus"
LLM_TIMEOUT=120
LLM_CONCURRENCY=8
LLM_TEMPERATURE=0.3
LLM_TOP_P=0.8
LLM_MAX_TOKENS=512
LLM_PRESENCE_PENALTY=0.0
LLM_FREQUENCY_PENALTY=0.0
LLM_SEED=""
LLM_EXTRA_BODY='{"enable_thinking": false}'

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
  --tokenizer_path "${TOKENIZER_PATH}"
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --max_obs_length "${MAX_OBS_LENGTH}"
  --llm_api_base "${LLM_API_BASE}"
  --llm_api_path "${LLM_API_PATH}"
  --llm_model "${LLM_MODEL}"
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
