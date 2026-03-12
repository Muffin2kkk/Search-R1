# Search-R1 Agent Navigation Guide

This file is designed to help AI agents quickly understand the structure and key components of the **Search-R1** codebase without having to read through all the files.

## Project Overview
**Search-R1** is a reinforcement learning framework designed for training reasoning-and-searching interleaved LLMs. It extends the ideas of DeepSeek-R1(-Zero) by incorporating search engine access and provides a fully open-source RL training pipeline based on [veRL](https://github.com/volcengine/verl).

## Key Directories & Files

### 1. Core Package (`/search_r1/`)
This directory contains the main logic for the LLM agent and search engine interactions.
- **`search_r1/llm_agent/`**: Contains the logic for the LLM agent.
  - `generation.py`: Logic for generating text and handling tool calls.
  - `tensor_helper.py`: Helper functions for tensor operations.
- **`search_r1/search/`**: Contains various search engine and retrieval server implementations.
  - `retrieval_server.py`: The main local retrieval server (FastAPI).
  - `google_search_server.py` / `serp_search_server.py`: Servers for online search engines.
  - `rerank_server.py` / `retrieval_rerank_server.py`: Servers for reranking search results.
  - `index_builder.py` / `build_index.sh`: Scripts to build local dense/sparse indices for corpora.

### 2. Reinforcement Learning Framework (`/verl/`)
This directory contains the veRL framework, which handles the RL training loop (PPO, GRPO, etc.).
- **`verl/trainer/`**: Contains the core training loop and algorithms.
- **`verl/workers/`**: Contains the actor, critic, reward, and reference model workers.
- **`verl/models/`**: Contains model definitions and wrappers.

### 3. Training & Inference Entry Points (Root Directory)
- **`train_ppo.sh`**: Shell script to launch PPO training.
- **`train_grpo.sh`**: Shell script to launch GRPO training.
- **`infer.py`**: Script to run inference with a trained model.
- **`retrieval_launch.sh`**: Script to launch the local retrieval server before training/inference.

### 4. Data Processing (`/scripts/`)
- **`scripts/data_process/`**: Scripts to process datasets (e.g., `nq_search.py` for Natural Questions).
- **`scripts/download.py`**: Script to download required corpora and indices.

## Typical Workflows

1. **Data Preparation**: Run `scripts/download.py` to get the corpus, then use `scripts/data_process/nq_search.py` to prepare the QA data.
2. **Launch Search Server**: Run `bash retrieval_launch.sh` (usually in a separate environment) to start the local retrieval API.
3. **Training**: Run `bash train_ppo.sh` or `bash train_grpo.sh` to start the RL training process using the veRL framework.
4. **Inference**: Run `python infer.py` to test the model's reasoning and search capabilities.

## Agent Navigation Tips
- If you need to modify how the model calls the search tool or processes search results, look into `search_r1/llm_agent/generation.py`.
- If you need to add a new search engine or modify the retrieval logic, look into `search_r1/search/`.
- If you need to modify the RL training algorithm (e.g., PPO/GRPO hyperparameters, reward calculation), look into the `verl/` directory and the `train_*.sh` scripts.
- If you need to process a new dataset, create a new script in `scripts/data_process/` following the format in `nq_search.py`.
