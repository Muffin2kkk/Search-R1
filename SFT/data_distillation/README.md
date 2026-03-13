# SFT Data Distillation

这个目录用于通过远端强模型 API + 本地检索服务，采集适合 SFT 的搜索式 rollout 数据。

目录中的脚本有：

- `strong_model_client.py`
  远端强模型的统一适配层，当前按 OpenAI-compatible `chat/completions` 接口封装。
- `strong_rollout_collector.py`
  rollout 编排器，负责多轮对话状态、本地检索调用、单次搜索和并行搜索解析。
- `collect_strong_rollouts.py`
  离线采集脚本，读取 parquet 数据集并生成强模型 rollout 数据。
- `prepare_sft_dataset.py`
  后处理脚本，将 rollout 数据裁剪成 SFT 真正需要的最小字段。
- `run_collect_example.sh`
  采集示例启动脚本，适合先改变量再直接运行。

## 功能概览

当前采集链路支持：

- 单次搜索：`<search> query </search>`
- 并行搜索：`<searches> <search> query1 </search> <search> query2 </search> </searches>`
- 最终答案：`<answer> final answer </answer>`
- 本地检索批量调用
- 简单检索缓存
- 可配置远端 LLM 超参数

## 前置条件

在运行这些脚本之前，需要先准备两部分：

1. 本地检索服务

确保你的检索服务已经启动，例如：

```bash
python /root/Search-R1/search_r1/search/retrieval_server.py
```

默认检索地址是：

```text
http://127.0.0.1:8000/retrieve
```

2. 待采集的数据集

输入目录需要包含类似下面的文件：

```text
train.parquet
test.parquet
```

每条数据至少应包含：

- `prompt`
- 可选的 `extra_info.index`

## 推荐使用流程

推荐按下面顺序执行：

1. 启动本地检索服务
2. 修改并运行 `run_collect_example.sh`，或直接运行 `collect_strong_rollouts.py`
3. 运行 `prepare_sft_dataset.py` 裁剪成 SFT 训练需要的最小字段

## 快速启动

如果你不想每次手写一长串参数，推荐直接使用：

```text
run_collect_example.sh
```

使用方式：

1. 打开 `run_collect_example.sh`
2. 修改其中的变量，至少包括：

- `INPUT_DIR`
- `OUTPUT_DIR`
- `LLM_API_BASE`
- `LLM_MODEL`
- `LLM_API_KEY`

3. 运行：

```bash
bash /root/Search-R1/SFT/data_distillation/run_collect_example.sh
```

脚本里已经包含了：

- 检索地址
- batch 大小
- 最大搜索轮数
- LLM 并发数
- 常见采样参数
- 是否额外写 parquet

你只需要按需改变量即可。

## 1. 采集强模型 Rollouts

脚本：

```text
collect_strong_rollouts.py
```

作用：

- 读取输入 parquet
- 调用远端强模型 API
- 当模型输出 `<search>` 或 `<searches>` 时，调用本地 `/retrieve`
- 将检索结果包装成 `<information> ... </information>` 回填给模型
- 达到 `max_turns` 后强制要求模型输出 `<answer>`
- 输出 rollout 数据为 `jsonl`，可选同时写 `parquet`

### 常用参数

- `--input_dir`
  输入数据目录，包含 `train.parquet`、`test.parquet`
- `--output_dir`
  rollout 输出目录
- `--splits`
  要处理的数据切分，如 `train,test`
- `--batch_size`
  每批处理的样本数
- `--max_turns`
  最大搜索轮数
- `--search_url`
  本地检索服务地址
- `--retriever_topk`
  每个 query 返回多少条检索结果
- `--retrieval_cache_size`
  检索缓存大小
- `--llm_api_base`
  远端 LLM API 的 base URL
- `--llm_api_path`
  默认是 `/chat/completions`
- `--llm_api_key`
  API key
- `--llm_model`
  强模型名称
- `--llm_concurrency`
  单批内部并发请求数
- `--llm_temperature`
- `--llm_top_p`
- `--llm_max_tokens`
- `--llm_seed`
- `--llm_extra_body`
  额外透传给远端接口的 JSON 字符串
- `--write_parquet`
  除 jsonl 外，也额外写出 parquet

### 运行示例

```bash
python /root/Search-R1/SFT/data_distillation/collect_strong_rollouts.py \
  --input_dir /root/Search-R1/data/nq_hotpotqa_train \
  --output_dir /root/Search-R1/data/nq_hotpotqa_train/strong_rollouts_qwen \
  --splits train,test \
  --batch_size 16 \
  --max_turns 2 \
  --search_url http://127.0.0.1:8000/retrieve \
  --retriever_topk 3 \
  --retrieval_cache_size 10000 \
  --llm_api_base https://your-qwen-endpoint/v1 \
  --llm_api_path /chat/completions \
  --llm_model your-qwen-model-name \
  --llm_api_key YOUR_KEY \
  --llm_concurrency 8 \
  --llm_temperature 0.7 \
  --llm_top_p 0.95 \
  --llm_max_tokens 512 \
  --write_parquet
```

### 使用示例脚本

如果你更喜欢先改配置再直接执行，可以用：

```bash
bash /root/Search-R1/SFT/data_distillation/run_collect_example.sh
```

默认脚本中：

- `LLM_API_KEY` 会优先读取环境变量 `LLM_API_KEY`
- 如果你没在环境变量里设置，就会回退到脚本里的占位值

例如：

```bash
export LLM_API_KEY="your_real_key"
bash /root/Search-R1/SFT/data_distillation/run_collect_example.sh
```

### 输出字段

采集输出会保留原始样本字段，并新增：

- `messages`
- `rollout_turns`
- `final_answer`
- `final_response`
- `collector_status`
- `collector_error`
- `search_turn_count`
- `invalid_action_count`
- `strong_model_usage`

其中：

- `messages` 适合直接作为 SFT 的对话输入
- `rollout_turns` 适合调试或分析
- `collector_status=completed` 表示正常拿到了 `<answer>`

## 2. 裁剪为 SFT 最小数据

脚本：

```text
prepare_sft_dataset.py
```

作用：

- 从 rollout 数据中提取最小 SFT 字段
- 默认只保留 `messages`
- 可选保留少量辅助字段
- 支持输入 `jsonl` 或 `parquet`
- 支持输出 `jsonl` 或 `parquet`

### 常用参数

- `--input_path`
  输入 rollout 文件，支持 `.jsonl` 或 `.parquet`
- `--output_path`
  输出 SFT 文件，支持 `.jsonl` 或 `.parquet`
- `--completed_only`
  仅保留 `collector_status=completed` 的样本
- `--keep_data_source`
- `--keep_ability`
- `--keep_extra_info`
- `--keep_rollout_turns`

### 运行示例

只保留 `messages`：

```bash
python /root/Search-R1/SFT/data_distillation/prepare_sft_dataset.py \
  --input_path /root/Search-R1/data/nq_hotpotqa_train/strong_rollouts_qwen/train.jsonl \
  --output_path /root/Search-R1/data/nq_hotpotqa_train/strong_rollouts_qwen/train_sft.jsonl \
  --completed_only
```

保留部分辅助字段：

```bash
python /root/Search-R1/SFT/data_distillation/prepare_sft_dataset.py \
  --input_path /root/Search-R1/data/nq_hotpotqa_train/strong_rollouts_qwen/train.parquet \
  --output_path /root/Search-R1/data/nq_hotpotqa_train/strong_rollouts_qwen/train_sft.parquet \
  --completed_only \
  --keep_data_source \
  --keep_extra_info
```

## 3. 两个底层模块的职责

### `strong_model_client.py`

这是远端强模型调用层。

主要职责：

- 把请求封装为 OpenAI-compatible chat completion 格式
- 统一管理模型名、API 地址、采样参数
- 支持批量并发请求

如果你后面想切换到别的兼容网关，通常只需要改这里。

### `strong_rollout_collector.py`

这是 rollout 的核心调度层。

主要职责：

- 维护每条样本的对话状态
- 解析 `<search>` / `<searches>` / `<answer>`
- 批量调用本地检索服务
- 将检索结果格式化为 `<information> ... </information>`
- 达到最大轮数后要求模型直接回答

如果你后面想加更复杂的控制逻辑，例如：

- 更强的失败重试
- 更复杂的消息裁剪
- 更丰富的输出格式

通常改这个文件即可。

## 常见建议

- 如果你的远端强模型比较贵，可以先加 `--limit 100` 小规模试跑。
- 如果 API 不稳定，先把 `--batch_size` 和 `--llm_concurrency` 调小。
- 如果上下文容易过长，优先减小 `--max_turns` 或 prompt 长度。
- 如果同类问题很多，当前自带的检索缓存会对吞吐有帮助。

## 一个最小可跑示例

先采集：

```bash
export LLM_API_KEY="your_real_key"
bash /root/Search-R1/SFT/data_distillation/run_collect_example.sh
```

再裁成 SFT：

```bash
python /root/Search-R1/SFT/data_distillation/prepare_sft_dataset.py \
  --input_path /root/Search-R1/data/nq_hotpotqa_train/strong_rollouts_qwen/train.jsonl \
  --output_path /root/Search-R1/data/nq_hotpotqa_train/strong_rollouts_qwen/train_sft.jsonl \
  --completed_only
```

## 说明

这套脚本目前没有自动启动本地检索服务，也不会帮你检查远端 API 是否可用。  
建议先确认：

- 本地 `retrieve` 接口能访问
- 远端 Qwen 接口的 `base_url / model / key` 正确
- 输入数据的 `prompt` 字段格式符合预期
- `run_collect_example.sh` 中的路径和变量已经改成你的实际配置
