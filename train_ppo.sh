export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR='data/nq_hotpotqa_train'

WAND_PROJECT='Search-R1'

# export BASE_MODEL='meta-llama/Llama-3.2-3B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.2-3b-em
# export BASE_MODEL='/root/autodl-tmp/Qwen3-4B-Instruct-2507'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen3-4b-it-em
export BASE_MODEL='/root/autodl-tmp/Qwen2.5-3B-Instruct'
export EXPERIMENT_NAME=nqhotpotqa-search-r1-ppo-qwen2.5-3b-ng
# export BASE_MODEL='meta-llama/Llama-3.2-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.2-3b-it-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.1-8b-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.1-8b-it-em

# export BASE_MODEL='Qwen/Qwen2.5-3B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-3b-em
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-3b-it-em
# export BASE_MODEL='Qwen/Qwen2.5-7B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-7b-em
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-7b-it-em

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

# ==============================================================================
# PPO 训练超参数说明：
#
# [数据相关 (data)]
# data.train_files / val_files: 训练集和验证集的数据路径
# data.train_data_num / val_data_num: 使用的数据量限制（null表示使用全部数据）
# data.train_batch_size / val_batch_size: 训练和验证的全局批次大小 (Global Batch Size)
# data.max_prompt_length: 最大提示词 (Prompt) 长度
# data.max_response_length: 模型生成的最大回复长度
# data.max_start_length: 初始输入文本的最大长度
# data.max_obs_length: 环境/单次检索返回的观察值 (Observation) 的最大长度
# data.shuffle_train_dataloader: 是否打乱训练数据
#
# [算法相关 (algorithm)]
# algorithm.adv_estimator: 优势函数估计方法，gae表示广义优势估计 (Generalized Advantage Estimation)
# algorithm.kl_ctrl.kl_coef: KL散度惩罚系数，用于限制Actor模型更新幅度，防止偏离参考模型太远
# algorithm.no_think_rl: 是否禁用思考过程的强化学习（设为false表示启用）
#
# [Actor模型与生成 (actor_rollout_ref)]
# actor_rollout_ref.model.path: Actor 策略模型的初始权重路径
# actor_rollout_ref.actor.optim.lr: Actor 模型的学习率
# actor_rollout_ref.model.enable_gradient_checkpointing: 开启梯度检查点，用计算时间换取显存空间
# actor_rollout_ref.actor.optim.lr_warmup_steps_ratio: 学习率预热 (Warmup) 步数占总步数的比例
# actor_rollout_ref.actor.ppo_mini_batch_size: PPO 算法更新时的 Mini-batch 大小
# actor_rollout_ref.actor.ppo_micro_batch_size: Actor 单张显卡单次前向/反向传播的 Micro-batch 大小
# actor_rollout_ref.actor.fsdp_config.*_offload: 开启 FSDP 的参数/梯度/优化器状态 CPU 卸载，大幅节省显存
# actor_rollout_ref.rollout.name: 推理生成后端，这里使用 vllm 加速生成
# actor_rollout_ref.rollout.tensor_model_parallel_size: vLLM 生成时的张量并行 (TP) 大小
# actor_rollout_ref.rollout.gpu_memory_utilization: vLLM 引擎的 GPU 显存利用率上限
# actor_rollout_ref.rollout.temperature: 文本生成的温度参数，控制探索的随机性
# actor_rollout_ref.ref.*: Reference (参考) 模型的配置，用于计算 KL 散度
#
# [Critic模型 (critic)]
# critic.model.path: Critic 价值模型的初始权重路径（通常与 Actor 相同）
# critic.optim.lr: Critic 模型的学习率（通常比 Actor 稍大）
# critic.ppo_micro_batch_size: Critic 的 Micro-batch 大小
# critic.model.fsdp_config.*_offload: Critic 模型的 FSDP CPU 卸载配置
#
# [训练器配置 (trainer)]
# trainer.n_gpus_per_node: 每台机器使用的 GPU 数量
# trainer.save_freq / test_freq: 模型保存和验证的频率（以步数为单位）
# trainer.total_epochs / total_training_steps: 总训练轮数和总训练步数
# trainer.default_local_dir: 模型 Checkpoint 的本地保存目录
#
# [环境与检索器 (Environment & Retriever)]
# max_turns: 智能体与环境交互的最大轮数（用于多轮搜索/对话）
# retriever.url: 外部检索引擎的 API 服务地址
# retriever.topk: 每次检索返回的 Top-K 文档数量
# ==============================================================================

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=6144 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=False \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=4 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=3 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log