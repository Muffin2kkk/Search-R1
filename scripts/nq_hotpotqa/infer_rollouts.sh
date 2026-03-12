export CUDA_VISIBLE_DEVICES=0,1
export DATA_DIR='data/nq_hotpotqa_train'

WAND_PROJECT='Search-R1'
export BASE_MODEL='/root/autodl-tmp/Qwen2.5-3B-Instruct'
export EXPERIMENT_NAME=nqhotpotqa-search-r1-qwen2.5-3b-infer-rollouts

export VLLM_ATTENTION_BACKEND=XFORMERS
# Use data parallel across 2 GPUs for Qwen2.5-3B so each GPU keeps a full
# model replica. Leave some headroom for the retrieval encoder.
export ROLLOUT_TP_SIZE=1
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.80
export INFERENCE_BATCH_SIZE=24
export RESUME_INFERENCE=${RESUME_INFERENCE:-false}
export ROLLOUTS_DIR=${ROLLOUTS_DIR:-null}

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.inference_batch_size=$INFERENCE_BATCH_SIZE \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=False \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=false \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=[] \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.inference_only=true \
    +trainer.resume_inference=$RESUME_INFERENCE \
    'trainer.inference_splits=[train,test]' \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.rollouts_dir=$ROLLOUTS_DIR \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log
