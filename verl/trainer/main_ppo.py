# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
import glob
import os
import time

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


def _normalize_sample_identifier(value):
    if hasattr(value, 'item'):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    value = str(value).strip()
    if value == '' or value.lower() in {'none', 'nan'}:
        return None
    return value


def _make_sample_key(uid=None, index=None):
    normalized_uid = _normalize_sample_identifier(uid)
    if normalized_uid is not None:
        return f'uid:{normalized_uid}'
    normalized_index = _normalize_sample_identifier(index)
    if normalized_index is not None:
        return f'index:{normalized_index}'
    return None


def _resolve_rollouts_dir(config, inference_only: bool):
    configured_dir = config.trainer.get('rollouts_dir', None)
    if configured_dir not in (None, '', 'null'):
        return configured_dir

    resume_inference = inference_only and bool(config.trainer.get('resume_inference', False))
    if resume_inference:
        pattern = f"/root/autodl-tmp/rollouts/{config.trainer.experiment_name}_*"
        candidates = [path for path in glob.glob(pattern) if os.path.isdir(path)]
        if candidates:
            latest_dir = max(candidates, key=os.path.getmtime)
            print(f"Resuming inference from existing rollouts dir: {latest_dir}")
            return latest_dir

    time_str = time.strftime("%Y%m%d_%H%M%S")
    return f'/root/autodl-tmp/rollouts/{config.trainer.experiment_name}_{time_str}'


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0., log_file=None, split_name=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.log_file = log_file
        self.split_name = split_name

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        records_to_log = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)

            reward_tensor[i, valid_response_length - 1] = score
            
            if self.log_file:
                # Convert numpy arrays to native Python types for JSON serialization
                data_source_val = data_source.item() if hasattr(data_source, 'item') else str(data_source)
                ground_truth_val = ground_truth.item() if hasattr(ground_truth, 'item') else str(ground_truth)
                index = data_item.non_tensor_batch['index'] if 'index' in data_item.non_tensor_batch else None
                uid = data_item.non_tensor_batch['uid'] if 'uid' in data_item.non_tensor_batch else None
                index_val = index.item() if hasattr(index, 'item') else index
                uid_val = uid.item() if hasattr(uid, 'item') else uid
                sample_key = _make_sample_key(uid=uid_val, index=index_val)
                records_to_log.append({
                    "split": self.split_name,
                    "index": index_val,
                    "uid": uid_val,
                    "sample_key": sample_key,
                    "data_source": data_source_val,
                    "ground_truth": ground_truth_val,
                    "prompt": prompt_str,
                    "response": response_str,
                    "full_sequence": sequences_str,
                    "score": float(score)
                })

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
        
        if self.log_file and len(records_to_log) > 0:
            import json
            import os
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "a") as f:
                for record in records_to_log:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf, open_dict

    # print initial config
    from pprint import pprint
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    inference_only = bool(config.trainer.get('inference_only', False))
    rollout_role = Role.Rollout if inference_only else Role.ActorRollout

    role_worker_mapping = {
        rollout_role: ray.remote(ActorRolloutRefWorker),
    }
    mapping = {
        rollout_role: global_pool_id,
    }

    if not inference_only:
        role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.Critic] = global_pool_id
        mapping[Role.RefPolicy] = global_pool_id

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable and not inference_only:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    rollouts_dir = _resolve_rollouts_dir(config, inference_only=inference_only)
    os.makedirs(rollouts_dir, exist_ok=True)
    with open_dict(config):
        config.trainer.rollouts_dir = rollouts_dir

    train_rollout_log = os.path.join(rollouts_dir, 'train_rollouts.jsonl' if inference_only else 'epoch_0_rollouts.jsonl')
    val_rollout_log = os.path.join(rollouts_dir, 'test_rollouts.jsonl' if inference_only else 'val_rollouts.jsonl')

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, 
                              log_file=train_rollout_log,
                              split_name='train')

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1,
                                  log_file=val_rollout_log,
                                  split_name='test' if inference_only else 'val')

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
