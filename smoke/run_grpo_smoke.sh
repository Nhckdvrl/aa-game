#!/bin/bash
# GRPO pipeline smoke test: 4B companion policy on GPUs 4-7 (colocate vLLM), simulated player on GPU0 server.
set -e
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PLAYER_URL=http://127.0.0.1:8100/v1 PLAYER_MODEL=q9 MAX_TURNS=6
cd /data2/xiangding/companion/smoke
bash /data1/xiangding/cvis-ablate/scripts/h20_env_ablate.sh python -m torch.distributed.run --nproc_per_node 4 --master_port 29911 \
  /data1/xiangding/cvis-ablate/env/lib/python3.10/site-packages/swift/cli/rlhf.py \
  --rlhf_type grpo --model /data2/xiangding/models/Qwen3.5-4B --tuner_type full \
  --dataset /data2/xiangding/companion/smoke/train.jsonl --external_plugins /data2/xiangding/companion/proto/swift_plugin.py \
  --multi_turn_scheduler gym_scheduler --gym_env companion_env --use_gym_env true --max_turns 6 \
  --remove_unused_columns false --split_dataset_ratio 0 --enable_thinking false \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 1 --num_generations 4 --generation_batch_size 32 \
  --use_vllm true --vllm_mode colocate --vllm_tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.4 --vllm_max_model_len 8192 \
  --gradient_checkpointing true --torch_dtype bfloat16 --learning_rate 1e-6 --beta 0.01 --temperature 1.0 \
  --max_length 8192 --max_completion_length 1024 --output_dir /data2/xiangding/companion/smoke/out --add_version false \
  --max_steps 3 --save_steps 1000 --logging_steps 1 --seed 42 --deepspeed zero3 --sleep_level 1 --log_completions true
