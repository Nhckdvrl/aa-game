#!/bin/bash
# usage: sft_voice.sh GPUS TRAIN_JSONL VAL_JSONL OUT
set -euo pipefail
GPUS=$1; TRAIN=$2; VAL=$3; OUT=$4
NPROC=$(echo $GPUS | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES=$GPUS
mkdir -p $OUT
exec bash /data1/xiangding/cvis-ablate/scripts/h20_env_ablate.sh python -m torch.distributed.run --nproc_per_node $NPROC --master_port 29933 \
  /data1/xiangding/cvis-ablate/env/lib/python3.10/site-packages/swift/cli/sft.py \
  --model /data2/xiangding/models/Qwen3.5-4B --tuner_type full --dataset $TRAIN --val_dataset $VAL \
  --enable_thinking false --num_train_epochs 1 --per_device_train_batch_size 8 --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 --warmup_ratio 0.03 --lr_scheduler_type cosine --max_length 1024 --torch_dtype bfloat16 \
  --gradient_checkpointing true --deepspeed zero2 --eval_steps 100 --save_steps 100 --save_total_limit 2 \
  --save_only_model true --logging_steps 10 --output_dir $OUT --add_version false --seed 42 --report_to none \
  --loss_scale default --split_dataset_ratio 0
