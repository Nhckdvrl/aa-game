#!/bin/bash
# usage: launch_grpo.sh ARM GPUS(e.g. 4,5,6,7) PLAYER_URLS(comma) PLAYER_MODEL POLICY_PATH RUN_NAME [MASTER_PORT] [MEMORY full|progressive]
set -euo pipefail
ARM=$1; GPUS=$2; URLS=$3; PMODEL=$4; POLICY=$5; NAME=$6; PORT=${7:-29911}; MEMORY=${8:-full}
ROOT=/data2/xiangding/companion
OUT=$ROOT/train/$NAME
mkdir -p $OUT
NPROC=$(echo $GPUS | tr ',' '\n' | wc -l)
PDBS=${PDBS:-4}; GA=${GA:-8}; EXTRA=${EXTRA:-}   # 9B: PDBS=2 GA=16 keeps 128 samples per step
# warm-started v2 runs use GBS=96 (3 GPUs x 4, or 4 GPUs x PDBS=3) and optionally SEED
[ -n "${SCALE_REWARDS:-}" ] && EXTRA="$EXTRA --scale_rewards $SCALE_REWARDS"   # e.g. none = Dr. GRPO
# flash-linear-attention kernels for Qwen3.5's gated delta rule (2x faster fwd+bwd than the torch fallback)
[ -d /data1/xiangding/fla_pkgs ] && export PYTHONPATH=/data1/xiangding/fla_pkgs${PYTHONPATH:+:$PYTHONPATH}
DATA=$ROOT/train/train.jsonl; MAXLEN=12288; MAXTURNS=16
# v2 episodes need more turns (one memory fragment per turn, only when the player brings it up)
[ "$MEMORY" = progressive ] && { DATA=$ROOT/train/train_v2.jsonl; MAXLEN=16384; MAXTURNS=24; }
[ -f $DATA ] || /data1/xiangding/cvis-ablate/env/bin/python $ROOT/scripts/make_train_data.py 20000 $DATA $MEMORY
export CUDA_VISIBLE_DEVICES=$GPUS HDS_ARM=$ARM HDS_PLAYER_URLS=$URLS HDS_PLAYER_MODEL=$PMODEL HDS_EPISODE_LOG=$OUT/episodes.jsonl
cd $OUT
exec bash /data1/xiangding/cvis-ablate/scripts/h20_env_ablate.sh python -m torch.distributed.run --nproc_per_node $NPROC --master_port $PORT \
  /data1/xiangding/cvis-ablate/env/lib/python3.10/site-packages/swift/cli/rlhf.py \
  --rlhf_type grpo --model $POLICY --tuner_type full \
  --dataset $DATA --external_plugins $ROOT/hds/swift_env.py \
  --multi_turn_scheduler gym_scheduler --gym_env hds_companion --use_gym_env true --max_turns $MAXTURNS \
  --remove_unused_columns false --split_dataset_ratio 0 --enable_thinking false --dataset_shuffle true --seed ${SEED:-42} \
  --per_device_train_batch_size $PDBS --gradient_accumulation_steps $GA --num_generations 8 --generation_batch_size ${GBS:-128} \
  --use_vllm true --vllm_mode colocate --vllm_tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.35 --vllm_max_model_len $MAXLEN \
  --gradient_checkpointing true --torch_dtype bfloat16 --learning_rate 1e-6 --beta 0.01 --temperature 1.0 --warmup_ratio 0.02 \
  --max_length $MAXLEN --max_completion_length 320 --completion_length_limit_scope per_round --output_dir $OUT/ckpt --add_version false \
  --max_steps 400 --save_steps 25 --save_total_limit 20 --save_only_model true --logging_steps 1 \
  --deepspeed zero3 --sleep_level 1 --report_to none $EXTRA
