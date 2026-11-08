#!/bin/bash
# Run on n04 once its training GPUs 4-7 are free: the two large held-out players, TP=2 each.
# Qwen3.5-122B-A10B-FP8 -> :8401 (q122), GLM-4.5-Air-FP8 -> :8402 (glm45)
M=/data2/xiangding/models; L=/data2/xiangding/companion/runs/matrix_final; mkdir -p $L
W=/data1/xiangding/cvis-ablate/scripts/h20_env_ablate.sh
CUDA_VISIBLE_DEVICES=4,5 setsid bash $W vllm serve $M/Qwen3.5-122B-A10B-FP8 --served-model-name q122 --port 8401 \
  --tensor-parallel-size 2 --max-model-len 16384 --gpu-memory-utilization 0.90 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' > $L/serve_q122.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=6,7 setsid bash $W vllm serve $M/GLM-4.5-Air-FP8 --served-model-name glm45 --port 8402 \
  --tensor-parallel-size 2 --max-model-len 16384 --gpu-memory-utilization 0.90 --enable-prefix-caching \
  > $L/serve_glm45.log 2>&1 < /dev/null &
echo started
