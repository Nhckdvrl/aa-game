#!/bin/bash
# Run on n04: serve each model on one GPU and run the single-user latency bench on the same contexts.
# usage: latency_all.sh GPU PORT CONTEXT_JSONL "name=path ..." OUT_JSONL
GPU=$1; PORT=$2; CTX=$3; MODELS=$4; OUT=$5
cd /data2/xiangding/companion
for pair in $MODELS; do
  name=${pair%%=*}; path=${pair#*=}
  CUDA_VISIBLE_DEVICES=$GPU setsid bash /data1/xiangding/cvis-ablate/scripts/h20_env_ablate.sh vllm serve $path \
    --served-model-name ckpt --port $PORT --max-model-len 16384 --gpu-memory-utilization 0.85 --enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0,"video":0}' > runs/serve_lat_$name.log 2>&1 < /dev/null &
  SPID=$!
  for i in $(seq 1 120); do curl -sf -o /dev/null localhost:$PORT/v1/models && break; sleep 5; done
  /data1/xiangding/cvis-ablate/env/bin/python scripts/latency_bench.py --url http://127.0.0.1:$PORT/v1 --model ckpt --runs $CTX --n 60 --name $name >> $OUT
  kill -- -$SPID 2>/dev/null || kill $SPID; sleep 15
done
