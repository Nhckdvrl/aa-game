#!/bin/bash
# Run on a node with a free GPU: serve one checkpoint (vLLM) and measure reply latency at 1..128 simultaneous players.
# usage: throughput.sh GPU PORT NAME CKPT CONTEXT_JSONL OUT_JSONL
GPU=$1; PORT=$2; NAME=$3; CKPT=$4; CTX=$5; OUT=$6
cd /data2/xiangding/companion; unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
CUDA_VISIBLE_DEVICES=$GPU setsid bash /data1/xiangding/cvis-ablate/scripts/h20_env_ablate.sh vllm serve $CKPT --served-model-name ckpt \
  --port $PORT --max-model-len 16384 --gpu-memory-utilization 0.85 --enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0}' \
  > runs/serve_tp_$NAME.log 2>&1 < /dev/null &
SPID=$!
for i in $(seq 1 120); do curl -sf -o /dev/null localhost:$PORT/v1/models && break; sleep 5; done
for c in 1 8 32 64 128; do
  /data1/xiangding/cvis-ablate/env/bin/python scripts/latency_bench.py --url http://127.0.0.1:$PORT/v1 --model ckpt --runs $CTX --n 20 --concurrency $c --name $NAME >> $OUT
done
kill -- -$SPID 2>/dev/null || kill $SPID
echo "[$(date +%H:%M)] throughput $NAME done"
