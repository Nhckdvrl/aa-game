#!/bin/bash
# Serve one checkpoint and run the test split once per persona card (HDS_PERSONA), against the 35B player pool.
# usage: persona_eval.sh GPU PORT NAME CKPT [N] [CARDS]
GPU=$1; PORT=$2; NAME=$3; CKPT=$4; N=${5:-120}; CARDS=${6:-"tsundere gentle kuudere chuuni lazy"}
cd /data2/xiangding/companion; OUT=runs/persona_eval; mkdir -p $OUT
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
POOL="http://172.31.3.227:8101/v1 http://172.31.3.227:8102/v1 http://172.31.3.227:8103/v1 http://172.31.3.227:8104/v1 http://172.31.3.224:8101/v1 http://172.31.3.224:8102/v1"
CUDA_VISIBLE_DEVICES=$GPU setsid bash /data1/xiangding/cvis-ablate/scripts/h20_env_ablate.sh vllm serve $CKPT \
  --served-model-name ckpt --port $PORT --max-model-len 16384 --gpu-memory-utilization 0.85 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' > $OUT/serve_$NAME.log 2>&1 < /dev/null &
SPID=$!
for i in $(seq 1 120); do curl -sf -o /dev/null localhost:$PORT/v1/models && break; sleep 5; done
for p in $CARDS; do
  HDS_PERSONA=$p /data1/xiangding/cvis-ablate/env/bin/python scripts/eval.py --companion-url http://127.0.0.1:$PORT/v1 \
    --companion-model ckpt --player-urls $POOL --player-model q35 --split test --n $N --memory progressive \
    --out $OUT/${NAME}_$p.jsonl | tail -1
done
kill -- -$SPID 2>/dev/null || kill $SPID
for i in $(seq 1 60); do ss -ltn "sport = :$PORT" | grep -q LISTEN || break; sleep 2; done
echo "[$(date +%H:%M)] persona_eval $NAME done"
