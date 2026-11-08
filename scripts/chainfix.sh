#!/bin/bash
# Run on a node with a free GPU. Serve one checkpoint, then evaluate it on the two-world cipher+cipher slice of
# the test split under each room variant (HDS_CHAIN_MODE orig / reveal / warn). usage: chainfix.sh GPU PORT NAME CKPT [N]
GPU=$1; PORT=$2; NAME=$3; CKPT=$4; N=${5:-120}
cd /data2/xiangding/companion; OUT=runs/chainfix; mkdir -p $OUT
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
POOL="http://172.31.3.227:8101/v1 http://172.31.3.227:8102/v1 http://172.31.3.227:8103/v1 http://172.31.3.227:8104/v1 http://172.31.3.224:8101/v1 http://172.31.3.224:8102/v1"
CUDA_VISIBLE_DEVICES=$GPU setsid bash /data1/xiangding/cvis-ablate/scripts/h20_env_ablate.sh vllm serve $CKPT \
  --served-model-name ckpt --port $PORT --max-model-len 16384 --gpu-memory-utilization 0.85 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' > $OUT/serve_$NAME.log 2>&1 < /dev/null &
SPID=$!
for i in $(seq 1 120); do curl -sf -o /dev/null localhost:$PORT/v1/models && break; sleep 5; done
for mode in orig reveal warn; do
  HDS_CHAIN_MODE=$mode /data1/xiangding/cvis-ablate/env/bin/python scripts/eval_subset.py --companion-url http://127.0.0.1:$PORT/v1 \
    --companion-model ckpt --player-urls $POOL --player-model q35 --first cipher_lock --n $N --memory progressive \
    --out $OUT/${NAME}_$mode.jsonl
done
kill -- -$SPID 2>/dev/null || kill $SPID
for i in $(seq 1 60); do ss -ltn "sport = :$PORT" | grep -q LISTEN || break; sleep 2; done
echo "[$(date +%H:%M)] chainfix $NAME done"
