#!/bin/bash
# Serve one checkpoint on one GPU, evaluate it, then stop the server it started (by its own PID).
# usage: eval_ckpt.sh GPU PORT CKPT_DIR PLAYER_URLS(space-sep, quoted) PLAYER_MODEL SPLIT N OUT_JSONL [PLAYER_MODE] [MEMORY]
#   PLAYER_MODE: plain (default) | voice:<voice_url>:<voice_model>
set -uo pipefail
GPU=$1; PORT=$2; CKPT=$3; PURLS=$4; PMODEL=$5; SPLIT=$6; N=$7; OUT=$8; MODE=${9:-plain}; MEMORY=${10:-full}
LOG=$(dirname $OUT)/serve_$(basename $OUT .jsonl).log
mkdir -p $(dirname $OUT)
CUDA_VISIBLE_DEVICES=$GPU setsid bash /data1/xiangding/cvis-ablate/scripts/h20_env_ablate.sh vllm serve $CKPT \
  --served-model-name ckpt --port $PORT --max-model-len 16384 --gpu-memory-utilization 0.85 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' > $LOG 2>&1 < /dev/null &
SPID=$!
for i in $(seq 1 120); do curl -sf -o /dev/null localhost:$PORT/v1/models && break; sleep 5; done
cd /data2/xiangding/companion
/data1/xiangding/cvis-ablate/env/bin/python scripts/eval.py --companion-url http://127.0.0.1:$PORT/v1 --companion-model ckpt \
  --player-urls $PURLS --player-model $PMODEL --split $SPLIT --n $N --out $OUT --player-mode $MODE --memory $MEMORY
RC=$?
kill -- -$SPID 2>/dev/null || kill $SPID
exit $RC
