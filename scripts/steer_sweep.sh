#!/bin/bash
# Run on n04. One steering config per GPU: start steer_server.py, run the val split (v2 memory) against the
# training player pool, stop the server. usage: steer_sweep.sh GPU PORT TAG "server args" [N] [MEMORY]
GPU=$1; PORT=$2; TAG=$3; SARGS=$4; N=${5:-96}; MEMORY=${6:-progressive}   # env HDS_PERSONA passes through to eval.py
cd /data2/xiangding/companion; mkdir -p runs/steer
POOL="http://172.31.3.227:8101/v1 http://172.31.3.227:8102/v1 http://172.31.3.227:8103/v1 http://172.31.3.227:8104/v1 http://172.31.3.224:8101/v1 http://172.31.3.224:8102/v1"
W=/data1/xiangding/cvis-ablate/scripts/h20_env_ablate.sh
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/data1/xiangding/fla_pkgs setsid bash $W python scripts/steer_server.py --port $PORT $SARGS \
  > runs/steer/serve_$TAG.log 2>&1 < /dev/null &
SPID=$!
for i in $(seq 1 120); do curl -sf -o /dev/null localhost:$PORT/v1/models && break; sleep 5; done
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
/data1/xiangding/cvis-ablate/env/bin/python scripts/eval.py --companion-url http://127.0.0.1:$PORT/v1 --companion-model ckpt \
  --player-urls $POOL --player-model q35 --split val --n $N --out runs/steer/$TAG.jsonl --memory $MEMORY --concurrency 48 | tail -1
kill -- -$SPID 2>/dev/null || kill $SPID
for i in $(seq 1 60); do ss -ltn "sport = :$PORT" | grep -q LISTEN || break; sleep 2; done
echo "[$(date +%H:%M)] done $TAG"
