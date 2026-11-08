#!/bin/bash
# Run on n04. Serve each companion checkpoint in turn on one GPU and evaluate it against every player
# in PLAYERS_JSON (test split). Cells already on disk are skipped by eval_matrix.py.
# usage: matrix_run.sh GPU PORT OUTDIR PLAYERS_JSON N "name=path name=path ..." [MEMORY]
set -uo pipefail
GPU=$1; PORT=$2; OUT=$3; PLAYERS=$4; N=$5; COMPANIONS=$6; MEMORY=${7:-full}; PTYPES=${PTYPES:-}
cd /data2/xiangding/companion
mkdir -p $OUT
for pair in $COMPANIONS; do
  name=${pair%%=*}; path=${pair#*=}
  cfg=$OUT/cfg_$name.json
  python3 - "$name" "$PORT" "$PLAYERS" "$N" "$MEMORY" "$PTYPES" > $cfg <<'EOF'
import json, sys
name, port, players, n, memory, ptypes = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6]
print(json.dumps({"split": "test", "n": n, "memory": memory, "ptypes": ptypes.split(",") if ptypes else None,
                  "companions": [{"name": name, "url": f"http://127.0.0.1:{port}/v1", "model": "ckpt"}],
                  "players": json.load(open(players))}))
EOF
  CUDA_VISIBLE_DEVICES=$GPU setsid bash /data1/xiangding/cvis-ablate/scripts/h20_env_ablate.sh vllm serve $path \
    --served-model-name ckpt --port $PORT --max-model-len 16384 --gpu-memory-utilization 0.85 --enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0,"video":0}' > $OUT/serve_$name.log 2>&1 < /dev/null &
  SPID=$!
  for i in $(seq 1 120); do curl -sf -o /dev/null localhost:$PORT/v1/models && break; sleep 5; done
  echo "[$(date +%H:%M)] companion $name"
  /data1/xiangding/cvis-ablate/env/bin/python scripts/eval_matrix.py $cfg --out $OUT --concurrency 96
  kill -- -$SPID 2>/dev/null || kill $SPID
  sleep 15
done
