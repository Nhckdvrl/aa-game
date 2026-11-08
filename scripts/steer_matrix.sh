#!/bin/bash
# Run on n04. Serve one steering config (steer_server.py) and evaluate it on the test split against every player in
# scripts/players_n04.json (q35, glm9b, voice, humanstats). usage: steer_matrix.sh GPU PORT NAME "server args" [N] [MEMORY]
GPU=$1; PORT=$2; NAME=$3; SARGS=$4; N=${5:-240}; MEMORY=${6:-progressive}
cd /data2/xiangding/companion; OUT=runs/steer_matrix; mkdir -p $OUT
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
W=/data1/xiangding/cvis-ablate/scripts/h20_env_ablate.sh
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/data1/xiangding/fla_pkgs setsid bash $W python scripts/steer_server.py --port $PORT $SARGS \
  > $OUT/serve_$NAME.log 2>&1 < /dev/null &
SPID=$!
for i in $(seq 1 120); do curl -sf -o /dev/null localhost:$PORT/v1/models && break; sleep 5; done
python3 - "$NAME" "$PORT" "$N" "$MEMORY" > $OUT/cfg_$NAME.json <<'PY'
import json, sys
name, port, n, memory = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
print(json.dumps({"split": "test", "n": n, "memory": memory, "ptypes": None,
                  "companions": [{"name": name, "url": f"http://127.0.0.1:{port}/v1", "model": "ckpt"}],
                  "players": json.load(open(__import__("os").environ.get("PLAYERS", "scripts/players_n04.json")))}))
PY
/data1/xiangding/cvis-ablate/env/bin/python scripts/eval_matrix.py $OUT/cfg_$NAME.json --out $OUT --concurrency 48
kill -- -$SPID 2>/dev/null || kill $SPID
for i in $(seq 1 60); do ss -ltn "sport = :$PORT" | grep -q LISTEN || break; sleep 2; done
echo "[$(date +%H:%M)] done $NAME"
