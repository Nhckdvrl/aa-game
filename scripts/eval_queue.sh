#!/bin/bash
# Run on n01: serialized jobs on the single n04 eval GPU, logging every step to logs/eval_queue.log.
L=/data1/xiangding/spellbrush/companion/logs/eval_queue.log
log() { echo "[$(date +%H:%M)] $*" >> $L; }
q() { for i in 1 2 3 4 5; do timeout ${2:-3600} ssh -o ConnectTimeout=10 aic-n04 "$1" && return 0; sleep 15; done; return 1; }

log "start CharacterJudge scoring"
q 'cd /data2/xiangding/companion && CUDA_VISIBLE_DEVICES=3 setsid bash /data1/xiangding/cvis-ablate/scripts/h20_env_ablate.sh vllm serve /data2/xiangding/models/CharacterJudge --served-model-name cj --port 8302 --max-model-len 8192 --gpu-memory-utilization 0.85 > runs/serve_cj.log 2>&1 < /dev/null & SPID=$!; for i in $(seq 1 120); do curl -sf -o /dev/null localhost:8302/v1/models && break; sleep 5; done; /data1/xiangding/cvis-ablate/env/bin/python scripts/charjudge_runs.py runs/matrix/base4b__q35.jsonl runs/matrix/A50__q35.jsonl runs/matrix/B75__q35.jsonl runs/matrix/C75__q35.jsonl runs/matrix/D75__q35.jsonl --url http://127.0.0.1:8302/v1 --model cj 2>&1 | grep -v Warning > runs/charjudge_matrix.txt; kill -- -$SPID 2>/dev/null; cat runs/charjudge_matrix.txt' 3600 >> $L 2>&1
log "CharacterJudge done; val sweep C/D late checkpoints + E/F"
bash /data1/xiangding/spellbrush/companion/scripts/val_sweep.sh 3 8300 96 "C:aic-n05 D:aic-n05 E:aic-n04 F:aic-n06" >> $L 2>&1
log "queue finished"
