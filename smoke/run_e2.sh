#!/bin/bash
# E2: same 64 episodes, companion = Qwen3.5-9B (GPU0 :8100), player simulator = Qwen3.5-35B-A3B-FP8 (GPU1 :8101).
set -e
if ! curl -s localhost:8101/v1/models >/dev/null; then
  setsid nohup bash /data2/xiangding/serve.sh 1 8101 /data2/xiangding/models/Qwen3.5-35B-A3B-FP8 q35 > /data2/xiangding/logs/serve_q35.log 2>&1 < /dev/null &
  until curl -s localhost:8101/v1/models >/dev/null; do sleep 5; done
fi
cd /data2/xiangding/companion/proto
/data1/xiangding/cvis-ablate/env/bin/python run_episodes.py --base-url http://127.0.0.1:8100/v1 --player-base-url http://127.0.0.1:8101/v1 \
  --cmodel q9 --pmodel q35 --n 8 --out /data2/xiangding/companion/e2_q9_q35.jsonl
