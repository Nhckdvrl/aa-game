#!/bin/bash
# Run on n01 (from logs/snap). Stop run OLD on NODE now (exact torchrun PID by output dir; no checkpoint wait)
# and launch NEW on the same GPUs once they are free.
# usage: stop_launch.sh NODE OLD GPUS ARM NEW MEMORY MASTER_PORT POLICY [ENV_ASSIGNMENTS]
set -uo pipefail
NODE=$1; OLD=$2; GPUS=$3; ARM=$4; NEW=$5; MEMORY=$6; PORT=$7; POLICY=$8; ENVS=${9:-}
POOL="http://172.31.3.227:8101/v1,http://172.31.3.227:8102/v1,http://172.31.3.227:8103/v1,http://172.31.3.227:8104/v1,http://172.31.3.224:8101/v1,http://172.31.3.224:8102/v1"
R=/data2/xiangding/companion/train
q() { for i in 1 2 3 4 5; do out=$(timeout 30 ssh -o ConnectTimeout=10 $NODE "$1" 2>/dev/null) && { echo "$out"; return 0; }; sleep 10; done; return 1; }
q "for p in \$(pgrep -f '^python -m torch.distributed.run'); do tr '\\0' ' ' < /proc/\$p/cmdline | grep -q '$R/$OLD/ckpt ' && kill \$p && echo stopped \$p; done"
idx=$(echo $GPUS | tr ',' '\n' | awk '{print $1+1}' | paste -sd, -)
until q "[ \$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '$(echo $idx | sed 's/,/p;/g')p' | awk '{s+=\$1} END {print s}') -lt 2000 ]" >/dev/null; do sleep 15; done
echo "[$(date +%H:%M)] GPUs $GPUS free on $NODE"
timeout 30 ssh $NODE "cd /data2/xiangding/companion && (setsid nohup env $ENVS bash scripts/launch_grpo.sh $ARM $GPUS $POOL q35 $POLICY $NEW $PORT $MEMORY > train/$NEW.log 2>&1 < /dev/null &); echo launched $NEW"
