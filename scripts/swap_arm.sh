#!/bin/bash
# Run on n01. When OLD arm on NODE has saved checkpoint STEP (and moved past it), stop exactly that run
# and launch NEW arm on the same GPUs. Launch happens once (no retry wrapper).
# usage: swap_arm.sh NODE OLD_RUN STEP GPUS NEW_ARM NEW_RUN MEMORY MASTER_PORT [POLICY_PATH] [ENV_ASSIGNMENTS]
set -uo pipefail
NODE=$1; OLD=$2; STEP=$3; GPUS=$4; ARM=$5; NEW=$6; MEMORY=$7; PORT=$8
POLICY=${9:-/data2/xiangding/models/Qwen3.5-4B}; ENVS=${10:-}
POOL="http://172.31.3.227:8101/v1,http://172.31.3.227:8102/v1,http://172.31.3.227:8103/v1,http://172.31.3.227:8104/v1,http://172.31.3.224:8101/v1,http://172.31.3.224:8102/v1"
R=/data2/xiangding/companion/train
q() { for i in 1 2 3 4 5; do out=$(timeout 30 ssh -o ConnectTimeout=10 $NODE "$1" 2>/dev/null) && { echo "$out"; return 0; }; sleep 10; done; return 1; }
until q "test -f $R/$OLD/ckpt/checkpoint-$STEP/model.safetensors.index.json && python3 -c \"import json,sys; s=[json.loads(l).get('global_step/max_steps','0/0') for l in open('$R/$OLD/ckpt/logging.jsonl')]; sys.exit(0 if int(s[-1].split('/')[0]) > $STEP else 1)\"" >/dev/null; do sleep 60; done
echo "[$(date +%H:%M)] $OLD reached checkpoint-$STEP"
q "for p in \$(pgrep -f '^python -m torch.distributed.run'); do tr '\\0' ' ' < /proc/\$p/cmdline | grep -q '$R/$OLD/ckpt ' && kill \$p && echo stopped \$p; done"
idx=$(echo $GPUS | tr ',' '\n' | awk '{print $1+1}' | paste -sd, -)
until q "[ \$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '$(echo $idx | sed 's/,/p;/g')p' | awk '{s+=\$1} END {print s}') -lt 2000 ]" >/dev/null; do sleep 15; done
echo "[$(date +%H:%M)] GPUs $GPUS free on $NODE"
timeout 30 ssh $NODE "cd /data2/xiangding/companion && (setsid nohup env $ENVS bash scripts/launch_grpo.sh $ARM $GPUS $POOL q35 $POLICY $NEW $PORT $MEMORY > train/$NEW.log 2>&1 < /dev/null &); echo launched $NEW"
