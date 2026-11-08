#!/bin/bash
# Run on n01 (from logs/snap). Stop run NAME on NODE (exact torchrun PID by output dir), archive its dir under
# train/failed/NAME_TAG, and relaunch it with the same settings once its GPUs are free. NODE=local for n01.
# usage: restart_arm.sh NODE NAME GPUS ARM PORT MEMORY POLICY TAG [ENV_ASSIGNMENTS]
set -uo pipefail
NODE=$1; NAME=$2; GPUS=$3; ARM=$4; PORT=$5; MEMORY=$6; POLICY=$7; TAG=$8; ENVS=${9:-}
POOL="http://172.31.3.227:8101/v1,http://172.31.3.227:8102/v1,http://172.31.3.227:8103/v1,http://172.31.3.227:8104/v1,http://172.31.3.224:8101/v1,http://172.31.3.224:8102/v1"
R=/data2/xiangding/companion/train
q() { if [ $NODE = local ]; then bash -c "$1"; return; fi
      for i in 1 2 3 4 5; do out=$(timeout 30 ssh -o ConnectTimeout=10 $NODE "$1" 2>/dev/null) && { echo "$out"; return 0; }; sleep 10; done; return 1; }
q "for p in \$(pgrep -f '^python -m torch.distributed.run'); do tr '\\0' ' ' < /proc/\$p/cmdline | grep -q '$R/$NAME/ckpt ' && kill \$p && echo stopped \$p; done"
idx=$(echo $GPUS | tr ',' '\n' | awk '{print $1+1}' | paste -sd, -)
until q "[ \$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '$(echo $idx | sed 's/,/p;/g')p' | awk '{s+=\$1} END {print s}') -lt 2000 ]" >/dev/null; do sleep 15; done
echo "[$(date +%H:%M)] GPUs $GPUS free on $NODE"
q "mkdir -p $R/failed && mv $R/$NAME $R/failed/${NAME}_$TAG && mv $R/$NAME.log $R/failed/${NAME}_$TAG.log; echo archived"
L="cd /data2/xiangding/companion && (setsid nohup env $ENVS bash scripts/launch_grpo.sh $ARM $GPUS $POOL q35 $POLICY $NAME $PORT $MEMORY > train/$NAME.log 2>&1 < /dev/null &); echo launched $NAME"
if [ $NODE = local ]; then bash -c "$L"; else timeout 30 ssh $NODE "$L"; fi
