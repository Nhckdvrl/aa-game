#!/bin/bash
# Run on n01. For each arm checkpoint not yet evaluated: stream it to n04, evaluate on the val split
# against the training simulator pool on one n04 GPU, and record the result. Re-runnable (skips done).
# usage: val_sweep.sh GPU PORT N "A:aic-n04 C2:aic-n05:progressive ..."   (memory defaults to full)
set -uo pipefail
GPU=$1; PORT=$2; N=$3; ARMS=$4
POOL="http://172.31.3.227:8101/v1 http://172.31.3.227:8102/v1 http://172.31.3.227:8103/v1 http://172.31.3.227:8104/v1 http://172.31.3.224:8101/v1 http://172.31.3.224:8102/v1"
EVAL=aic-n04
for spec in $ARMS; do
  arm=$(echo $spec | cut -d: -f1); node=$(echo $spec | cut -d: -f2); mem=$(echo $spec | cut -d: -f3); mem=${mem:-full}
  ckpts=$(ssh $node "ls -d /data2/xiangding/companion/train/arm_${arm}_4b/ckpt/checkpoint-* 2>/dev/null | sort -t- -k2 -n")
  for c in $ckpts; do
    step=${c##*-}
    out=/data2/xiangding/companion/runs/val/arm_${arm}_step${step}.jsonl
    ssh $EVAL "test -s $out" && continue
    dst=/data2/xiangding/companion/ckpts/arm_${arm}/checkpoint-$step
    if [ "$node" != "$EVAL" ]; then
      ssh $EVAL "mkdir -p $(dirname $dst)"
      ssh $node "tar -C $(dirname $c) -cf - checkpoint-$step" | ssh $EVAL "tar -C $(dirname $dst) -xf -"
    else
      dst=$c
    fi
    echo "[$(date +%H:%M)] eval arm $arm step $step"
    ssh $EVAL "bash /data2/xiangding/companion/scripts/eval_ckpt.sh $GPU $PORT $dst '$POOL' q35 val $N $out plain $mem" | tail -2
  done
done
