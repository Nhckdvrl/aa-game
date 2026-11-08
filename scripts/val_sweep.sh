#!/bin/bash
# Run on n01. For each arm checkpoint not yet evaluated: stream it to n04, evaluate on the val split
# against the training simulator pool on one n04 GPU, and record the result. Re-runnable (skips done).
# usage: val_sweep.sh GPU PORT N "A:aic-n04 C2:aic-n05:progressive F2:aic-n01:progressive:9b ..."
# (memory defaults to full, size to 4b; non-4b runs are recorded as arm_<ARM>_<size>_step<N>)
set -uo pipefail
GPU=$1; PORT=$2; N=$3; ARMS=$4
POOL="http://172.31.3.227:8101/v1 http://172.31.3.227:8102/v1 http://172.31.3.227:8103/v1 http://172.31.3.227:8104/v1 http://172.31.3.224:8101/v1 http://172.31.3.224:8102/v1"
EVAL=aic-n04
for spec in $ARMS; do
  arm=$(echo $spec | cut -d: -f1); node=$(echo $spec | cut -d: -f2); mem=$(echo $spec | cut -d: -f3); mem=${mem:-full}
  size=$(echo $spec | cut -d: -f4); size=${size:-4b}; tag=$arm; [ $size != 4b ] && tag=${arm}_$size
  ckpts=$(ssh $node "ls -d /data2/xiangding/companion/train/arm_${arm}_${size}/ckpt/checkpoint-* 2>/dev/null | sort -t- -k2 -n")
  for c in $ckpts; do
    step=${c##*-}
    out=/data2/xiangding/companion/runs/val/arm_${tag}_step${step}.jsonl
    ssh $EVAL "test -s $out" && continue
    dst=/data2/xiangding/companion/ckpts/arm_${tag}/checkpoint-$step
    if [ "$node" != "$EVAL" ]; then
      ssh $EVAL "mkdir -p $(dirname $dst)"
      ssh $node "tar -C $(dirname $c) -cf - checkpoint-$step" | ssh $EVAL "tar -C $(dirname $dst) -xf -"
    else
      dst=$c
    fi
    echo "[$(date +%H:%M)] eval arm $tag step $step"
    ssh $EVAL "bash /data2/xiangding/companion/scripts/eval_ckpt.sh $GPU $PORT $dst '$POOL' q35 val $N $out plain $mem" | tail -2
  done
done
