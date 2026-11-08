#!/bin/bash
# Refit monitor for a shaping run (run on n01). Each time the run saves a checkpoint: copy it here, play the val
# episodes with it (train split by default), take its own dump-concept hidden states, refit the direction
# (scripts/refit_shape_vec.py: logistic probe; written only if it is a usable, not worse monitor) and swap the file the trainer reads (the plugin reloads it when HDS_SHAPE_RELOAD=1). Stops at STOP (HHMM).
# usage: refit_loop.sh NODE ARM GPU PORT LIVE_VEC_ON_NODE STOP [RULE]
#   RULE: other (default; dump vs every other turn, tau = their mean) | partial (dump vs partial hints, tau = their mean + 1 sd)
set -uo pipefail
NODE=$1; ARM=$2; GPU=$3; PORT=$4; LIVE=$5; STOP=$6; RULE=${7:-other}
SPLIT=${REFIT_SPLIT:-train}; N=${REFIT_N:-240}   # the monitor is fit on the policy's own training-seed episodes
C=/data2/xiangding/companion; R=$C/train/arm_${ARM}_4b/ckpt; V=$C/runs/val_refit; mkdir -p $V $C/ckpts/arm_$ARM
POOL="http://172.31.3.227:8101/v1 http://172.31.3.227:8102/v1 http://172.31.3.227:8103/v1 http://172.31.3.227:8104/v1 http://172.31.3.224:8101/v1 http://172.31.3.224:8102/v1"
W=/data1/xiangding/cvis-ablate/scripts/h20_env_ablate.sh; P=/data1/xiangding/cvis-ablate/env/bin/python
NP="-u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY"
log() { echo "[$(date +%H:%M)] $*"; }
while [ $(date +%H%M) -lt $STOP ]; do
  for c in $(timeout 30 ssh $NODE "ls -d $R/checkpoint-* 2>/dev/null | sort -t- -k2 -n"); do
    step=${c##*-}; [ -s $C/runs/probe/refit_${ARM}_$step.pt ] && continue
    dst=$C/ckpts/arm_$ARM/checkpoint-$step
    if [ ! -f $dst/model.safetensors.index.json ]; then
      ssh $NODE "tar -C $R -cf - --exclude='global_step*' checkpoint-$step" | tar -C $C/ckpts/arm_$ARM -xf -
      [ -f $dst/model.safetensors.index.json ] || { log "copy of step $step failed"; rm -rf $dst; continue; }
    fi
    log "step $step: val episodes"
    env $NP bash $C/scripts/eval_ckpt.sh $GPU $PORT $dst "$POOL" q35 $SPLIT $N $V/arm_${ARM}_step$step.jsonl plain full | tail -1
    log "step $step: hidden states"
    (cd $C && CUDA_VISIBLE_DEVICES=$GPU env $NP PYTHONPATH=/data1/xiangding/fla_pkgs bash $W python scripts/probe_data.py \
      --model $dst --concept dump --runs $V/arm_${ARM}_step$step.jsonl --n $N --out $C/runs/probe/dump_${ARM}_$step.npz 2>&1 | grep -E "contexts|wrote")
    prev=$C/runs/probe/refit_${ARM}_prev.pt; scp -q $NODE:$LIVE $prev
    if [ $RULE = partial ]; then
      (cd $C && $P scripts/make_shape_vec_partial.py runs/probe/dump_${ARM}_$step.npz 19 runs/probe/refit_${ARM}_$step.pt \
        $V/arm_${ARM}_step$step.jsonl --n $N --min-auroc 0.65 2>&1 | grep -v Warn)
    else
      (cd $C && $P scripts/refit_shape_vec.py runs/probe/dump_${ARM}_$step.npz 19 runs/probe/refit_${ARM}_$step.pt $prev 2>&1 | grep -v Warn)
    fi
    if [ ! -s $C/runs/probe/refit_${ARM}_$step.pt ]; then cp $prev $C/runs/probe/refit_${ARM}_$step.pt; log "step $step: kept the previous direction"; continue; fi
    scp -q $C/runs/probe/refit_${ARM}_$step.pt $NODE:$LIVE.tmp && ssh $NODE "mv $LIVE.tmp $LIVE" && log "step $step: direction swapped in"
  done
  sleep 60
done
log "refit loop stopped"
