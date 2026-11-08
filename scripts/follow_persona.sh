#!/bin/bash
# Run on n01: when F2wm saves checkpoint STEP on n06, relay it to n05 (no node-to-node SSH) and run the per-card
# evaluation there on GPU 7. usage: follow_persona.sh STEP [STEP...]
for STEP in "$@"; do
  until timeout 30 ssh -o ConnectTimeout=10 aic-n06 "test -f /data2/xiangding/companion/train/arm_F2wm_4b/ckpt/checkpoint-$STEP/model.safetensors.index.json" 2>/dev/null; do sleep 60; done
  sleep 60  # let the save finish
  for i in 1 2 3; do
    ssh aic-n06 "tar -C /data2/xiangding/companion/train/arm_F2wm_4b/ckpt -cf - checkpoint-$STEP" | ssh aic-n05 "tar -C /data2/xiangding/companion/ckpts/arm_F2wm -xf -" \
      && ssh aic-n05 "test -f /data2/xiangding/companion/ckpts/arm_F2wm/checkpoint-$STEP/model.safetensors.index.json" && break
    sleep 15
  done
  until ! timeout 30 ssh aic-n05 'pgrep -f "persona_eval.sh 7 " > /dev/null' 2>/dev/null; do sleep 30; done
  timeout 40 ssh aic-n05 "cd /data2/xiangding/companion; (setsid nohup bash scripts/persona_eval.sh 7 8317 F2wm$STEP ckpts/arm_F2wm/checkpoint-$STEP > logs/pe_F2wm$STEP.log 2>&1 < /dev/null &)"
  echo "[$(date +%H:%M)] launched persona_eval F2wm$STEP"
done
