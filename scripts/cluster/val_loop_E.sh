#!/bin/bash
# Second validation worker for the expression-shaping arms (n06 GPU4), every 10 min until 08:00.
# Arms are re-read from logs/val_arms_E.txt each pass.
L=/data1/xiangding/spellbrush/companion/logs/val_queue_E.log
while [ $(date +%H%M) -lt 0800 ] || [ $(date +%H%M) -gt 1200 ]; do
  echo "[$(date +%H:%M)] val E pass" >> $L
  EVAL=aic-n06 bash /data1/xiangding/spellbrush/companion/logs/snap/val_sweep_v4.sh 4 8304 96 "$(cat /data1/xiangding/spellbrush/companion/logs/val_arms_E.txt)" >> $L 2>&1
  sleep 600
done
