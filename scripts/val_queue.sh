#!/bin/bash
# Run on n01 (setsid): validation sweep of all arms' checkpoints on n04 GPU3; progress in logs/val_queue.log.
L=/data1/xiangding/spellbrush/companion/logs/val_queue.log
echo "[$(date +%H:%M)] val sweep start" >> $L
bash /data1/xiangding/spellbrush/companion/scripts/val_sweep.sh 3 8300 96 "E:aic-n04 F:aic-n06 C:aic-n05 D:aic-n05 C2:aic-n05:progressive F2:aic-n05:progressive" >> $L 2>&1
echo "[$(date +%H:%M)] val sweep done" >> $L
