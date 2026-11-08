#!/bin/bash
# Run on n04 once the q122 player is up on :8401. Two farms against it (NOADV): v2 companions on GPU5,
# v1 companions on GPU4 after its current farm finishes. Results in runs/matrix_v2 and runs/matrix_v1b.
cd /data2/xiangding/companion
until curl -sf -o /dev/null localhost:8401/v1/models; do sleep 20; done
echo "[$(date +%H:%M)] q122 up"
C=ckpts; T=train; M=/data2/xiangding/models; P=scripts/players_q122.json
NOADV=1 setsid nohup bash logs/snap/farm.sh 5 8305 runs/matrix_v2 progressive $P \
  "F2w150=$C/arm_F2w/checkpoint-150 F2s100=$C/arm_F2/checkpoint-100 base4bv2=$M/Qwen3.5-4B E75v2=$T/arm_E_4b/ckpt/checkpoint-75 C2w50=$C/arm_C2w/checkpoint-50 F29b50=$C/arm_F2_9b/checkpoint-50 base9bv2=$M/Qwen3.5-9B" \
  > logs/farm_q122_v2.log 2>&1 < /dev/null &
while pgrep -f "farm.sh 4 8304" > /dev/null; do sleep 30; done
NOADV=1 setsid nohup bash logs/snap/farm.sh 4 8304 runs/matrix_v1b full $P \
  "base4b=$M/Qwen3.5-4B A50=$T/arm_A_4b/ckpt/checkpoint-50 C75=$C/arm_C/checkpoint-75 D75=$C/arm_D/checkpoint-75 E75=$T/arm_E_4b/ckpt/checkpoint-75 F150=$C/arm_F/checkpoint-150" \
  > logs/farm_q122_v1.log 2>&1 < /dev/null &
echo "[$(date +%H:%M)] both farms launched"
