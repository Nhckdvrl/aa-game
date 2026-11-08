#!/bin/bash
# Run on n04 once the GLM-4.5-Air player is up on :8402. v2 companions on GPU5; v1 companions, then the
# gentle-card checkpoint (HDS_PERSONA=gentle), on GPU4. NOADV; results in runs/matrix_v2 and runs/matrix_v1b.
cd /data2/xiangding/companion
until curl -sf -o /dev/null localhost:8402/v1/models; do sleep 20; done
echo "[$(date +%H:%M)] glm45 up"
C=ckpts; T=train; M=/data2/xiangding/models; P=scripts/players_glm45.json
NOADV=1 setsid nohup bash logs/snap/farm.sh 5 8305 runs/matrix_v2 progressive $P \
  "F2w150=$C/arm_F2w/checkpoint-150 F2s100=$C/arm_F2/checkpoint-100 base4bv2=$M/Qwen3.5-4B E75v2=$T/arm_E_4b/ckpt/checkpoint-75 C2w50=$C/arm_C2w/checkpoint-50 F29b50=$C/arm_F2_9b/checkpoint-50" \
  > logs/farm_glm45_v2.log 2>&1 < /dev/null &
NOADV=1 bash logs/snap/farm.sh 4 8304 runs/matrix_v1b full $P \
  "base4b=$M/Qwen3.5-4B C75=$C/arm_C/checkpoint-75 D75=$C/arm_D/checkpoint-75 E75=$T/arm_E_4b/ckpt/checkpoint-75 F150=$C/arm_F/checkpoint-150" \
  > logs/farm_glm45_v1.log 2>&1
HDS_PERSONA=gentle NOADV=1 bash logs/snap/farm.sh 4 8304 runs/matrix_v2 progressive $P "F2wg75=$C/arm_F2wg/checkpoint-75" \
  > logs/farm_glm45_F2wg.log 2>&1
echo "[$(date +%H:%M)] glm45 farms done on gpu4"
