#!/bin/bash
# Run on n01. Persona "fragments" on the main model (F2w step 150, tsundere card): single vectors and a mix, layer 19,
# each on one GPU against the test split (35B + real-length players). Waits for the vectors and free GPUs.
cd /data2/xiangding/companion
C=train/arm_F2w_4b/ckpt/checkpoint-150; V=runs/steer
until [ -s $V/vec150_gentle.pt ] && [ -s $V/vec150_kuudere.pt ] && [ -s $V/vec150_chuuni.pt ]; do sleep 20; done
run() {  # GPU PORT NAME ARGS
  until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$(( $1 + 1 ))p") -lt 4000 ]; do sleep 30; done   # other users run small services (~2 GB) on some n01 GPUs
  PLAYERS=scripts/players_2.json setsid nohup bash scripts/steer_matrix.sh $1 $2 $3 "$4" > logs/frag_$3.log 2>&1 < /dev/null &
}
run 2 8602 frag_gentle "--model $C --mix $V/vec150_gentle.pt:19:1"
run 3 8603 frag_kuudere "--model $C --mix $V/vec150_kuudere.pt:19:1"
run 4 8604 frag_chuuni "--model $C --mix $V/vec150_chuuni.pt:19:1"
run 5 8605 frag_mix_gc "--model $C --mix $V/vec150_gentle.pt:19:0.5,$V/vec150_chuuni.pt:19:0.5"
run 6 8606 frag_kuudere15 "--model $C --mix $V/vec150_kuudere.pt:19:1.5"
run 7 8607 frag_chuuni15 "--model $C --mix $V/vec150_chuuni.pt:19:1.5"
wait
echo "[$(date +%H:%M)] fragments launched"
