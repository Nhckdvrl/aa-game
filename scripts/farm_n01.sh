#!/bin/bash
# Run on n01 (setsid nohup, from logs/snap). For each companion: the 4-player test matrix (n=240), then the
# held-out 套话型 adversarial player (n=120). usage: farm_n01.sh GPU PORT OUTDIR MEMORY "name=path ..."
GPU=$1; PORT=$2; OUT=$3; MEMORY=$4; COMPANIONS=$5
S=/data2/xiangding/companion/scripts
for pair in $COMPANIONS; do
  bash $S/matrix_run.sh $GPU $PORT $OUT $S/players_n01.json 240 "$pair" $MEMORY
  PTYPES=套话型 bash $S/matrix_run.sh $GPU $PORT $OUT $S/players_adv.json 120 "$pair" $MEMORY
done
echo "[$(date +%H:%M)] farm gpu$GPU done"
