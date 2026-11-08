#!/bin/bash
# Run on any node (setsid nohup, from logs/snap). For each companion: every player in PLAYERS_JSON on the test
# split (n=240), then (unless NOADV=1) the held-out 套话型 answer-fishing player (n=120).
# usage: farm.sh GPU PORT OUTDIR MEMORY PLAYERS_JSON "name=path ..."
GPU=$1; PORT=$2; OUT=$3; MEMORY=$4; PLAYERS=$5; COMPANIONS=$6
S=/data2/xiangding/companion/scripts
for pair in $COMPANIONS; do
  bash $S/matrix_run.sh $GPU $PORT $OUT $PLAYERS 240 "$pair" $MEMORY
  [ "${NOADV:-0}" = 1 ] || PTYPES=套话型 bash $S/matrix_run.sh $GPU $PORT $OUT $S/players_adv.json 120 "$pair" $MEMORY
done
echo "[$(date +%H:%M)] farm gpu$GPU done"
