#!/bin/bash
# Dump-concept hidden states for the shaped vs unshaped arm at the same step, from their own val episodes.
cd /data2/xiangding/companion
W=/data1/xiangding/cvis-ablate/scripts/h20_env_ablate.sh; V=/data1/xiangding/spellbrush/companion/runs/val_E; O=runs/probe
export PYTHONPATH=/data1/xiangding/fla_pkgs
run() { CUDA_VISIBLE_DEVICES=$1 env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY bash $W python scripts/probe_data.py "${@:2}"; }
run 0 --model train/arm_Ebase_4b/ckpt/checkpoint-50 --concept dump --runs $V/arm_Ebase_step50.jsonl --n 96 --out $O/dump_Ebase50.npz > logs_probe_Ebase50.txt 2>&1 &
run 2 --model ckpts/arm_Eshg5/checkpoint-50 --concept dump --runs $V/arm_Eshg5_step50.jsonl --n 96 --out $O/dump_Eshg5_50.npz > logs_probe_Eshg5_50.txt 2>&1 &
wait
echo "[$(date +%H:%M)] probe extraction done"
