#!/bin/bash
# Is the dump decision readable later in the reply (moved past the penalized window) or not at all?
cd /data2/xiangding/companion
W=/data1/xiangding/cvis-ablate/scripts/h20_env_ablate.sh; V=/data1/xiangding/spellbrush/companion/runs/val_E; O=runs/probe
export PYTHONPATH=/data1/xiangding/fla_pkgs
run() { CUDA_VISIBLE_DEVICES=$1 env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY bash $W python scripts/probe_data.py "${@:2}"; }
for K in 8 24; do
  run 0 --model train/arm_Ebase_4b/ckpt/checkpoint-50 --concept dump --runs $V/arm_Ebase_step50.jsonl --n 96 --offset $K --out $O/dump_Ebase50_k$K.npz > logs_probe_Ebase50_k$K.txt 2>&1 &
  run 2 --model ckpts/arm_Eshg5/checkpoint-50 --concept dump --runs $V/arm_Eshg5_step50.jsonl --n 96 --offset $K --out $O/dump_Eshg5_50_k$K.npz > logs_probe_Eshg5_50_k$K.txt 2>&1 &
  wait
done
echo "[$(date +%H:%M)] offset probes done"
