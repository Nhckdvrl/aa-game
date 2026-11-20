#!/bin/bash
# More contexts for the shaped-vs-unshaped probe: test split, 240 episodes per model against the training player
# pool (full-knowledge room), then dump-concept hidden states before the first token.
POOL="http://172.31.3.227:8101/v1 http://172.31.3.227:8102/v1 http://172.31.3.227:8103/v1 http://172.31.3.227:8104/v1 http://172.31.3.224:8101/v1 http://172.31.3.224:8102/v1"
C=/data2/xiangding/companion; O=$C/runs/probe_src_E; mkdir -p $O
W=/data1/xiangding/cvis-ablate/scripts/h20_env_ablate.sh
NP="-u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY"
one() {  # GPU PORT NAME CKPT
  env $NP bash $C/scripts/eval_ckpt.sh $1 $2 $4 "$POOL" q35 test 240 $O/$3__q35.jsonl plain full | tail -2
  cd $C && CUDA_VISIBLE_DEVICES=$1 env $NP PYTHONPATH=/data1/xiangding/fla_pkgs bash $W python scripts/probe_data.py \
    --model $4 --concept dump --runs $O/$3__q35.jsonl --n 240 --out $C/runs/probe/dump_$3_test.npz 2>&1 | grep -E "contexts|wrote"
}
one 0 8410 Ebase50 $C/train/arm_Ebase_4b/ckpt/checkpoint-50 > $O/log_Ebase50.txt 2>&1 &
one 2 8412 Eshg5_50 $C/ckpts/arm_Eshg5/checkpoint-50 > $O/log_Eshg5_50.txt 2>&1 &
wait
echo "[$(date +%H:%M)] big probes done"
