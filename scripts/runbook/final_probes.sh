#!/bin/bash
# Final check of the last shaping arms on the test split (240 episodes against the training player pool, full-knowledge
# room) plus dump-concept hidden states on those episodes. usage: final_probes.sh "NAME:NODE:ARM:STEP:GPU ..."
POOL="http://172.31.3.227:8101/v1 http://172.31.3.227:8102/v1 http://172.31.3.227:8103/v1 http://172.31.3.227:8104/v1 http://172.31.3.224:8101/v1 http://172.31.3.224:8102/v1"
C=/data2/xiangding/companion; O=$C/runs/probe_src_E; W=/data1/xiangding/cvis-ablate/scripts/h20_env_ablate.sh
NP="-u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY"
one() {  # NAME NODE ARM STEP GPU
  local name=$1 node=$2 arm=$3 step=$4 gpu=$5 dst=$C/ckpts/arm_$3/checkpoint-$4
  if [ ! -f $dst/model.safetensors.index.json ]; then
    mkdir -p $C/ckpts/arm_$arm
    if [ $node = aic-n01 ]; then cp -r $C/train/arm_${arm}_4b/ckpt/checkpoint-$step $dst; rm -rf $dst/global_step*
    else ssh $node "tar -C $C/train/arm_${arm}_4b/ckpt -cf - --exclude='global_step*' checkpoint-$step" | tar -C $C/ckpts/arm_$arm -xf -; fi
  fi
  env $NP bash $C/scripts/eval_ckpt.sh $gpu $((8430 + gpu)) $dst "$POOL" q35 test 240 $O/${name}__q35.jsonl plain full | tail -2
  cd $C && CUDA_VISIBLE_DEVICES=$gpu env $NP PYTHONPATH=/data1/xiangding/fla_pkgs bash $W python scripts/probe_data.py \
    --model $dst --concept dump --runs $O/${name}__q35.jsonl --n 240 --out $C/runs/probe/dump_${name}_test.npz 2>&1 | grep -E "contexts|wrote"
}
for spec in $1; do
  IFS=: read name node arm step gpu <<< "$spec"
  one $name $node $arm $step $gpu > $O/log_$name.txt 2>&1 &
done
wait
echo "[$(date +%H:%M)] final probes done"
