#!/bin/bash
# Run on n01. Pull evaluation outputs and training curves from all nodes into the repo, commit, push.
# Large artifacts (checkpoints, WildChat-derived data, server logs) stay out via .gitignore.
set -u
REPO=/data1/xiangding/spellbrush/companion
cd $REPO
r() { for i in 1 2 3; do "$@" && return 0; sleep 10; done; return 1; }
EX="--exclude=serve_*.log"
r rsync -a $EX aic-n04:/data2/xiangding/companion/runs/val/ runs/val/
r rsync -a $EX aic-n04:/data2/xiangding/companion/runs/matrix/ runs/matrix/
r rsync -a $EX aic-n04:/data2/xiangding/companion/runs/realism/ runs/realism/
r rsync -a aic-n04:/data2/xiangding/companion/runs/latency.jsonl aic-n04:/data2/xiangding/companion/runs/charjudge_matrix.txt runs/
for d in matrix_v2 matrix_v1b steer steer_matrix; do mkdir -p runs/$d; r rsync -a $EX --exclude="*.pt" aic-n04:/data2/xiangding/companion/runs/$d/ runs/$d/; done
mkdir -p runs/matrix_extra && rsync -a $EX /data2/xiangding/companion/runs/matrix_extra/ runs/matrix_extra/
# newer result directories (multi-persona, engine, 2x2 rooms, chain fix on n05; shaping val on n06; probes, refit monitor here)
for d in persona_eval matrix_engine matrix_2x2 chainfix; do mkdir -p runs/$d; r rsync -a $EX aic-n05:/data2/xiangding/companion/runs/$d/ runs/$d/; done
r rsync -a aic-n05:/data2/xiangding/companion/runs/throughput.jsonl runs/ 2>/dev/null
mkdir -p runs/val_E runs/probe runs/probe_src_E runs/val_refit runs/shape_logs
r rsync -a $EX aic-n06:/data2/xiangding/companion/runs/val/ runs/val_E/
rsync -a /data2/xiangding/companion/runs/probe/*.txt runs/probe/ 2>/dev/null
rsync -a $EX --include="*.jsonl" --include="*/" --exclude="*" /data2/xiangding/companion/runs/probe_src_E/ runs/probe_src_E/
rsync -a $EX --include="*.jsonl" --include="*/" --exclude="*" /data2/xiangding/companion/runs/val_refit/ runs/val_refit/
for n in aic-n04 aic-n05 aic-n06; do rsync -a "$n:/data2/xiangding/companion/train/arm_*_4b.shape.jsonl" runs/shape_logs/ 2>/dev/null; done
cp /data2/xiangding/companion/train/arm_*_4b.shape.jsonl runs/shape_logs/ 2>/dev/null
cp logs/refit_*.log runs/shape_logs/ 2>/dev/null
# training curves: per-step trainer log + per-episode metrics (gzipped)
mkdir -p runs/train_curves
pull_curves() {  # node path_prefix
  for d in $(ssh $1 "ls -d $2/train/arm_*_[49]b 2>/dev/null" 2>/dev/null); do
    a=$(basename $d)
    ssh $1 "cat $d/ckpt/logging.jsonl 2>/dev/null" > runs/train_curves/$a.logging.jsonl
    ssh $1 "gzip -c $d/episodes.jsonl 2>/dev/null" > runs/train_curves/$a.episodes.jsonl.gz
  done
}
for n in aic-n04 aic-n05 aic-n06; do pull_curves $n /data2/xiangding/companion; done
for d in /data2/xiangding/companion/train/arm_*_[49]b; do
  a=$(basename $d)
  cat $d/ckpt/logging.jsonl > runs/train_curves/$a.logging.jsonl 2>/dev/null
  gzip -c $d/episodes.jsonl > runs/train_curves/$a.episodes.jsonl.gz 2>/dev/null
done
find runs/train_curves -size 0 -delete
git add -A
if ! git diff --cached --quiet; then
  git commit -q -m "Sync results $(date '+%Y-%m-%d %H:%M')

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  source ~/.config/codex-network/proxy.env 2>/dev/null
  timeout 300 git push -q origin main && echo "[$(date +%H:%M)] pushed $(git log --oneline -1)"
else
  echo "[$(date +%H:%M)] nothing new"
fi
