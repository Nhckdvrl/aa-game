Driver scripts used on the cluster for the expression-shaping branch (RESULTS §15), kept as run records. Paths are
the cluster's (`/data2/xiangding/companion` on each node, ms-swift env under `/data1/xiangding/cvis-ablate/env`).

- `val_loop_E.sh` + `val_sweep_v4.sh`: second validation worker for the shaping arms (copies each finished
  checkpoint to the eval node, waits for `trainer_state.json`).
- `probe_shaped.sh`, `probe_offset.sh`, `probe_big.sh`, `final_probes.sh`: test-split episodes plus dump-concept
  hidden states for shaped vs unshaped checkpoints (`scripts/probe_data.py`, `--offset K` for positions inside the
  reply), evaluated with `scripts/probe_eval.py` and `scripts/shape_check.py`.
- `sync_results_v1.sh`: hourly pull of results from all nodes into this repo.
- The refit monitor itself is `scripts/refit_loop.sh`; shaping directions are in `runs/probe/vectors/`.

Launch pattern for a shaping arm (full-knowledge room, base 4B, penalty at the start of each reply):
`HDS_SHAPE_VEC=runs/probe/vectors/shape_dump_L19.pt HDS_SHAPE_LAMBDA=5 HDS_SHAPE_POS=gen HDS_SHAPE_WIN=6
HDS_SHAPE_LOG=train/arm_X.shape.jsonl [HDS_SHAPE_RELOAD=1] [HDS_SHAPE_REL=1.5] bash scripts/launch_grpo.sh E GPUS POOL q35 Qwen3.5-4B arm_X_4b PORT full`
(absolute paths: the launcher changes directory).
