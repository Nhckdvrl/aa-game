# scripts/

Everything that runs the pipeline. Library code (environment, rewards, judges, players) lives in `hds/`. Python
scripts run from the repository root (`python scripts/x.py`). Shell scripts were written for the H20 cluster: they
assume the repo at `/data2/xiangding/companion` and the ms-swift/vLLM env behind
`/data1/xiangding/cvis-ablate/scripts/h20_env_ablate.sh`, so adjust those two paths to rerun elsewhere.

## Train
| script | what it does |
|---|---|
| `make_train_data.py` | swift GRPO dataset: one row per training episode spec (train split, training player types) |
| `launch_grpo.sh` | one multi-turn GRPO run: `launch_grpo.sh ARM GPUS PLAYER_URLS PLAYER_MODEL POLICY RUN_NAME [PORT] [MEMORY]`; env knobs `HDS_PERSONA`, `HDS_ENGINE`, `HDS_CHAIN_MODE`, `HDS_SHAPE_*` |
| `select_ckpts.py` | constraint-aware checkpoint pick per run from the validation sweep |
| `soup.py` | average two checkpoints shard by shard |

## Evaluate
| script | what it does |
|---|---|
| `eval.py` | one companion against a simulated player on fixed split seeds |
| `eval_ckpt.sh` | serve one checkpoint on one GPU (vLLM), run `eval.py`, stop the server |
| `val_sweep.sh`, `summarize_val.py` | evaluate every saved checkpoint on the validation split; per-step summary |
| `eval_matrix.py`, `matrix_run.sh`, `farm.sh`, `matrix_table.py` | companion × player transfer matrix on the test split, and its table |
| `eval_subset.py`, `chainfix.sh` | a filtered slice of the test split (two-world cipher episodes) under room variants |
| `reaudit.py` | re-apply the current rule checks to saved transcripts |
| `info_report.py`, `dump_rate.py` | exact information release; how often one reply hands over the whole answer |
| `stats_and_recovery.py` | bootstrap confidence intervals; behavior right after a failed attempt |
| `implicit_signals.py`, `fit_type_weights.py`, `request_respect.py` | implicit player signals in the logs; per-type reward weights; does the companion stop when asked not to spoil |
| `latency_bench.py`, `throughput.sh` | single-player latency on replayed contexts; latency at 1–128 simultaneous players |

## Judges and simulated players
| script | what it does |
|---|---|
| `judge_runs.py`, `charjudge_runs.py` | persona judge / CharacterJudge scores for saved runs |
| `sim_realism.py` | how human-like a simulator's language is (char n-gram discriminator against real WildChat users) |
| `extract_wildchat_followups.py`, `backtranslate_style.py`, `make_voice_sft.py`, `sft_voice.sh`, `voice_demo.py` | the WildChat-voiced player: real Chinese follow-ups → neutral drafts → a 4B model that rewrites simulator lines in real users' style |
| `players/*.json` | player pools (endpoints and player types) used by the matrix and steering runs |

## Characters and persona control
| script | what it does |
|---|---|
| `persona_eval.sh`, `persona_id.py` | run the test split once per character card; blind five-way identification of which card the replies came from |
| `persona_vectors.py`, `steer_server.py`, `steer_matrix.sh`, `style_judge.py` | contrastive persona vectors, an OpenAI-compatible server that steers activations, its evaluation, and a tone judge |

## Probes and expression shaping (Goodfire branch, RESULTS §14–15)
| script | what it does |
|---|---|
| `probe_data.py` | hidden states at the generation position (or `--offset K` tokens into the reply) of exactly replayed contexts; concepts `persona`, `dump`, `viol` |
| `probe_eval.py` | linear probes per layer: refit on each model (L2) and reuse of a reference model's probe (L1) |
| `make_shape_vec.py`, `make_shape_vec_partial.py` | a shaping direction from probe data (dump vs other turns / dump vs partial hints) |
| `shape_check.py` | where a model's dump and non-dump turns sit on a penalized direction |
| `refit_shape_vec.py`, `refit_loop.sh` | the refit monitor: refit the direction on each new checkpoint's own episodes and swap it into the running trainer |

The penalty itself is `hds/shaping.py`, a second swift plugin.

## Page
`build_page_v2.py` builds `docs/index.html` from the saved runs (it imports helpers from `build_page.py`).

## cluster/
Drivers and babysitters for the specific runs on the cluster (node names and GPU ids hard-coded): restarts, queues,
evaluation farms, the second validation worker, the final probe batches, the hourly results sync. Kept as a
record of how the numbers were produced; see `cluster/README_shaping_runs.md`.
