# runs/

Saved outputs behind every number in `RESULTS.md`. Episode files are JSON lines: one episode per line with the
spec, the full transcript (player and companion turns, rule-audit flags per companion turn) and `metrics`. File
names are `<companion>__<player>.jsonl` in matrix directories and `arm_<run>_step<N>.jsonl` in validation sweeps.
Model weights are not here (selected ones are on Hugging Face, `Nhckdvrl/hint-dont-solve-rin`).

| path | what | used in |
|---|---|---|
| `feasibility/` | first baseline and prompting runs before any training | `docs/feasibility_report_zh.md`, RESULTS §1 |
| `val/` | validation sweeps of every reward run, every 25 steps (96 episodes, training simulator) | RESULTS §2–3, §7a–7b |
| `train_curves/` | per-step trainer logs and per-episode training metrics (gzipped) of every run | RESULTS §2 |
| `matrix/`, `matrix_extra/` | full-knowledge room: companions × held-out players, test split | RESULTS §4–6 |
| `matrix_v1b/` | the per-player-type pacing run (fourth and fifth hacks) | RESULTS §7a |
| `matrix_v2/` | co-discovery room: companions × held-out players, test split | RESULTS §7b–7c, page |
| `matrix_2x2/` | swapped rooms: models trained in one room evaluated in the other | RESULTS §7d |
| `chainfix/` | two-world cipher slice under room variants (when the second table comes back) | RESULTS §7e |
| `realism/` | simulator-vs-real-user style discriminator | RESULTS §7 |
| `steer/`, `steer_matrix/` | persona vectors and steering evaluations (the personality dial) | RESULTS §9, §12 |
| `latency.jsonl`, `throughput.jsonl` | single-player latency; latency under 1–128 simultaneous players | RESULTS §10 |
| `charjudge_matrix.txt` | CharacterJudge scores per companion | RESULTS §4 |
| `matrix_engine/` | engine-side features without retraining, and the final product package | RESULTS §11, §13 |
| `persona_eval/` | five character cards per checkpoint, and blind persona identification | RESULTS §12 |
| `probe/` | probe reports (persona, dump, violation; shaped vs unshaped) and `vectors/`, the shaping directions | RESULTS §14–15 |
| `val_E/` | validation sweeps of the expression-shaping runs | RESULTS §15 |
| `probe_src_E/` | test-split episodes of shaped and unshaped checkpoints (the probes' source contexts) | RESULTS §15 |
| `val_refit/` | episodes the refit monitor played before each refit (`bad_copy/`: one evaluation of a half-copied checkpoint, kept for the record) | RESULTS §15 |
| `shape_logs/` | per-step shaping penalty, projection and reload log of each shaping run, and the refit-loop logs | RESULTS §15 |

`wildchat_zh_followups_*.jsonl` (derived from WildChat-4.8M) are git-ignored.
