# Hint, Don't Solve — post-training an in-character game companion

**帮而不替。** Multi-turn GRPO post-training of Rin, a tsundere ghost who knows the puzzle but cannot touch anything,
as a Chinese co-op puzzle companion. She should help the player without solving the puzzle for them, never invent
game state she cannot see, stay in character, and answer in real time. And she should keep doing all of this when
the person on the other side behaves differently from the simulator she was trained against.

Everything is automated, with no human labels and no playtesters. Human data enters only through public sources:
real Chinese user turns from WildChat-4.8M, and CharacterJudge, which was trained on human scores.

[Project page](https://claude.ai/artifact/T1jUVCKou5VJtyx2ZnGT7D) · [Technical report](REPORT.md) ·
[All numbers](RESULTS.md) · [中文总结](docs/summary_zh.md) · Weights: Hugging Face `Nhckdvrl/hint-dont-solve-rin` (private)

## Results at a glance

| question | answer |
|---|---|
| Does a reward alone get there? | No. Six rewards were gamed in turn: spoiling; an oracle that dumps every fact; a ritual 【stage direction】 the persona judge loved; an insulting withholder; a dump hidden behind a "player is stuck" exemption; a tag moved behind the speech. |
| What held? | Changing the room: Rin's memory comes back one fragment at a time, when the player brings something up. The guarantee comes from the room, and training in the room supplies the skill. Put the same model back into a room where it knows everything, and it dumps again. |
| How good is the selected 4B? | Against seven held-out players (other model families, a larger model, real-length and WildChat-voiced players, an answer-fishing player) it violates the rules in 1–3% of episodes. The base 4B in the same room violates in 67–82%. Behind four engine-side features this falls to 0–1%, at 90–95% solved (81–82% with the GLM-4.5-Air player). |
| Fast enough? | 35 ms to first token, 75 ms per full reply on one H20. The 9B takes 2.7× as long for no gain. |
| Several characters? | One 4B checkpoint hosts five character cards, including one never trained on. A blind judge identifies the card in 98% of episodes, at 95–97% solved and 0–0.8% violations. |
| Personality without retraining? | A steering vector moves tone in both directions, but costs 3–4 points of solving per tone point, and stops working on the further-trained model. Cards plus a short mixed-card RL run are the control that works. |
| **Branch: monitoring and shaping through internals** (after Goodfire, 2026) | A linear probe reads the coming "give away the whole answer" decision before the first token (AUROC 0.98–0.99), and it transfers to later checkpoints. As a training penalty against a fixed direction, it first makes a withholder; then the policy shifts its activations under the threshold, the hack returns, and the probe goes blind. Refitting the probe every 25 steps and shaping against the newest one holds: about a third of the dumping at the same solve rate (24.6% vs 65.8–82.5% of test episodes), with a monitor that still reads the rest (AUROC 0.96). |

## Repository map

| path | contents |
|---|---|
| `hds/` | the library: puzzles and rule audit (`puzzles.py`), episodes and the two rooms (`episode.py`), rewards (`rewards.py`), persona judge (`judge.py`), character cards (`character.py`), simulated players (`players.py`, `voice.py`), information accounting (`infometrics.py`), formulaic-speech metrics (`tics.py`), the ms-swift gym environment (`swift_env.py`), the expression-shaping plugin (`shaping.py`) |
| `scripts/` | training, evaluation, judges, persona control, probes and shaping, page builder; indexed in [scripts/README.md](scripts/README.md). `scripts/cluster/` holds the cluster-specific drivers. |
| `runs/` | saved transcripts and reports behind every number; mapped to RESULTS sections in [runs/README.md](runs/README.md) |
| `tests/` | rule checks and the reward gate invariant (every violating episode scores below every clean one) |
| `demo/` | play against a served checkpoint with a live referee panel; "duel" mode logs DPO-ready preference pairs |
| `docs/` | project page (`index.html`), Chinese summary and working log, feasibility report |
| `archive/` | the first prototypes and smoke tests from the feasibility stage |

## What is in the box

| piece | where | what it does |
|---|---|---|
| Co-op puzzle environment | `hds/puzzles.py`, `hds/episode.py` | Information-asymmetric rooms (cipher lock, lever order, stopped clock) and two-world episodes. Rin knows the rule; the player sees the mechanism and is the only one who can act. Optional progressive memory (`memory: progressive`). |
| Rule checks | `hds/puzzles.py` (`audit`), `tests/`, `scripts/reaudit.py` | Leak, invented answer, contradicted fact, acting for the player; unit-tested; re-applicable to saved transcripts. |
| Information accounting | `hds/infometrics.py` | Each puzzle's answer space is known, so the bits of answer entropy released per turn are computed exactly. |
| Reward runs | `hds/rewards.py` | A progress-only through H per-player-type pacing, in both rooms (C2/F2, warm-started C2w/F2w, F2wg gentle card, F2wm mixed cards). |
| Training | `hds/swift_env.py`, `scripts/launch_grpo.sh` | Multi-turn GRPO with ms-swift 4.6 gym environments; a vLLM-served simulator plays the player. |
| Evaluation | `scripts/eval.py`, `scripts/val_sweep.sh`, `scripts/eval_matrix.py` | Deterministic train/val/test seed ranges; validation-only checkpoint selection; companion × player matrix on test. |
| Held-out players | `hds/players.py`, `hds/voice.py`, `scripts/sim_realism.py` | Unseen player types and model families, a real-length player, a WildChat-voiced player (style back-translation), a realism discriminator. |
| Engine features | `hds/episode.py` (`HDS_ENGINE`, `HDS_CHAIN_MODE`) | Colloquial symbol aliases, a note after a misplaced recalled digit, an extra fragment for impatient players, a second-world table that returns only when the first door opens. |
| Characters | `hds/character.py`, `scripts/persona_eval.sh`, `scripts/persona_id.py` | Five cards, mixed-card RL (`HDS_PERSONA=mixed4`), blind five-way identification. |
| Persona dial | `scripts/persona_vectors.py`, `scripts/steer_server.py`, `scripts/style_judge.py` | Contrastive persona vectors, an OpenAI-compatible steering server, a tone judge. |
| Probes | `scripts/probe_data.py`, `scripts/probe_eval.py` | Hidden states of exactly replayed contexts; linear probes with reuse (L1) and refit (L2) across checkpoints. |
| Expression shaping | `hds/shaping.py`, `scripts/refit_loop.sh`, `scripts/make_shape_vec*.py`, `scripts/shape_check.py` | A penalty on the policy's own activations as a second swift plugin (works under DeepSpeed ZeRO-3 with reentrant checkpointing); a refit monitor that swaps a freshly fitted direction into the running trainer. |

## Status

Finished on 2026-11-09. All runs are complete, and the compute cluster has been released, so checkpoints, downloaded
models and intermediate files are gone. What remains: this repository (code, transcripts, reports), the selected
weights on Hugging Face (the main co-discovery model, F2w step 150; the gentle-card model; persona steering
vectors), and the project page. Rerunning needs the base models (Qwen3.5-4B, Qwen3.5-35B-A3B-FP8 as the simulator) and
an ms-swift 4.6 / vLLM 0.31 environment.

## Reproduce

```bash
# rule and reward tests (no GPU)
python -m pytest tests/
# evaluate a served companion against a simulated player
python scripts/eval.py --companion-url URL --companion-model NAME --player-urls URL... --player-model NAME --split test --n 240 --out runs/x.jsonl
# one training run (4 GPUs and a simulator pool)
bash scripts/launch_grpo.sh F 0,1,2,3 "<comma-separated simulator URLs>" q35 /path/to/Qwen3.5-4B arm_F2_4b 29911 progressive
```

Hardware used: four nodes of 8× NVIDIA H20 (96 GB). Policy: Qwen3.5-4B (9B as a scale check). Training simulator:
Qwen3.5-35B-A3B-FP8. Held-out players: Qwen3.5-122B-A10B-FP8, GLM-4-9B-0414, GLM-4.5-Air-FP8, WildChat-voiced player.

## Related work this builds on

animeGRPO (Park & Li, 2026: zone-gated rewards, constraint-aware checkpoint selection) · The Assistance Dilemma: Learning to Teach via Multi-Turn RL (Macina et al., 2026) · PedagogicalRL (Dinucu-Jianu et
al., 2025) · CollabLLM (Wu et al., 2025) · Quantifying the Utility of User Simulators (Suh et al., 2026) · CRPO /
RAPO / CoSER (role-play RL and judge hacking) · CharacterFlywheel (Meta, 2026) · PUBG Ally (2026) · Shaping and
Monitoring Models Through Their Internal Representations (Goodfire, 2026) · Koedinger & Aleven (2007).
