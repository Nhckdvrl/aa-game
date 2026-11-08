# Hint, Don't Solve — post-training an in-character game companion

A small but complete post-training project for an LLM companion that plays *with* the player: it should
help the player solve puzzles without solving them for the player, never invent game state it cannot see,
stay in character, and answer briefly enough for real-time play — and it should keep doing so when the
person on the other side behaves differently from the simulator it was trained against.

Everything here is automated: no human labelling or playtesting was used. Human data enters only through
public sources (real Chinese user turns from WildChat-4.8M; CharacterJudge, trained on human scores).

Results so far: [RESULTS.md](RESULTS.md).

## What is in the box

| piece | where | what it does |
|---|---|---|
| Co-op puzzle environment | `hds/puzzles.py`, `hds/episode.py` | Information-asymmetric rooms (cipher lock, lever order, stopped clock) plus two-world "memory" episodes. The companion (Rin, a tsundere ghost who cannot touch anything) knows the rule; the player sees the mechanism and is the only one who can act. Optional progressive memory (`memory: progressive`): her knowledge returns one fragment at a time, triggered by what the player brings up. |
| Rule checks | `hds/puzzles.py` (`audit`), `tests/` | Leak (states the full answer before the player), invented answer, contradicted fact, acting for the player. Unit-tested; `scripts/reaudit.py` re-applies the current rules to saved transcripts. |
| Information accounting | `hds/infometrics.py` | Each puzzle's answer space is known, so the bits of answer entropy the companion releases per turn are computed exactly; also release when the player is stuck vs. not. |
| Formulaic-speech metrics | `hds/tics.py` | Same leading 【action】 tag on most replies, near-copies of the previous reply, distinct-2. Catches what LLM persona judges miss. |
| Persona judges | `hds/judge.py`, CharacterJudge | Length-neutral episode rubric (used in rewards D–G) and CharacterJudge (evaluation only: its public rubric rewards long replies). |
| Simulated players | `hds/players.py`, `hds/voice.py` | 4 training player types + 2 held-out types; JSON-schema-constrained output. Held-out families: GLM-4-9B, and a "human voice" player whose lines are rewritten by a 4B model fine-tuned on 60k real Chinese WildChat follow-ups (style back-translation, with a fact-preservation guard). |
| Reward arms | `hds/rewards.py` | A progress-only · B weighted penalties · C zone gate (any violation → −1) · D gate + persona · E gate + persona, no turn-efficiency bonus · F E + anti-formula penalty · G F + pacing penalty. The gate invariant (every violating episode scores below every clean one) is unit-tested for C–G. |
| Training | `hds/swift_env.py`, `scripts/launch_grpo.sh` | Multi-turn GRPO with ms-swift 4.6 gym environments; the policy plays the companion, a vLLM-served simulator plays the player. 16 prompts × 8 rollouts per step. |
| Evaluation | `scripts/eval.py`, `scripts/val_sweep.sh`, `scripts/eval_matrix.py`, `scripts/matrix_run.sh` | Deterministic train/val/test seed ranges (`hds/splits.py`); constraint-aware checkpoint selection on val; companion × player transfer matrix on test. |
| Simulator realism | `scripts/sim_realism.py` | Topic-free char n-gram discriminator (real vs simulated player lines) with a real-vs-real sanity row. |
| Latency | `scripts/latency_bench.py` | Single-user TTFT / first-sentence / full-reply time on replayed contexts. |
| Demo & preference tool | `demo/` | Play with any served checkpoint; a referee panel shows the rule checks live; "duel" mode shows two anonymized candidates and logs the pick as a DPO-ready preference pair. |

## Reproduce

```bash
# unit tests (no GPU)
python -m pytest tests/        # or run the test functions directly
# baseline evaluation against a simulator
python scripts/eval.py --companion-url URL --companion-model NAME --player-urls URL... --player-model NAME --split test --n 240 --out runs/x.jsonl
# one training arm (on a node with 4 free GPUs and a simulator pool)
bash scripts/launch_grpo.sh C 4,5,6,7 "<comma-separated simulator URLs>" q35 /path/to/Qwen3.5-4B arm_C_4b 29911
```

Hardware used: 3 nodes × 8 NVIDIA H20 (96 GB). Policy: Qwen3.5-4B (9B as a scale check). Training simulator:
Qwen3.5-35B-A3B-FP8. Held-out players: Qwen3.5-122B-A10B-FP8, GLM-4-9B-0414, WildChat-voiced player.

## Related work this builds on

animeGRPO (Park & Li, 2026: zone-gated perceptual rewards, constraint-aware checkpoint selection) ·
The Assistance Dilemma: Learning to Teach via Multi-Turn RL (Macina et al., 2026: handover gates for tutors) ·
PedagogicalRL (Dinucu-Jianu et al., 2025) · CollabLLM (Wu et al., 2025) · Quantifying the Utility of User
Simulators (Suh et al., 2026) · CRPO / RAPO / CoSER (role-play RL and its judge hacking) · CharacterFlywheel
(Meta, 2026: reward-overfitting monitoring) · PUBG Ally (2026: game teammate failure modes) ·
Koedinger & Aleven (2007: the assistance dilemma).
