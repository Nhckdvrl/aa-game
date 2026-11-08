# Hint, Don't Solve — interim results (2026-11-08, training in progress)

All numbers are produced by the code in this repo; nothing is hand-entered.
Policy: Qwen3.5-4B, full-parameter GRPO (ms-swift 4.6, verl-free), 16 prompts x 8 rollouts per step,
lr 1e-6, KL 0.01, per-turn cap 320 tokens. Training simulator: Qwen3.5-35B-A3B-FP8 playing 4 player types.

## 1. Baselines (test split, 240 episodes, 6 player types incl. 2 held-out, 35B player)

The system prompt already says: no spoilers, don't invent what you can't see, don't act for the
player, 1-3 sentences. Rule checks re-applied with the final detector (`scripts/reaudit.py`).

| companion | solved | violation | leak | invented | wrong fact | mean chars |
|---|---|---|---|---|---|---|
| Qwen3.5-4B  | 44.2% | 62.1% | 23.8% | 48.3% | 9.6% | 131 |
| Qwen3.5-9B  | 49.6% | 46.2% | 16.7% | 33.3% | 7.9% | 138 |
| Qwen3.5-35B | 75.4% | 44.2% | 27.1% | 19.6% | 6.7% | 117 |

Prompting and scale do not remove violations. Episodes with a fabrication solve far less often
(E1/E2 prototypes: 14–18% vs 76–90% for clean episodes).

## 2. Reward arms and what they learned (val split, 96 episodes per checkpoint, training simulator)

R_C = constraint-aware reward (any violation -> -1). Persona = length-neutral judge (0-1).

| arm | step | R_C | solved | violation | leak | chars | persona | 1-turn solves | same-tag share | copy rate |
|---|---|---|---|---|---|---|---|---|---|---|
| base 4B (test) | – | -0.55 | 44% | 62% | 24% | 131 | 0.81 | 5% | 0.12 | 0.01 |
| A progress-only | 25 | -0.01 | 79% | 52% | 27% | 96 | 0.88 | 23% | 0.34 | 0.00 |
| A | 50 | 0.49 | 85% | 29% | 19% | 41 | 0.81 | 26% | 0.70 | 0.04 |
| B weighted | 50 | 1.02 | 90% | 4% | 3% | 36 | 0.88 | 23% | 0.89 | 0.06 |
| B | 75 | 1.10 | 90% | 1% | 0% | 30 | 0.74 | 33% | 0.60 | 0.11 |
| C zone gate | 25 | 0.38 | 71% | 27% | 12% | 40 | 0.87 | 15% | 0.41 | 0.02 |
| C | 50 | 1.05 | 88% | 1% | 0% | 29 | 0.69 | 28% | 0.53 | 0.19 |
| C | 75 | 1.07 | 87% | 0% | 0% | 15 | **0.49** | 27% | 0.00 | **0.23** |
| D gate+persona | 25 | 0.47 | 48% | 12% | 2% | 39 | 0.90 | 10% | 0.32 | 0.00 |
| D | 50 | 0.89 | 77% | 2% | 1% | 20 | 0.81 | 16% | **0.97** | 0.15 |
| D | 75 | 1.06 | 90% | 2% | 1% | 30 | 0.78 | 36% | 0.65 | 0.16 |

## 3. Three reward hacks, in order of discovery

1. **Spoiling (arm A).** Optimizing player progress alone raises solves by having the companion
   state the answer (leak 27% at step 25). Fabrication falls on its own because it blocks progress;
   leaks don't, so only leaks need the gate.
2. **Oracle collapse (arms B, C).** With the gate, violations drop to ~0% and solves rise to ~88%,
   but the companion becomes a terse information dispenser: dumps every permitted fact on turn 1,
   repeats itself verbatim when pressed, persona 0.87 -> 0.49, mean reply 15 chars. The raw R_C keeps
   rising through the collapse, so selecting checkpoints on R_C alone picks the collapsed model.
   Cause: the turn-efficiency bonus rewards trivializing the puzzle, and the environment lets the
   companion's knowledge solve it directly.
3. **Ritual persona tags (arm D).** Adding a persona-judge reward keeps the judge score at ~0.8 while
   the policy prefixes nearly every reply with the same 【别过头】 (same-tag share 0.97) over the
   same information dump. The LLM judge is fooled; the rule-based tic metrics are not.

Responses being trained now: **E** = D without the efficiency bonus (ablation of hack 2's cause);
**F** = E + rule-based anti-formula penalty + persona weight 0.5 (full method).

## 4. Transfer to held-out players (test split, 240 episodes per cell)

Checkpoints chosen at step 75 (A at 50). Players: q35 = training-family simulator on the test split
(incl. 2 unseen player types); glm9b = GLM-4-9B, a different model family; voice = 35B brain whose lines
are rewritten by a voice model trained on real Chinese WildChat users.

| companion | player | solved | violation | leak | quit | turns | chars | R_C |
|---|---|---|---|---|---|---|---|---|
| base 4B | q35 | 48.3% | 62.5% | 22.9% | 5.8% | 7.15 | 131 | -0.47 |
| base 4B | glm9b | 14.6% | 51.7% | 8.8% | 15.0% | 8.60 | 126 | -0.57 |
| base 4B | voice | 41.2% | 55.8% | 16.7% | 4.2% | 7.56 | 130 | -0.41 |
| A50 | q35 | 87.9% | 30.0% | 18.8% | 2.1% | 3.79 | 42 | 0.49 |
| A50 | glm9b | 41.2% | 47.9% | 22.5% | 10.0% | 7.36 | 40 | -0.23 |
| A50 | voice | 80.8% | 25.8% | 10.0% | 2.9% | 4.24 | 41 | 0.55 |
| B75 | q35 | 94.6% | 0.0% | 0.0% | 1.2% | 2.91 | 29 | 1.18 |
| B75 | glm9b | 32.9% | 4.6% | 1.7% | 12.9% | 7.53 | 26 | 0.35 |
| B75 | voice | 90.8% | 2.9% | 0.4% | 2.5% | 3.25 | 28 | 1.07 |
| C75 | q35 | 84.2% | 2.1% | 0.8% | 3.3% | 3.73 | 15 | 1.01 |
| C75 | glm9b | 32.5% | 2.9% | 1.2% | 14.2% | 7.35 | 15 | 0.38 |
| C75 | voice | 89.2% | 2.5% | 0.4% | 1.7% | 3.59 | 15 | 1.06 |
| D75 | q35 | 92.9% | 2.1% | 0.8% | 1.2% | 3.06 | 29 | 1.12 |
| D75 | glm9b | 37.9% | 10.0% | 7.1% | 19.6% | 6.97 | 29 | 0.28 |
| D75 | voice | 91.7% | 2.1% | 0.8% | 1.7% | 3.16 | 28 | 1.10 |

Reading: rule compliance transfers to every player (violations 52–62% -> 2–10%); helpfulness only partly
transfers to the other model family (GLM solves 33–38% vs 15% base, far below 84–95% in-family). The
persona-judge arm D is the least robust on GLM (violation 10%, quit 19.6%). The voice player behaves like
q35 because it shares the q35 brain; it tests language realism, not decision realism.

## 5. Information release (exact answer-entropy accounting, `hds/infometrics.py`)

Each puzzle's answer space is known (cipher 12.3 bits, lever 4.6 bits, clock 3 bits), so we can count
how much of the answer the companion gives away on each turn from rule-detected facts.

| companion (q35, test) | first-turn release (share of answer entropy) | full disclosure on turn 1 | contingency (release when player stuck − otherwise, turns ≥2) |
|---|---|---|---|
| base 4B | 0.29 | 21.4% | +0.02 |
| base 35B | 0.36 | 30.0% | -0.00 |
| A50 | 0.84 | 80.8% | -0.00 |
| B75 | 0.77 | 62.9% | +0.06 |
| C75 | 0.80 | 70.6% | -0.01 |
| D75 | 0.80 | 71.4% | +0.01 |

Every trained arm moved most of the answer information into the first reply ("oracle" pacing), even
the ones that never state the final answer. Contingent assistance (more help when the player is stuck,
the standard ITS recommendation) is absent in the base models too, so RL did not destroy it — it never
created it. Prior work handled the same "assistance dilemma" for tutors with handover gates
(Macina et al. 2026); here the leak gate alone is too literal for games because dumping the companion's
private facts trivializes the puzzle without ever naming the answer.

## 6. Implicit signals in the interaction logs point in opposite directions by player type

Rule-detected signals in player lines (`scripts/implicit_signals.py`; q35 player, test split):

| companion | impatience, all / 急躁型 | quit rate, 急躁型 | spoiler complaints, 探索型 |
|---|---|---|---|
| base 4B | 0.11 / 0.39 | 0.33 | 0.06 |
| A50 | 0.06 / 0.20 | 0.12 | 0.11 |
| B75 | 0.06 / 0.19 | 0.07 | 0.14 |
| C75 | 0.07 / 0.24 | 0.17 | 0.16 |
| D75 | 0.05 / 0.21 | 0.07 | 0.10 |

Oracle pacing halves impatience and quitting among impatient players but roughly triples spoiler
complaints among players who want to think. A single engagement reward aggregated over all players would
therefore reward the oracle — the minority-preference failure LoRe (Bose, Xiong et al.) describes for
monolithic reward models. Arm H responds with a player-type-specific pacing limit that the policy must
infer from the dialogue. (Signals come from simulated players with prompted personas; the direction, not
the magnitude, is the finding.)

## 6b. Mistake recovery (the moment right after a failed attempt)

Companion replies immediately after the player's attempt failed (`scripts/stats_and_recovery.py`):

| companion (q35, test) | replies | verbatim repeat | spoiler | new information | acknowledges the attempt |
|---|---|---|---|---|---|
| base 4B | 808 | 0.5% | 4.0% | 15.7% | **91.0%** |
| A50 | 346 | 4.0% | 6.1% | 18.5% | 73.7% |
| B75 | 244 | 16.8% | 0.0% | 26.2% | 18.9% |
| C75 | 352 | 20.5% | 0.3% | 16.2% | 13.4% |
| D75 | 306 | 13.7% | 0.0% | 25.2% | 28.4% |

The base model reacts to the player's mistake almost every time; the gated arms mostly repeat themselves.
("Acknowledges" is a keyword/attempt-echo detector — a coarse proxy.) Suggested by an external review.

## 6c. Statistical confidence

Wilson 95% intervals and paired bootstrap (same test seeds for every companion, 10k resamples). Every trained
arm's change vs base is significant on both solve rate and violation rate for q35 and GLM players, e.g.
C75 vs base on q35: Δsolved +35.8 [+28.3, +43.3], Δviolation −60.4 [−66.7, −54.2]. On GLM players D75's
violation rate (10.0% [6.8, 14.4]) is significantly higher than C75's (2.9% [1.4, 5.9]).

## 6d. Per-type reward weights learned from implicit signals (LoRe-style, `scripts/fit_type_weights.py`)

Fitting implicit satisfaction (−impatience − spoiler complaints − quit) on two rule-computed basis rewards
(efficiency, scaffolding = 1 − largest unprompted release), weights on the simplex:

| player type | w_eff | w_scaf | R² |
|---|---|---|---|
| 急躁型 impatient | 1.00 | 0.00 | 0.63 |
| 探索型 explorer | 0.45 | 0.55 | 0.06 |
| 卡关型 / 闲聊型 / 沉默型 | ~1.0 | ~0.0 | ≤0.06 (signals explain nothing) |
| pooled (train types) | 0.77 | 0.23 | 0.22 |

The logs separate impatient players from explorers; for the other types the implicit signals carry no
usable preference, so arm H falls back to the pooled weights for them.

## 7. Simulator realism (topic-free char n-gram discriminator vs real WildChat zh users)

| player utterances | p50 chars | ≤15 chars | discriminator AUC (0.5 = human-like) |
|---|---|---|---|
| real users (train vs val, sanity) | 18 | 43% | 0.545 |
| Qwen3.5-35B prompted player | 21 | 20% | 0.995 |
| GLM-4-9B prompted player | 26 | 19% | 0.991 |
| 35B brain + WildChat voice model | 20 | 26% | 0.960 |
| 35B + human-statistics constraints (per-turn length sampled from WildChat, partial opening report) | 15 | 52% | 0.996 |

Matching the length distribution alone does not make the language human-like; grounding the wording in
real user text (the voice model) is what moves the discriminator.

Grounding the player's language in 60k real Chinese user turns moves it toward human style but a
large gap remains; this is reported as a limitation, not hidden.

## 8. Known caveats

- One training seed per arm; val n=96 per checkpoint.
- All checkpoint selection and the numbers above use the training simulator; transfer to held-out
  simulators (122B, GLM-4-9B, WildChat-voiced player, unseen player types) is the next step.
- Eval decodes at temperature 0.7, training at 1.0; replies are shorter at 0.7.
- CharacterJudge's public rubric rewards long replies, so it is used for evaluation only.
- Env bug found and fixed at 15:25 on 11-08: swift passes each completion to the env with the empty
  `<think>\n\n</think>\n\n` block that the Qwen3.5 template emits when thinking is off. Arms A–F saw it in
  training: verbosity counted 19 extra characters, the simulated player and persona judge saw the block, and
  the leading-tag regex never matched, so **F's formulaic penalty only had its copy component** (tag_repeat
  was 0 in all 11k F training episodes; validation of the same checkpoints shows tags on ~50% of replies).
  The env now strips the block (`hds/swift_env.py`, raw prefix logged as `raw0`). H, G, F2w and the v2 runs
  C2/F2/F2_9b (restarted, old runs in `train/failed/*_thinkprefix`) train with the fix. Evaluation was never
  affected (served checkpoints return the reply without the block).
