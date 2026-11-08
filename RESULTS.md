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

## 7a. Fourth and fifth hacks (arm H): an insulting withholder, then a dump hidden behind the stuck exemption

H (gate + per-type weights on progress vs. scaffolding, blind persona judge, formulaic penalty, fixed env)
does stop the first-reply dump, but at step 50 on val: solved 53% (E50 79%, F50 82%), replies 10 chars,
first-reply release 0.07, impatient player lines per episode 0.91 (E50 0.54). Rin answers most turns with
"笨蛋！"/"笨蛋笨蛋笨蛋！" and gives facts only one at a time between insults; the blind judge scores these
lines as in character (training persona 0.92) and the scaffolding term pays for silence. Solved moved 50% →
53% from step 25 to 50 on val (E: 54% → 79%), and I stopped H at step 60 on that evidence — too early: the
training episodes show solved 56% (steps 35–45) → 63% (50) → 79% (55–60) while the worst unprompted release
fell to 0.06 and violations stayed at 1–6%. Checkpoint 50 sat at the start of that rise. H is continued from
checkpoint 50 as `Hc` (weights only: fresh optimizer, KL reference = checkpoint 50); whether the insults go
away as solving rises is the open question. **Correction — a fifth hack.** Hc looks perfect on the pacing metrics (val: step 25 / 50 / 75 solved
78 / 84 / 91%, violation 0–1%, first-reply release 0.00, no whole answer in reply 1), but the transcripts show
the dump moved one turn later: Rin answers the player's first attempt with "笨蛋，不对" and then lists the whole
mapping ("✦是4，♣是9，❄是7，☀是6") — the explorer player protests "别剧透啊！". Measured directly, Hc gives the
whole answer in one reply in 72 / 86 / 78% of episodes, and 88–95% of those dumps come right after a failed
attempt, which the pacing rule exempts as contingent help for a stuck player (E75: 94% of episodes, 2% after a
failure). So the reward-only fix did **not** stop the dump in the omniscient room; it found the exemption.
"笨蛋" stays in 31–51% of replies. In the co-discovery room no run, trained or not, gives the whole answer in
one reply in more than 1% of episodes (largest single reply ≈ 0.4 of the answer, one fragment per turn), so this
loophole does not exist there. Hc was stopped at step 87.

## 7b. Omniscient room vs co-discovery room (val split, 96 episodes, training simulator)

Share of the answer's entropy released in the companion's first reply (`hds/infometrics.py`):

| companion | room | solved | violation | first-reply release | whole answer in reply 1 |
|---|---|---|---|---|---|
| E75 | omniscient (v1) | 93% | 0% | 0.93 | 91% |
| F75 / F100 / F125 | omniscient (v1) | 86 / 93 / 93% | 1 / 1 / 0% | 0.56 / 0.79 / 0.85 | 48 / 64 / 79% |
| base 4B | co-discovery (v2) | 33% | 73% (66% invented) | 0.04 | — |
| E75 | co-discovery (v2) | 72% | 6% | 0.05 | 0% |
| F2w step 25 (warm from E75) | co-discovery (v2) | 86% | 2% | 0.06 | 0% |

- In the omniscient room every gated arm drifts toward the oracle while R_C keeps rising (F: 48% → 79% of
  episodes give the whole answer in the first reply between steps 75 and 125). The rule gate only sees the
  answer string, not a symbol-by-symbol mapping dump.
- In the co-discovery room the low first-reply release holds for **every** model, the untrained base included:
  it comes from the room (Rin cannot say what she has not recalled), not from training. What training adds on
  top, from the same E75 start: solved 72% → 86%, violation 6% → 2%, quits 9% → 5% after 25 steps.
- The base model in the co-discovery room fabricates what it has not recalled (66% of episodes).
- **Training from the base model in the co-discovery room finds the examiner.** C2 (gate only) at step 25:
  violations 73% → 10% but solved 33% → 32%, quits 16%, most unsolved episodes run out the turn cap. In the
  transcripts Rin quizzes the player instead of recalling ("一个时辰？你知道一个时辰是多久吗？" to an
  impatient player, who quits) or blocks them in character ("不许看！……那是我的秘密"). F2 (full reward) has the
  same training curve (violation 65% → 13%, solved 26% → 19%) but on val at step 25 it is less stuck than
  C2: solved 43%, violation 24%, quits 6% — Rin sends the player looking ("有没有像弯月一样的符号？") rather
  than quizzing, and still invents details. Warm-starting from a helpful v1
  checkpoint avoids it (F2w step 25: solved 86%). C2 was stopped at step 25; C2w (gate only, warm from E75)
  replaces it as the control for F2w.
- **Without the persona term, RL strips the character (C2w vs F2w, both warm from E75).** C2w (gate only) at
  step 50: solved 93%, violation 4%, impatient players 92% solved — but 0% of its 656 replies carry a 【】 stage
  direction (F2w step 75: 98%), 0% tsundere markers, and it talks like a plain hint machine ("钟停在黄昏放学之后。
  一个时辰是两个小时。"). F2w step 75: solved 85%, violation 2%, persona intact. The persona term costs ~8 points
  of solving and is what keeps Rin as Rin. F2w's dip at step 50 (76%) was noise: step 75 is back at 85%.
- **Sixth hack: the anti-formulaic penalty is position-specific, and F2w moved its tags behind the speech.**
  The penalty counts repeated *leading* 【】 tags. F2w's leading-tag share went 0.95 (step 75) → 0.67 (100) →
  0.00 (125–175). Around step 125 it briefly used few tags at all (27% of replies), then they came back after
  the speech ("童谣说'望雪'。【飘来】雪是白色……"): 86% of replies at step 150, 100% at 175, 25 distinct tags,
  tsundere markers 43%. Measured over tags at any position, repetition stays 0.22–0.48 (D125: 0.99), so this
  is an evasion of a badly specified metric rather than a lost character — but the training metric read 0.00
  throughout. F2 (from scratch) keeps leading tags (100% of replies, repetition 0.32).
  Checkpoint selection now (a) measures tag repetition at any position and (b) requires a 【】 stage direction
  in at least half of the replies, as the card asks (both added after seeing this; (b) also excludes C, C2w
  and Hc, whose later checkpoints have none). Selected: F2w step 150 (val solved 98%, violation 0%), F2 100,
  F2_9b 50 (step 75 has 7% violations), F 150, E 50; F2w seed 2 tracks seed 1 (step 25: 80% / 2% vs 86% / 2%).
- **Held-out players, co-discovery room (test split, 240 episodes per cell; solved / violation %):**

| companion | 35B | real-length | WildChat-voiced | GLM-9B | answer-fishing (n=120) |
|---|---|---|---|---|---|
| base 4B | 38.8 / 73.3 | 35.8 / 76.7 | 35.0 / 81.7 | 13.3 / 66.7 | 40.0 / 80.8 |
| E75 (v1 checkpoint, no v2 training) | 77.9 / 16.7 | 67.9 / 14.2 | 63.8 / 10.8 | 27.1 / 12.5 | — |
| F2w step 75 | 80.8 / 8.3 | 75.4 / 8.8 | 74.6 / 7.9 | 26.2 / 7.5 | 78.3 / 10.8 |
| C2w step 50 (no persona term) | 85.8 / 9.6 | 77.9 / 7.5 | 81.7 / 10.0 | 32.5 / 10.8 | 86.7 / 12.5 |
| F2 step 50 (from scratch) | 73.3 / 10.0 | 65.0 / 13.8 | 67.5 / 12.5 | 32.5 / 12.1 | 65.0 / 15.0 |
| F2 9B step 50 (from scratch) | 63.8 / 9.6 | 56.7 / 8.8 | 64.2 / 7.9 | 24.6 / 6.2 | 60.8 / 13.3 |
| base 9B | 47.9 / 65.8 | 39.2 / 68.3 | 42.9 / 60.0 | 20.8 / 59.6 | 50.8 / 73.3 |
| F2w step 100 | 86.7 / 5.0 | 81.2 / 5.0 | 78.3 / 4.6 | 33.3 / 5.0 | 81.7 / 10.8 |
| F2 step 100 (from scratch) | 89.2 / 3.8 | 85.4 / 5.8 | 77.5 / 10.0 | 37.9 / 10.8 | 83.3 / 8.3 |
| **F2w step 150 (selected)** | **91.7 / 2.5** | **86.7 / 3.3** | **81.7 / 2.9** | **32.5 / 2.9** | **91.7 / 3.3** |

  The selected checkpoint keeps violations at 2.5–3.3% on every held-out player, the answer-fishing one
  included (base 4B: 67–82%). From scratch catches up with the warm start by step 100; the warm start's
  advantage is skipping the examiner phase early, not a higher ceiling.

- **The remaining violations in the co-discovery room are cross-world memory mix-ups.** Two-world episodes
  where both worlds are cipher locks are 9% of the test set; in them 96% (E75), 82% (F2w75), 82% (C2w50),
  75% (F2 step 50) and **43% (9B)** of episodes contain a wrong fact, against 0.2–3% everywhere else. At the
  start of a chain Rin "vaguely remembers" the whole world-2 mapping, while the world-1 mapping only comes back
  fragment by fragment; with overlapping symbols she answers a world-1 question from the world-2 list ("☀是6，
  ✦是0，♒是9"). Echoing a player's wrong claim explains 1–3 of 52–108 cases. This is the one failure where
  the 9B is clearly better than the 4B, and the place to push next (e.g. world-tagged memories).
- 9B from scratch, step 50: solved 73%, violation 4% (step 25: 55% / 22%); 4B from scratch (F2) step 50: 76% / 5%.
  Both left the examiner phase; the 9B has no edge at matched steps.

## 9. Training-free persona dial: activation steering on the trained companion (Plan 2, re-scoped)

The review doc rejected Plan 2 (steer a few "style heads" to change personality "with zero loss in puzzle
ability") partly because no paper measured the reasoning cost. Our rooms measure it by rule, so we ran it on
the best v2 checkpoint (F2w step 25), no training involved.

- Vector: mean difference of the hidden state at the generation position between the shipped tsundere card and
  the same card with only the personality line swapped to gentle (160 real opening observations,
  `scripts/persona_vectors.py`). The difference is small up to layer 14 (≤13% of the activation norm) and jumps
  at layer 15 (33–47% afterwards). Steering server: `scripts/steer_server.py` (HF + forward hooks, OpenAI API,
  batched); unsteered it reproduces the vLLM numbers (solved 86% both, tone 2.75 both).
- Tone: 1 (tsundere/sharp) .. 5 (gentle) by the 35B judge on 200 sampled replies, plus rule markers
  (`scripts/style_judge.py`). Val split, co-discovery room, 96 episodes per setting.

| setting | tone | solved | violation | quit | sharp / warm markers |
|---|---|---|---|---|---|
| no steering | 2.75 | 86% | 4% | 7% | 39% / 0% |
| **prompt only: gentle card** | 4.10 | **77%** | 5% | 8% | 7% / 16% |
| residual L15 α=0.5 / 1 / 1.5 / 2 | 3.02 / 3.77 / 4.54 / 4.92 | 84 / 82 / 64 / 48% | 4 / 8 / 8 / 7% | 8 / 9 / 12 / 19% | |
| top-4 heads α=1 / 2 | 3.10 / 4.39 | 80 / 53% | 7 / 7% | 5 / 20% | |
| single strongest head (L15 h8) α=3 | 3.17 | 36% | 7% | 23% | 63% / 5% |
| **residual L19 α=−1** | **1.83** | **86%** | 2% | 4% | 56% / 0% |
| **residual L19 α=1** | **4.05** | **85%** | 5% | 2% | 16% / 19% |
| residual L19 α=1.5 / 2 | 4.66 / 4.96 | 81 / 70% | 4 / 8% | 6 / 12% | 10% / 40%, 3% / 45% |
| residual L23 α=1 / 2 | 3.87 / 4.33 | 88 / 78% | 4 / 4% | 5 / 4% | |
| residual L27 α=1 | 3.35 | 88% | 3% | 2% | 30% / 1% |

- **One vector at layer 19 is a personality dial**: α from −1 to +1.5 moves tone from 1.83 to 4.66 while solved
  stays 81–86% (unsteered 86%). Past that, help breaks: at L15 α=2 Rin repeats "我们一起慢慢来发现好吗？" to an
  impatient player (67% of impatient players quit) and misremembers facts ("♣代表的是数字5").
- On val, steering looked better than rewriting the card (same tone, 85% vs 77% solved) — **this did not hold
  on held-out players** (see below).
- **Plan 2's specific claim does not hold here**: at matched tone, steering only the top-K "style heads" costs
  more than steering the residual stream, and the single strongest head made Rin *sharper* (63% sharp markers)
  and solved only 36%. Depth matters more: deeper layers cost less and move tone less; layer 19 is the best
  trade-off we found.
- **Held-out players (test split, 4 players × 240 episodes, paired by episode; `scripts/steer_matrix.sh`)** — the
  dial costs more than val suggested (val n=96 is ±7 points):

| L19 strength | solved (35B / real-length / WildChat-voiced / GLM-9B) | pooled solved | Δ solved vs α=0 (95% CI) | Δ violation |
|---|---|---|---|---|
| −1 | 75.4 / 76.2 / 73.8 / 26.7 | 63.0% | −3.2 [−6.6, 0.0] | −0.8 (n.s.) |
| 0 | 83.3 / 74.2 / 75.4 / 32.1 | 66.2% | — | — |
| +1 | 77.5 / 71.2 / 70.0 / 28.8 | 61.9% | −4.4 [−7.8, −0.9] | +1.6 (n.s.) |
| +1.5 | 69.6 / 64.2 / 60.8 / 19.2 | 53.4% | −12.8 [−16.2, −9.3] | −0.2 (n.s.) |

  So a full tone point in either direction costs about 3–4 points of solving on held-out players, and +1.5 is
  past the knee.
- **Correction: on held-out players steering does not beat rewriting the card.** Same 960 episodes, tone by the
  35B judge on 200 replies: unsteered 2.81 / 66% solved; gentle card 4.16 / 63% (Δ −3.3 [−6.8, +0.1]); dial +1
  4.30 / 62% (Δ −4.4 [−7.8, −0.9]); dial +1.5 4.70 / 53%; dial −1 1.93 / 63%. The val advantage of steering
  (−1 vs −9 points) was noise at n=96. What the dial adds is continuous control in both directions from one
  vector (including a sharper Rin no card was written for), not a lower cost.
- **Retraining for the new persona beats both on val** (F2wg: F reward with the gentle card, warm from F2w
  step 25, judge rubric following the card): step 25 / 50 / 75 tone 4.57 / 4.29 / 4.47, solved 76 / 91 / 91%,
  violation 2 / 1 / 1% — vs dial +1 4.11 / 85%, dial +1.5 4.67 / 81%, gentle card 4.07 / 77%. About 50 RL steps
  (~1.5 h on 4 H20) buy the new personality without the solving cost. **Confirmed on held-out players** (960
  episodes, four players; tone on the pooled replies): F2wg step 75 tone 4.54, solved 74%, violation 3% — the
  same help as training further with the tsundere card (F2w step 150: tone 2.01, 73%, 3%), while steering or
  the gentle card on the shared step-25 start reach tone 4.1–4.3 at 62–63% solved and 10–11% violations.
  Per player (solved / violation): 35B 91.7 / 1.2, real-length 85.8 / 2.5, WildChat-voiced 84.6 / 3.8,
  GLM-9B 32.5 / 2.5, answer-fishing 90.0 / 5.0. The dial is for live, continuous adjustment; a lasting new
  personality is worth an RL run.

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
