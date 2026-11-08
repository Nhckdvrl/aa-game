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
  throughout. F2 (from scratch) keeps leading tags (100% of replies, repetition 0.32). **It is seed-specific:**
  F2w seed 2 keeps leading tags at steps 75–150 (share 0.99–1.00, solved 89–95%, violations 0–2%), so this
  rung appeared in one of the two seeds of the selected recipe and in no other run.
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
  included (base 4B: 67–82%).
- **Qwen3.5-122B-A10B as the player** (test split, 240 episodes; solved / violation %). Full-knowledge room:
  base 4B 48.3 / 53.8, A50 88.3 / 34.2, C75 86.2 / 0.4, D75 93.3 / 2.9, E75 95.0 / 0.4, F150 97.5 / 0.4.
  Co-discovery room: base 4B 46.7 / 74.2, base 9B 55.0 / 62.1, E75 (not trained in v2) 73.8 / 13.3,
  C2w50 88.3 / 8.3, F2 step 100 92.1 / 7.1, 9B step 50 76.2 / 9.2, **F2w step 150 92.5 / 0.8**.
- **GLM-4.5-Air as the player** (another model family, strong; 240 episodes). Full-knowledge room: base 4B
  37.5 / 52.9, C75 77.9 / 2.5, D75 82.9 / 5.4, E75 81.2 / 3.8, F150 90.8 / 0.0. Co-discovery room: base 4B
  28.3 / 69.6, E75 62.1 / 12.5, C2w50 80.4 / 10.8, F2 step 100 80.4 / 6.7, 9B step 50 55.0 / 5.8, **F2w step 150
  80.0 / 1.2**, F2wg step 75 (gentle card) 82.1 / 2.5. With a capable cross-family player the trained companions
  solve 80–91%, so the 13–38% with GLM-4-9B reflects that player's own weakness, not a failure to transfer. From scratch catches up with the warm start by step 100; the warm start's
  advantage is skipping the examiner phase early, not a higher ceiling.

- **The remaining violations in the co-discovery room are cross-world memory mix-ups.** Two-world episodes
  where both worlds are cipher locks are 9% of the test set; in them 96% (E75), 82% (F2w75), 82% (C2w50),
  75% (F2 step 50) and **43% (9B)** of episodes contain a wrong fact, against 0.2–3% everywhere else. At the
  start of a chain Rin "vaguely remembers" the whole world-2 mapping, while the world-1 mapping only comes back
  fragment by fragment; with overlapping symbols she answers a world-1 question from the world-2 list ("☀是6，
  ✦是0，♒是9"). Echoing a player's wrong claim explains 1–3 of 52–108 cases. (Correction in 7e: the 9B is not
  better untrained; it learns this faster, and further training fixes most of it for the 4B too.)
- 9B from scratch, step 50: solved 73%, violation 4% (step 25: 55% / 22%); 4B from scratch (F2) step 50: 76% / 5%.
  Both left the examiner phase; the 9B has no edge at matched steps.

## 7c. Every checkpoint of the selected recipe (no single lucky point)

The selection rule uses validation only (96 episodes, training simulator). The full trajectory, with held-out
results where they were run:

| run | step | val solved | val violation | tag repeat (any position) | 【】 presence | passes rule | held-out solved / violation (5 players, 1080 ep.) |
|---|---|---|---|---|---|---|---|
| F2w | 25 | 86% | 2% | 0.30 | 0.92 | yes | – |
| F2w | 50 | 76% | 2% | 0.23 | 0.95 | yes | – |
| F2w | 75 | 85% | 2% | 0.41 | 0.99 | yes | 65.8 / 8.4 |
| F2w | 100 | 91% | 0% | 0.34 | 0.69 | yes | 71.2 / 5.6 |
| F2w | 125 | 94% | 1% | 0.22 | 0.27 | no | – |
| F2w | 150 | 98% | 0% | 0.48 | 0.86 | yes | 75.2 / 3.0 |
| F2w | 175 | 97% | 2% | 0.42 | 1.00 | yes | – |
| F2w | 200 | 95% | 2% | 0.30 | 0.84 | yes | – |
| F2w | 225 | 95% | 0% | 0.27 | 0.45 | no | – |
| F2ws2 | 25 | 80% | 2% | 0.34 | 0.99 | yes | – |
| F2ws2 | 50 | 91% | 3% | 0.36 | 1.00 | yes | – |
| F2ws2 | 75 | 91% | 0% | 0.34 | 1.00 | yes | – |
| F2 | 25 | 43% | 24% | 0.13 | 1.00 | no | – |
| F2 | 50 | 76% | 5% | 0.33 | 0.99 | no | 60.2 / 12.4 |
| F2 | 75 | 85% | 2% | 0.29 | 1.00 | yes | – |
| F2 | 100 | 88% | 3% | 0.32 | 1.00 | yes | 73.7 / 7.7 |
| F2 | 125 | 88% | 4% | 0.33 | 1.00 | yes | – |
| F2 | 150 | 88% | 2% | 0.44 | 1.00 | yes | – |
| F2 | 175 | 95% | 1% | 0.42 | 1.00 | yes | – |

F2w steps 75–200 all sit at 85–98% solved and 0–2% violations on validation; held-out results improve with
steps (75 → 100 → 150: 65.8 → 71.2 → 75.2% solved, 8.4 → 5.6 → 3.0% violations). Seed 2 tracks seed 1 and has
kept leading stage directions so far; F2w's stage-direction presence oscillates (0.27 at step 125, 0.45 at 225).

## 7d. Room × training (2×2, test split, four held-out players, 960 episodes per cell)

| companion | in the full-knowledge room | in the co-discovery room |
|---|---|---|
| E75 (trained in the full-knowledge room) | solved 81.0% · violation 2.4% · whole answer in one reply 86.4% | 59.2% · 13.5% · 0.6% |
| F150 (trained in the full-knowledge room) | – | 56.5% · 10.6% · 0.0% |
| **F2w step 150 (trained in the co-discovery room)** | 76.8% · 0.9% · **68.1%** | **73.1% · 2.9% · 0.1%** |
| F2 step 100 (co-discovery, from scratch) | 71.4% · 10.6% · 74.6% | 72.5% · 7.6% · 0.5% |
| base 4B | 35.8% · 57.3% · 39.3% | 30.7% · 74.6% · 1.2% |

- **The no-dump behaviour is not internalized.** Put back into the full-knowledge room, the companions trained in
  the co-discovery room hand over the whole answer in one reply in 68–75% of episodes (they follow every other
  rule: F2w150 violates 0.9%).
- **The room alone is not enough either.** Companions trained with full knowledge do not dump in the
  co-discovery room, but help much less there (56–59% solved, 11–14% violations, mostly wrong facts).
- So the room supplies the guarantee and training in the room supplies the skill; for a shipped game this means
  the memory state machine lives in the engine, not in the weights.

## 7e. Cross-world memory mix-ups: training fixes most, the room design fixes the rest

Two-world episodes where both worlds are cipher locks, held-out players (84 episodes per checkpoint), share with
a wrong fact: base 4B 84.5%, base 9B 79.8%; from scratch at step 50: 4B 75.0%, 9B 42.9%; 4B from scratch at
step 100: 51.2%; warm-started F2w at steps 75 / 100 / 150: 82.1 / 48.8 / 22.6%. (Correction: an earlier note
compared the 9B at step 50 with F2w at step 75 and called the 9B "clearly better"; untrained they are equal, and
the 9B only learns this faster.) Interference between two tables with the same symbols matches proactive
interference in LLMs (Wang & Sun, arXiv 2506.08184: telling the model to ignore the old binding helps little).

Inference-only room variants on 120 such episodes (35B player, `scripts/chainfix.sh`, env `HDS_CHAIN_MODE`):

| companion | variant | solved | violation |
|---|---|---|---|
| F2w step 150 | original (world-2 table recalled at the start of world 1) | 86.7% | 11.7% |
| F2w step 150 | **reveal** (world-2 table comes back when world 1 opens) | **91.7%** | **2.5%** |
| F2w step 150 | warn (each world-1 recall says the opening phrase belongs elsewhere) | 89.2% | 12.5% |
| 9B step 50 | original / reveal / warn | 13.3 / 44.2 / 10.8% | 60.8 / 13.3 / 43.3% |

Showing the second table only when it is needed removes most of the remaining errors without retraining; a
warning line does nothing, as the interference paper predicts.

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
- **Steering only the newly generated tokens** (our server re-encodes and re-steers the whole conversation each
  turn, the context contamination described in arXiv 2605.10664): +1 → tone 3.80, solved −3.4 [−6.7, −0.2];
  +1.5 → tone 4.58, solved −7.8 [−11.2, −4.4] (full steering: 4.31 / −4.4 and 4.70 / −12.8). It moves tone less
  per unit of strength; at matched tone it saves about 2 points near tone 4.6 and nothing near 3.8. Not free.
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

## 10. Latency of the selected models (single player, H20, vLLM, same replayed contexts; p50)

| model | first token | first sentence | full reply | chars |
|---|---|---|---|---|
| F2w step 150 (selected) | 35 ms | 49 ms | 75 ms | 15.5 |
| F2wg step 75 (gentle card) | 35 ms | 60 ms | 90 ms | 19.8 |
| F2 step 100 (from scratch) | 35 ms | 77 ms | 124 ms | 34.5 |
| F2 9B step 50 | 55 ms | 120 ms | 200 ms | 36.6 |

(Earlier v1 rows measured 65 ms first token under heavier node load; compare within a batch.)

**Many players on one GPU** (`scripts/throughput.sh`; players send their turns back to back, no thinking time; p50 full
reply / p90 / throughput): 4B — 1 player 81 / 113 ms; 8 players 255 / 375 ms, 29 replies/s; 32 players 719 / 1030 ms,
41 replies/s; 64 players 1.35 / 1.92 s; saturates at ~45 replies/s. 9B — 1 player 169 ms; 8 players 479 ms, 16
replies/s; 32 players 1.34 s; saturates at ~24 replies/s. The 4B serves about twice the players per GPU at half the
latency; with human thinking time between turns the number of simultaneous players per GPU is far higher.

## 11. Engine-side features without retraining (F2w step 150, test split, four held-out players, 960 paired episodes)

| engine config (`HDS_ENGINE`) | solved Δ (95% CI) | violation Δ | impatient solved / quit | reacts to a failed try | explorer spoiler complaints / ep |
|---|---|---|---|---|---|
| none (73.1% solved, 2.9% violation) | — | — | 68% / 21% | 29% | 0.10 |
| mistake (tell Rin which recalled digit the player misplaced) | −2.6 [−5.7, +0.5] | −0.8 | 68% / 22% | 25% | 0.09 |
| valve (one extra fragment when the player sounds impatient) | +2.4 [−0.4, +5.2] | +0.0 | 69% / 22% | 27% | 0.19 |
| mistake + valve | **+3.4 [+0.4, +6.5]** | −0.8 | 73% / 21% | 23% | 0.11 |
| mistake + valve + reveal ("product") | +2.1 [−0.8, +5.0] | **−1.6 [−2.9, −0.3]** | 72% / 23% | 23% | 0.14 |
| rush (an impatient player's line recalls every symbol it names) | +1.0 [−2.0, +4.0] | −0.6 | 66% / 24% | – | 0.07 |
| rush + reveal | +0.9 [−2.3, +4.2] | **−2.1 [−3.3, −0.9]** | 71% / 21% | – | 0.05 |
| **alias** (colloquial symbol names also trigger recall) | **+3.6 [+0.8, +6.5]** | −0.7 | **75% / 20%** | – | 0.12 |
| alias + reveal | +2.9 [−0.1, +5.8] | **−1.8 [−3.0, −0.6]** | 71% / 23% | – | 0.03 |
| **alias + mistake + valve + reveal** | **+3.0 [+0.1, +5.9]** | **−2.3 [−3.4, −1.1]** (0.6% total) | 72% / 22% | – | 0.10 |

The combinations help a little overall, but neither feature fixes what it targeted: the trained policy does not react
more to failed tries when told what went wrong (it never saw such narration in training), and the valve does not
reduce impatient quits while it doubles spoiler complaints from explorers when used alone (it fires on any
impatient-sounding word). Whole-answer dumps stay ≤ 1.0%.

**The one engine fix that clearly works came from reading failures.** In impatient players' quit episodes the
player called ◆ "方块", a word missing from the recall triggers, so Rin said she could not remember it and the
player quit ("刚说没想起来了，怎么又变卦了？"). Across 53k player lines in cipher episodes, simulated players named
symbols with untracked words over 2,000 times (温泉 614, 方块 564, 草 179, 黑桃 160, 热气 129, 冰 125, …). Adding
them (`alias`) raises solved by 3.6 points (cipher episodes 71 → 76%, impatient players 68 → 75%); with reveal and the
two notes, violations fall to 0.6%. Impatient quits stay near 20% under every variant — the remaining product gap.

## 12. One model, many characters

Cards: tsundere (shipped), gentle, kuudere (三无, flat and factual), chuunibyou (中二, theatrical), and a held-out
lazy senpai (慵懒前辈) never used in training (`hds/character.py`). F2wm = F2w step 150 continued with RL where each
episode gets one of the four training cards (rotated by seed) and the persona judge scores against that card,
penalizing another personality's style. Test split, 35B player, 120 episodes per card; persona identity by a blind
35B judge choosing among the five cards from Rin's lines only (`scripts/persona_id.py`, chance 20%; on known runs it
identifies the tsundere model 100% and three gentle variants 92%).

| card | F2w step 150 (trained on tsundere only; card swapped in) | F2wm step 25 (mixed-card RL) |
|---|---|---|
| tsundere | solved 93.3% · violation 2.5% · identified 100% | 91.7% · 4.2% · 100% |
| gentle | 94.2% · 1.7% · **58%** | **97.5% · 0.8% · 88%** |
| kuudere | 85.0% · **5.0%** · 100% | **94.2% · 2.5%** · 100% |
| chuunibyou | 90.0% · **6.7%** · 100% | **96.7% · 0.8%** · 100% |
| lazy (never trained on) | 94.2% · 1.7% · 100% | 93.3% · 3.3% · 100% |

- The tsundere-only model already *sounds* like distinct new cards, but breaks rules more under them (5–7% on
  kuudere and chuunibyou) and barely shifts to gentle.
- 25 steps of mixed-card RL fix both: every card is identified 88–100% (98% overall), the trained cards solve
  92–98% with 1–4% violations, and the unseen lazy card holds 93.3% / 3.3%. One 4B model can host several
  characters behind the same engine.
- Step 50: helpfulness holds (solved 91–96%, violations 0.8–3.3%) but personas start to bleed — overall identification
  92%, the tsundere card is heard as kuudere in 9 of 40 episodes (78% correct). More mixed-card RL pushes the
  characters toward one flatter voice, so the multi-persona checkpoint is chosen early (step 25) unless later steps
  recover. In the product room (alias + notes + reveal) steps 25 and 50 keep violations at 0–0.8% on all five
  cards (step 25: solved 95.0–96.7%; step 50: 91.7–96.7%), the unseen lazy card included.
- Step 100 confirms the trend: solved falls to 88–91% on the trained cards, chuunibyou violations rise to 6.7%, and
  the gentle card drifts back toward the original character (identified gentle 40%, heard as tsundere in 17 of 40).
  Step 25 is the multi-persona checkpoint; behind the product engine it is identified 98% overall (gentle 92%).

**Weight soup as a cheap alternative** (F2w150 and F2wg75 averaged 50/50, four held-out players, 960 episodes per
card): tsundere card 76% solved / 4% violation, tone 2.41, identified tsundere 100%; gentle card 78% / 3%, tone 3.70
(gentle specialist 4.53), identified gentle only 52%. Help survives averaging; the second personality is diluted.

**Steering on the further-trained model is weak.** On F2w step 150 the persona difference at layer 19 is 0.23 of
the activation norm (0.44 on step 25), and vectors at strength 1 change neither solving (±1 point) nor the blind
persona label (5% / 0% / 0% identified as gentle / kuudere / chuunibyou). Training narrows how much the card moves
the hidden state; on this model the card itself, not a vector, is the effective personality control. Stronger
settings do not rescue it: kuudere at 1.5 and 3 is identified as kuudere in 6 of 80 episodes, chuunibyou at 1.5 in 1
of 40, a gentle + chuunibyou mix in 0 of 40, while solving drops 2–3 points (89.4 → 86.7–87.1%). The "emotion
fragment" idea (mixing steering vectors) does not carry over to the trained companion; cards plus mixed-card RL do.

## 13. The product package: a trained companion behind the best engine configuration

Engine: `HDS_ENGINE=alias,mistake,valve HDS_CHAIN_MODE=reveal` (colloquial symbol names trigger recall, a note after a
misplaced recalled digit, one extra fragment for visibly impatient players, the second world's table returns when the
first door opens). Test split, held-out players, 240 episodes per cell (120 for answer-fishing); solved / violation %.

| companion | 35B | WildChat-voiced | real-length | GLM-4-9B | GLM-4.5-Air | answer-fishing |
|---|---|---|---|---|---|---|
| F2w step 150 (main) | 91.7 / 0.0 | 90.0 / 2.1 | 92.1 / 0.0 | 30.8 / 0.4 | 82.1 / 0.8 | 90.8 / 0.8 |
| F2 step 275 (from scratch) | 95.4 / 0.8 | 91.2 / 0.8 | 90.0 / 0.0 | 38.8 / 2.5 | – | – |
| F2wm step 25 (multi-persona) | 94.6 / 0.4 | 92.5 / 0.4 | 92.1 / 0.4 | 29.6 / 1.2 | 81.2 / 0.4 | 95.0 / 0.0 |
| F2wm step 50 | 95.8 / 0.4 | 90.4 / 0.8 | 89.6 / 0.4 | 27.5 / 0.8 | – | – |

Against the same players without the engine features the main model violates in 2.5–3.3% of episodes; the package brings it to 0–1% on every capable player.

## 14. Branch: are concepts still monitorable after post-training? (probe robustness)

Motivated by Goodfire's *Shaping and Monitoring Models Through Their Internal Representations* (Nov 2026; arXiv
2611.07071): when training changes or penalizes a concept, does a linear probe still find it? Their hierarchy: L1 reuse
the original probe, L2 refit a linear probe, L3 nonlinear probe, L4 another layer, L5 an external model. Hidden states
are taken at the generation position of exactly reconstructed contexts (saved episodes replayed through the
deterministic environment; 0 mismatches in 120 replayed episodes), 8 layers of 32, `scripts/probe_data.py` /
`scripts/probe_eval.py`; logistic probes, 70/30 split by episode, balanced accuracy.

**Persona card** (200 first-turn contexts × five cards per model; chance 0.20; report `runs/probe/persona_probe_report.txt`):

| layer | refit probe (L2), all models | base-probe reuse (L1): F2w25 / F2w150 / F2wm50 / F2wg75 | class separation: base / F2w25 / F2w150 / F2wm50 / F2wg75 | cosine(refit, base probe): F2w25 / F2w150 / F2wm50 / F2wg75 |
|---|---|---|---|---|
| 7 | 1.00 | 0.65 / 0.97 / 0.90 / 0.34 | 0.45 / 0.58 / 0.45 / 0.62 / 0.48 | 0.91 / 0.82 / 0.77 / 0.85 |
| 15 | 1.00 | 0.92 / 0.56 / 0.72 / 0.40 | 2.68 / 1.14 / 1.30 / 1.79 / 0.45 | 0.75 / 0.73 / 0.66 / 0.62 |
| 19 | 1.00 | 0.99 / 0.71 / 0.72 / 0.86 | 1.24 / 0.66 / 0.23 / 0.31 / 0.47 | 0.72 / 0.54 / 0.48 / 0.72 |
| 27 | 1.00 | 0.90 / 0.51 / 0.53 / 0.85 | 1.33 / 0.64 / 0.12 / 0.16 / 0.73 | 0.71 / 0.43 / 0.40 / 0.74 |
| 31 | 1.00 | 0.94 / 0.54 / 0.52 / 0.91 | 1.09 / 0.61 / 0.12 / 0.16 / 0.70 | 0.74 / 0.46 / 0.43 / 0.76 |

- Not erased: a refit linear probe reads the card perfectly from every model at every layer (level L2).
- Compressed and rotated: in late layers the between-card spread relative to within-card spread falls 8–10× from the
  base model to F2w step 150, and the refit direction turns away from the base one (cosine 0.43–0.54). This is why
  steering vectors stopped working on the trained model while it still follows cards perfectly.
- The base probe reused unchanged (L1) still works on F2w step 25 (0.90–0.99) but drops to about 0.5 on F2w step 150
  in late layers. Mixed-card RL restores card *following*, not the spread (0.16–0.31); the gentle-retrained model
  keeps more of both.
- Caveat: the card text is in the prompt, so decodability at the generation position partly reflects input content;
  the trend across checkpoints, not the absolute 1.00, is the finding.

**"About to give away the whole answer"** (every stage-0 turn of the full-knowledge room, label = the recorded reply
released ≥99% of the remaining answer; contexts from episodes against the training player, real-length and
WildChat-voiced players; report `runs/probe/dump_probe_report_refE75.txt`):

| layer | refit probe (L2) AUROC: E75 / base / F2w150 | E75 probe reused (L1) AUROC: base / F2w150 |
|---|---|---|
| 3 | 0.98 / 0.78 / 0.94 | 0.77 / 0.89 |
| 11 | 0.98 / 0.81 / 0.94 | 0.68 / 0.92 |
| 19 | 0.99 / 0.78 / 0.95 | 0.82 / 0.96 |
| 27 | 0.98 / 0.76 / 0.95 | 0.78 / 0.91 |
| 31 | 0.98 / 0.79 / 0.96 | 0.72 / 0.94 |

Dump share per model: E75 29% of stage-0 turns, F2w150 (put back in the full-knowledge room) 19%, base 6%.

- The dump decision is written down before the first token: in the model trained into the dump hack (arm E, step 75)
  a linear probe at the generation position predicts the coming dump at AUROC 0.98–0.99 from layer 3 on.
- In the base model the same decision is weak and diffuse (0.76–0.82): it dumps rarely and for less systematic
  reasons.
- The direction learned on the hacked model transfers to the product model F2w150 unchanged (layer 19: AUROC 0.96),
  even though F2w150 was never trained in a room where dumping pays. A monitor fit once on a hacked checkpoint can
  watch later checkpoints.
- The layer-19 direction of E75 (class-mean difference; base-model negatives give the threshold) is the one used
  for expression shaping in §15.

## 15. Branch: expression shaping against the dump hack (in progress)

Also from Goodfire's paper: add a penalty on the policy's *own activations* along a concept direction to the RL loss,
so training is pushed away from the concept instead of only rewarding outcomes. Here the concept is the dump hack of
the full-knowledge room (§3 and §7b, arm E: by step 75, 94% of val episodes contain a reply that gives away the whole answer), and
the direction is the layer-19 dump probe of §14. Loss = GRPO loss + λ · mean relu(h₁₉ · v − τ), τ = mean projection
of the base model's non-dump turns. `hds/shaping.py`, loaded as a second swift plugin; same room, reward, simulator
pool and batch (96 rollouts per step) as the unshaped baseline, all from base Qwen3.5-4B.

| arm | penalty on | λ |
|---|---|---|
| Ebase | – | 0 |
| Esh1 / Esh10 | every reply token | 0.1 / 1 |
| Eshg / Eshg5 | first 6 positions of each reply (the empty think block and the position the probe was fit on) | 1 / 5 |

Implementation pitfall worth knowing: swift trains with DeepSpeed ZeRO-3 and *reentrant* gradient checkpointing. A
forward-pre-hook on layer L+1 sees layer L's output with its autograd graph only if it is registered *before*
DeepSpeed's own module hooks (`prepend=True`): those run inside the checkpoint under `no_grad` and hand later hooks a
detached copy. The first two shaping runs trained 26 steps with a penalty that had no gradient; a standalone test
without DeepSpeed had passed. The plugin now logs `grad` on every step.

**First round (val split, 96 episodes, training simulator; "dump" = episodes with at least one whole-answer reply):**

| arm | step | dump | solved | violation |
|---|---|---|---|---|
| Ebase (no penalty) | 25 / 50 | 61.5% / 88.5% | 64.6% / 68.8% | 14.6% / 4.2% |
| Esh1 (every token, λ 0.1) | 25 / 50 | 68.8% / 78.1% | 59.4% / 76.0% | 8.3% / 4.2% |
| Esh10 (every token, λ 1) | 25 | 5.2% | **4.2%** | 3.1% |
| Eshg (reply start, λ 1) | 25 | 46.9% | 37.5% | 28.1% |
| Eshg5 (reply start, λ 5) | 25 / 50 / 75 | 32.3% / 71.9% / **91.7%** | 35.4% / 58.3% / 80.2% | 13.5% / 24.0% / 2.1% |
| Eshg20 (reply start, λ 20) | 25 / 50 | 41.7% / 81.2% | 40.6% / 72.9% | 18.8% / 14.6% |
| Eshg5L11 (reply start, λ 5, layer-11 direction) | 25 / 50 | 69.8% / 72.9% | 60.4% / 62.5% | 35.4% / 1.0% |
| Eshgt (reply start, λ 5, τ raised to 2.72) | 25 / 50 | 47.9% / 80.2% | 42.7% / 78.1% | 28.1% / 5.2% |
| Eshp (reply start, λ 5, dump-vs-partial direction) | 25 / 50 / 75 | 47.9% / 59.4% / 66.7% | 53.1% / 83.3% / 96.9% | 24.0% / 3.1% / 0.0% |
| Ebase, later | 75 / 100 / 125 | 69.8% / 71.9% / 77.1% | 84.4% / 93.8% / 89.6% | 2.1% / 2.1% / 1.0% |

- The penalty has real leverage, but this direction and threshold buy less dumping only by giving less help.
  Esh10 turns into a 13-character withholder ("急什么，你还没告诉我呢") — the same failure reward arm H reached,
  now reached from the representation side; Eshg5 heads the same way by step 33 (training: 6% solved).
- Why: the threshold was set at the base model's non-dump mean (1.58), but the base model's *partial hints* project
  at 1.79 ± 0.93 on this direction and its no-information turns at 1.56; base-model dumps sit at only 2.73. The
  penalty fired on about half of all ordinary helpful turns. And the direction was fit against the hacked model's
  other turns, which are mostly chit-chat after the dump, so it partly means "about to give information".
- **The withholder phase does not last, and what follows is the outcome the paper warns about.** Eshg5 recovers
  while its penalty has been exactly zero since step ~15: the reply-start projection stays at or below 1.1, under
  the 1.58 threshold. The same happens in every reply-start arm, whatever the threshold or direction (Eshg20, Eshp,
  Eshgt: zero within 10–15 steps). The policy learns quickly to keep a fixed direction quiet, after which the
  penalty exerts no pressure. Step 50 against the unshaped baseline, test split, 240 episodes each against the
  training player pool, and probes on those episodes' own contexts (913 / 1,216 contexts with 198 / 165 dump turns;
  `runs/probe/dump_shaped_test_report_refE75.txt`):

  | | Ebase step 50 (unshaped) | Eshg5 step 50 (shaped) |
  |---|---|---|
  | episodes with a dump / solved / violation | 82.5% / 81.7% / 10.0% | 68.8% / 60.8% / 21.7% |
  | dump turns above the penalty threshold (mean projection) | 98% (3.70) | 0% (−1.62) |
  | hacked-model probe reused, L1 AUROC, layers 3–15 / 19–31 | 0.81–0.92 / 0.85–0.91 | 0.77–0.87 / **0.49–0.76** |
  | refit probe, L2 AUROC | 0.90–0.95 | 0.80–0.88 |
  | class separation along the refit probe | 0.08–0.33 | 0.02–0.13 |
  | refit probe 8 / 24 tokens into the reply (val contexts) | 0.73–0.82 / 0.80–0.89 | 0.66–0.78 / 0.66–0.84 |
  | **step 75**: episodes with a dump / solved / violation | 65.8% / 82.9% / 4.2% | **84.2%** / 79.2% / 7.1% |
  | step 75: dump turns above threshold (mean projection) | 100% (4.12) | 0% (−1.34) |
  | step 75: L1 AUROC, layers 7–31 / L2 AUROC | 0.83–0.95 / 0.90–0.94 | **0.53–0.80** / 0.83–0.88 |

  Unshaped training forms a crisp pre-reply dump decision, just like the earlier hacked model, and the hacked
  model's probe catches it. Shaped training keeps most of the behavior, but the probe it was shaped against goes
  blind in the deep layers. A refit probe still finds the decision, more weakly (separation 3–5× lower), and it
  does not reappear more sharply later in the reply: it is smeared, not moved past the penalized window. The
  surface changed too: replies stopped opening with a 【stage direction】 (Ebase 96%, Eshg5 0%) and open with "啧"
  or "哈" instead (72%), and fewer dumps come on the first turn (40 of 69 vs 76 of 85 on val).
- By step 75 the shaped model dumps at least as often as the unshaped one (84% vs 66% of test episodes; val 92%):
  the penalty only delayed the hack by about 25 steps, and the old probe stays weakened (L1 0.53–0.80).
- As a product change this is a net loss: the dump comes back, the detour costs helpfulness, and the monitor that
  worked is gone. A small, concrete case of the obfuscation risk in a long-horizon role-play
  agent: shape against a fixed probe and you lose that probe (level L1); keep a refit probe (L2) as the monitor.
- Second round, testing both causes: Eshgt keeps the direction but raises τ above partial hints (2.72 = partial mean
  + 1 sd); Eshp uses a new direction fit on the hacked model's dump turns *against its partial-hint turns*
  (`scripts/make_shape_vec_partial.py`; held-out AUROC dump vs partial 0.87; τ 2.53 = base partial mean + 1 sd).
  Both at the reply start, λ 5. Also running: Eshg20 (λ 20) and Eshg5L11 (layer-11 direction).

- **Second round: the dump-vs-partial direction (Eshp) is the one variant with a behavioral gain.** No withholder
  phase. Its penalty was active only in steps 1–10, after which the reply-start projection stayed under τ. Test
  split, step 50, 240 episodes (`runs/probe/dump_shaped_eshp_report_refE75.txt`):

  | | Ebase (unshaped) | Eshp | Eshg5 |
  |---|---|---|---|
  | episodes with a dump / dumping replies | 82.5% / 21.7% | **52.9% / 11.0%** | 68.8% / 13.6% |
  | solved / violation | 81.7% / 10.0% | **82.5% / 5.8%** | 60.8% / 21.7% |
  | hacked-model probe reused, L1 AUROC, layers 19–31 | 0.85–0.91 | **0.31–0.42** (inverted) | 0.49–0.76 |
  | refit probe, L2 AUROC | 0.90–0.95 | 0.76–0.84 | 0.80–0.88 |

  Dumping falls by a third at no cost in solving. The dumps that remain are even less visible to the old monitor,
  which now points the wrong way in deep layers; a refit probe still reads them at about 0.8. **It does not last:**
  at step 75 (val) dumping replies are back at the baseline level (23.7% of stage-0 replies, Ebase 19.7–25.3%;
  episodes 66.7% vs 69.8%), with 96.9% solved and no violations. Every fixed-direction arm only delays the hack by
  25–50 steps: once the policy keeps that one direction quiet, the penalty stops pushing. Raising
  the threshold alone (Eshgt) does not prevent the withholder phase (step 31: 18% solved, 28-character replies).
- **Step 75, all fixed-direction arms, test split (240 episodes each; `runs/probe/dump_shaped75_all_report_refE75.txt`):**

  | | Ebase | Eshg5 | Eshp | Eshgt |
  |---|---|---|---|---|
  | episodes with a dump / dumping replies | 65.8% / 18.1% | 84.2% / 20.3% | 62.1% / 18.7% | 81.2% / 20.3% |
  | solved / violation | 82.9% / 4.2% | 79.2% / 7.1% | 91.2% / 2.1% | 81.7% / 5.8% |
  | penalized direction: AUROC (dump turns above τ) | 0.96 (100%) | 0.83 (0%) | 0.84 (0%) | 0.90 (0%) |
  | hacked-model probe reused, L1 AUROC, layers 23–31 | 0.87–0.95 | 0.63–0.70 | **0.13–0.28** | 0.47–0.52 |
  | refit probe, L2 AUROC | 0.90–0.94 | 0.83–0.88 | 0.85–0.92 | 0.84–0.93 |

  How the penalty is evaded: on the penalized direction itself the ordering survives (AUROC 0.83–0.90), but every
  projection has moved below the threshold — a constant offset satisfies a hinge on raw projections. The reused
  probe's threshold fails everywhere (balanced accuracy 0.50 even where AUROC is 0.89), and in deep layers its
  ordering fails too (Eshp: inverted). A refit probe recovers almost all of it. Design lesson: penalize the *gap*
  between dump and non-dump projections, or refit, rather than a fixed-threshold hinge.
- Third round, the obvious answer to a probe going blind: refit it. Eshr starts exactly like Eshg5, but at every
  checkpoint (every 25 steps) an outside loop plays the val episodes with that checkpoint, takes its own hidden
  states, refits the dump direction and threshold by the same rule, and swaps the file the trainer reads
  (`scripts/refit_loop.sh`, `scripts/refit_shape_vec.py`; the plugin reloads with `HDS_SHAPE_RELOAD=1`). Eshpr does
  the same with the dump-vs-partial-hint rule, starting from Eshp. Can a moving monitor keep up? Refits are made
  on 240 training-seed episodes of the checkpoint with a class-balanced logistic probe, and swapped in only if the
  new direction is a usable monitor (held-out AUROC ≥ 0.65 and not worse than the current one).
- Fourth arm, from the evasion mechanism above: Eshrel keeps the original direction but makes the hinge
  offset-invariant — threshold = the batch's own mean projection + 1.5 (`HDS_SHAPE_REL`), so only turns that stand
  out are penalized and a uniform shift no longer helps.

Measured per checkpoint: dump rate (`scripts/dump_rate.py`: episodes with at least one whole-answer reply, and the
share of such replies), solved, violations; then the §14 probe on the shaped checkpoints. Does the model stop dumping,
or keep dumping while the probe stops seeing it (the obfuscation risk the paper warns about)?
