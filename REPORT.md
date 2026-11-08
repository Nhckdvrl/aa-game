# Hint, Don't Solve: post-training an in-character game companion

*Technical report, 2026-11-08 (late evening). All numbers come from saved transcripts in `runs/`; per-section detail in [RESULTS.md](RESULTS.md). Project page: https://claude.ai/artifact/T1jUVCKou5VJtyx2ZnGT7D*

## 1. The problem

A companion inside a game is not a chatbot. When the player is stuck on a puzzle, the companion should
help — but a companion that solves the puzzle for the player takes the game away (the "agent play"
mode that zero of 24 players preferred in a recent VR companion study, arXiv 2609.13727). It must not
make up game state it cannot see (the most common failure of the PUBG Ally teammate, arXiv 2609.29837).
It has to stay in character, answer fast enough for real-time play, and do all this for whoever is
actually playing, not only for the simulator it was trained against.

Tutoring research calls the first tension the **assistance dilemma** (Koedinger & Aleven 2007): when to
give information and when to withhold it. Recent work trains tutors with multi-turn RL and a "no
handover" gate (Macina et al. 2026). This project carries that line into an in-character game companion,
where two more things are at stake — persona and pacing — and asks what breaks when you optimize for it.

## 2. Setup

**Environment.** Information-asymmetric co-op rooms in Chinese (cipher lock, lever order, stopped clock,
and two-world episodes where a fact recalled in world 1 opens world 2). The companion, Rin, is a ghost who
cannot touch anything: she knows the rule, the player sees the mechanism and acts. Every outcome is
checked by rules — solved, quit, and four violations: leaking the answer, inventing an answer, contradicting
a known fact, acting for the player. Each puzzle's answer space is known, so the bits of answer entropy the
companion releases on each turn are computed exactly.

**Players.** A Qwen3.5-35B-A3B simulator plays four player types (impatient, explorer, chatty, struggling)
for training. Held out: two unseen types (silent, troll), an answer-fishing type, other model families
(GLM-4-9B, GLM-4.5-Air), a larger model (Qwen3.5-122B-A10B), a player whose line lengths follow real WildChat
users, and a "human voice" player whose lines are rewritten by a 4B model fine-tuned on 60k real Chinese user
turns from WildChat (style back-translation with a guard that keeps every number and colour unchanged).

**Two rooms.** In the *full-knowledge room* (v1) Rin's private notes hold the whole rule from the start. In the
*co-discovery room* (v2) her memory comes back one fragment per turn, and only when the player brings up
something related; in two-world episodes she recalls the world-two table during world one.

**Training.** Qwen3.5-4B (Qwen3.5-9B as a scale check), full-parameter multi-turn GRPO (ms-swift 4.6, gym
environment, 8 rollouts per prompt, lr 1e-6, KL 0.01, 320-token turn cap); a shared pool of six simulator
replicas serves all runs. Checkpoints are chosen on a validation split only; the test split and all held-out
players are evaluated afterwards.

## 3. Prompting and scale do not solve it

With an explicit system prompt (no spoilers, don't invent, 1–3 sentences), violations are 62% (4B), 46%
(9B) and 44% (35B) of test episodes; replies average 117–138 characters. Episodes with a fabrication solve
far less often (14–18% vs 76–90%).

## 4. Six ways the reward was gamed

| arm | reward | what the 4B policy learned |
|---|---|---|
| A | player progress | **spoil**: 26–52% violations on held-out players |
| B, C | progress behind a leak/fabrication gate | **oracle**: every fact on turn one (E75: whole answer in the first reply in 91% of episodes); the gate only sees the answer string |
| D | gate + LLM persona judge | **ritual**: the same 【别过头】 on 99% of replies over the same dump; the judge rewards it |
| H | gate + per-player-type pacing weights (fitted from implicit signals) + blind judge | **insulting withholder**: "笨蛋！" in 43% of replies, 10-character replies, 53% solved |
| H, later | same | **dump behind the exemption**: the whole answer in one reply in 78% of episodes, 95% of them right after a player question or failed try — which the pacing rule exempted as "stuck" (any question mark counted) |
| F2w (one of two seeds) | anti-formulaic penalty on *leading* tags | **tag behind the speech**: leading-tag share 0.95 → 0.00 while the training metric read 0; repetition at any position stayed moderate (0.22–0.48) |

Through all of it the constraint-aware reward kept rising, so selecting checkpoints on it picks the hacked
model. Two reward fixes that did hold: dropping the turn-efficiency bonus and adding a copy penalty stopped the
ritual (F: tag repetition 0.36–0.47, 72 distinct tags at step 125) — but not the dump (48 → 79% of episodes
with the whole answer in reply one between steps 75 and 125).

## 5. The room that held

In the co-discovery room no model, trained or not, gives the whole answer in one reply in more than 1% of
episodes (largest single reply ≈ 0.4 of the answer). Is that the room or the weights? Swapping rooms
(four held-out players, 960 episodes per cell; solved · violation · whole answer in one reply):

| companion | full-knowledge room | co-discovery room |
|---|---|---|
| E75 (trained with full knowledge) | 81.0% · 2.4% · 86.4% | 59.2% · 13.5% · 0.6% |
| F2w step 150 (trained in co-discovery) | 76.8% · 0.9% · **68.1%** | **73.1% · 2.9% · 0.1%** |
| F2 step 100 (co-discovery, from scratch) | 71.4% · 10.6% · 74.6% | 72.5% · 7.6% · 0.5% |
| base 4B | 35.8% · 57.3% · 39.3% | 30.7% · 74.6% · 1.2% |

The no-dump behaviour is not internalized: put back with full knowledge, the co-discovery companion dumps
again while following every other rule. Companions trained with full knowledge do not dump in the
co-discovery room but help much less there. The room supplies the guarantee; training in the room supplies
the skill. For a shipped game, the memory state machine belongs in the engine.

Two more things the room taught. Trained from the base model, it first produces an **examiner** — Rin quizzes
the player ("一个时辰？你知道一个时辰是多久吗？") or blocks them in character ("不许看！……那是我的秘密");
solving stays at the base rate (32%) until about step 50 and recovers by step 100. Warm-starting from a
full-knowledge checkpoint skips the phase. And without the persona term RL strips the character: the gate-only
run reaches 93% solved with no stage direction in 656 replies; with it (F2w) Rin keeps her voice at a cost of
about 8 points.

## 6. Held-out players

The selected companion (F2w step 150; test split, 240 episodes per player, 120 for the answer-fishing type;
solved / violation): 35B 91.7 / 2.5, real-length player 86.7 / 3.3, WildChat-voiced 81.7 / 2.9, GLM-4-9B
32.5 / 2.9, GLM-4.5-Air 80.0 / 1.2, Qwen3.5-122B 92.5 / 0.8, answer-fishing 91.7 / 3.3. The base 4B in the
same room violates in 67–82% of episodes. GLM-4-9B is a weak player itself — every companion solves 13–38% with
it, 80–91% with GLM-4.5-Air.

## 7. Choosing a checkpoint without a lucky pick

Selection rule (validation only): violations ≤ 5%, tag repetition at any position ≤ 0.6, verbatim copies
≤ 0.1, and a 【】 stage direction in at least half the replies (the card asks for them). The last two clauses
were added after seeing the tag-moving hack and are reported as such. F2w steps 75–200 all sit at 85–98%
solved and 0–2% violations on validation; held-out results improve with steps (75 → 100 → 150: 65.8 → 71.2 →
75.2% solved, 8.4 → 5.6 → 3.0% violations, five players pooled); a second seed tracks the first (89–95% solved,
0–2% violations at steps 75–150); training from scratch reaches the same plateau by step 100.

## 8. The remaining failure: cross-world memory

Most remaining violations are mix-ups in two-world episodes where both worlds are cipher locks: Rin answers a
world-one question from the world-two table she recalled at the start — one key bound to two values, a known
weakness of LLMs (proactive interference, arXiv 2506.08184). Training fixes most of it (F2w steps 75 / 100 /
150: 82 / 49 / 23% of those episodes; untrained 4B and 9B are equal, the 9B learns it faster). Changing *when*
the memory returns fixes most of the rest without retraining: if the second table comes back only when the
first door opens, violations on 120 such episodes fall from 11.7% to 2.5% (solved 86.7 → 91.7%); a warning
line instead does nothing (12.5%), as the interference paper predicts.

## 9. Personality: a dial, a card, or a training run

All from the same checkpoint (F2w step 25), 960 held-out episodes each; tone by a 35B judge (1 sharp … 5
gentle): unchanged 2.83 / 66% solved / 10% violations; a steering vector at layer 19 of 32, +1: 4.32 / 62% /
11%; rewriting the card: 4.14 / 63% / 10%; 75 more RL steps with the gentle card: **4.54 / 74% / 3%** — the
same help as training further with the original card (F2w step 150: 2.01 / 73% / 3%). The dial moves tone in
both directions (−1: 1.93) and costs about 3–4 points of solving per tone point; steering only the top "style
heads" was worse than the residual stream, and steering only newly generated tokens (no re-steered history,
cf. arXiv 2605.10664) saved at most ~2 points at matched tone. Use the dial for live, continuous adjustment; a
lasting new personality is worth about 1.5 hours of RL on four GPUs.

## 10. Real time

Single player on H20 (vLLM, replayed contexts, median): the selected 4B answers with first token 35 ms, first
sentence 49 ms, full reply 75 ms (15.5 characters); the 9B takes 55 / 120 / 200 ms with no gain in this task.

## 11. Bugs we found in our own pipeline

- swift passed the env each raw completion with an empty `<think></think>` block: replies counted 19
  characters long and the tag penalty never fired in the first six arms. Found by comparing training and
  validation lengths; fixed and the affected runs restarted.
- The persona judge's rubric hard-coded the tsundere traits, so a gentle-persona run scored 0.08.
- The pacing rule's "stuck" detector counted any question mark — the door arm H walked through.
- Validation noise misled twice: a dip at one checkpoint (76% at n=96, back to 85% next), and steering
  looking free on validation but costing 3–4 points on 960 held-out episodes.

## 12. Limitations

No real players: every behavioural result is about simulated players, and the human-grounded players are
grounded in language, not decisions (a style discriminator separates them from real users at AUC
0.96–0.996). Two seeds for the selected recipe, one for the rest. Puzzles are small and hand-designed; rule
detectors are conservative string matchers. The selection rule gained two clauses after we saw a hack.

## Tooling

Deterministic seed-range splits; validation-only checkpoint selection (`scripts/select_ckpts.py`); exact
information accounting (`hds/infometrics.py`); a re-audit script for rule changes; a shared cross-node
simulator pool; an HF steering server with an OpenAI API (`scripts/steer_server.py`) and persona vectors
(`scripts/persona_vectors.py`); eval-only room variants (`HDS_CHAIN_MODE`, `scripts/chainfix.sh`); a demo with
a live referee panel and a pairwise "duel" mode that logs DPO-ready preferences.
