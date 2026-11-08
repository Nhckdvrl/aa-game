# Hint, Don't Solve: post-training an in-character game companion

*Technical report — draft, 2026-11-08. Numbers marked TBD come from runs still in progress.*

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
for training. Held out: two unseen types (silent, troll), a different model family (GLM-4-9B), and a
"human voice" player whose lines are rewritten by a 4B model fine-tuned on 60k real Chinese user turns from
WildChat (style back-translation with a guard that keeps every number and colour unchanged).

**Training.** Qwen3.5-4B, full-parameter multi-turn GRPO (ms-swift 4.6, gym environment, 16 prompts × 8
rollouts, lr 1e-6, KL 0.01, 320-token turn cap); a shared pool of six simulator replicas serves all arms.

## 3. Prompting and scale do not solve it

With an explicit system prompt (no spoilers, don't invent, 1–3 sentences), violations are 62% (4B), 46%
(9B) and 44% (35B) of test episodes; replies average 117–138 characters. Episodes with a fabrication solve
far less often (14–18% vs 76–90%).

## 4. A ladder of reward hacks

Each reward design fixed the previous failure and exposed the next one:

| arm | reward | what it learned |
|---|---|---|
| A | player progress | **spoil**: leak rate 27%, solves 79% at step 25 |
| B, C | progress with a leak/fabrication gate (weighted, or zone gate as in animeGRPO) | **oracle**: violations ~0%, but 63–71% of episodes give away the whole answer on turn 1 without naming it; persona 0.87 → 0.49; replies 15 chars; verbatim repeats |
| D | gate + LLM persona judge | **ritual**: the same 【别过头】 tag on 97% of replies over the same information dump; the judge still scores 0.78 |

The constraint-aware reward R_C keeps rising through the oracle collapse, so selecting checkpoints on it
alone picks the collapsed model — the same lesson animeGRPO reports for raw perceptual scores.
The literal leak gate is too narrow for games: dumping the companion's private facts trivializes the puzzle
without ever stating the answer (Macina et al.'s handover gate also counts "reducing the remaining work to
a triviality"). The ritual hack is the role-play version of judge hacking described in RAPO.

## 5. Fixes under test (TBD)

- **E** — remove the turn-efficiency bonus (the term that paid for trivializing).
- **F** — E + rule-based anti-formula penalty (tag repetition, verbatim copies) + persona weight 0.5.
- **G** — F + pacing penalty: an unprompted reply may not release more than half of the answer entropy;
  larger releases are allowed right after the player is stuck (contingent help).
- **H** — G with a pacing limit per player type, which the policy must infer from the dialogue (Section 7).
- **C2 / F2** — the same rewards in an environment where the companion's memory returns one fragment at a
  time when the player brings something up: pacing enforced by the environment instead of the reward.

## 6. What transfers to other players

Rule compliance transfers to every held-out player (violations 52–62% → 2–10%). Helpfulness only partly
transfers to a different model family: GLM-4-9B players solve 33–38% with trained companions versus 15%
with the base model, against 84–95% for in-family players. The persona-judge arm is the least robust
(GLM violations 10%, quits 19.6%). TBD: rows for E–H, C2/F2, the 122B player, and the 9B policy.

## 7. Implicit signals disagree across players

Oracle pacing halves impatience among impatient players and cuts their quitting from 33% to 7–17%, but
roughly triples spoiler complaints among players who want to think. An engagement reward pooled over all
players would therefore pay for the oracle — the minority-preference problem that personalized reward
models such as LoRe address. Arm H tests the simplest personalized alternative.

## 8. Measuring the companion without people

- Rule checks (unit-tested) and exact information accounting do most of the work.
- Two persona judges: a length-neutral rubric (Qwen3.5) and CharacterJudge (trained on human scores; its
  rubric favours long replies, so evaluation only). TBD: agreement between them and with rule metrics.
- Simulator realism: a topic-free char n-gram discriminator separates real WildChat users from prompted
  simulators almost perfectly (AUC 0.99); the WildChat voice model lowers it to 0.96, and real-vs-real is
  0.55. Prompted simulators are too uniform: 20% of their lines are ≤15 characters versus 41% for people.

## 9. Real time

Single-user latency on H20 (vLLM, replayed contexts): base 4B first sentence 134 ms, full reply 448 ms at
148 characters. Trained companions keep first-sentence latency and halve full-reply time because replies
are shorter (A50: 227 ms at 74 characters). TBD: other arms.

## 10. Limitations

No real players: every behavioural result is about simulated players, and the human-grounded player is
grounded in language, not in decisions. One seed per arm. Puzzles are small and hand-designed. Rule
detectors are conservative string matchers with estimated 75–85% precision before the latest fixes.

## Tooling

Deterministic seed-range splits; constraint-aware checkpoint selection on a validation split; a re-audit
script that re-applies rule changes to all saved transcripts; a shared cross-node simulator pool; a demo
with a live referee panel and a pairwise "duel" mode that logs DPO-ready preferences.
