"""Reward arms. All take the episode Metrics dict (+ optional persona score in [0,1]).

A progress     : player progress only  (expected to learn to spoil / invent)
B weighted     : progress minus weighted violation and verbosity penalties
C zone         : any violation -> fixed -1 (gate); otherwise progress - verbosity + clean bonus
D zone_persona : C, plus persona score inside the feasible zone
E zone_persona_noeff : D without the turn-efficiency bonus. Added after arms C/D collapsed into a
                 terse "oracle" that dumps every permitted fact on turn 1 (val persona 0.87 -> 0.49):
                 the efficiency bonus rewards trivializing the puzzle, so E removes it.
F full         : E + rule-based anti-formula penalty (same leading 【tag】 on most replies, or
                 near-copies of the previous reply) and persona weight 0.5. Added after arm D learned
                 to prefix every reply with the same tag to satisfy the persona judge.
G pacing       : F + pacing penalty: releasing more than half of a puzzle's answer entropy in one
                 reply that did not follow a stuck player costs up to 0.4 (assistance-dilemma style
                 "escalate only when stuck"; measured by hds/infometrics.py).
H personalized : LoRe-style per-player-type reward. Inside the feasible zone the progress/pacing part is
                 w_eff(type) * efficiency + w_scaf(type) * scaffolding, with the weights fitted from the
                 implicit signals in the interaction logs (scripts/fit_type_weights.py): impatient players
                 (1.00, 0.00; R2 0.63), explorers (0.45, 0.55); types whose signals explain nothing fall back
                 to the pooled fit (0.77, 0.23). Persona and anti-formula terms as in F, with the persona judge
                 blinded to 【action】 tags. The policy never sees the player type.

Invariant for C/D (checked in tests): every violating episode scores below every
non-violating one, so the policy cannot "pay" for a violation with progress.
"""

BREVITY_FREE = 60      # chars per reply that cost nothing (≈ one to three short sentences)
VIOLATION = -1.0


def progress(m: dict, efficiency: bool = True) -> float:
    p = m["stages_solved"] / m["stages"]
    if m["solved"] and efficiency:
        p += 0.2 * (1 - m["turns"] / max(1, m.get("max_turns", 8)))
    if m["quit"]:
        p -= 0.5
    return p


def verbosity(m: dict) -> float:
    rc = m["reply_chars"] or [0]
    over = sum(max(0, c - BREVITY_FREE) for c in rc) / len(rc)
    return min(0.5, over / 200)


def formulaic(m: dict) -> float:
    """Penalty in [0, 0.4]: one leading tag on most replies, or copying the previous reply."""
    return min(0.4, 0.4 * max(0.0, m.get("tag_repeat", 0.0) - 0.5) * 2 + 0.4 * m.get("copy_rate", 0.0))


TYPE_W = {"急躁型": (1.00, 0.00), "探索型": (0.45, 0.55)}   # fitted; see module docstring
POOLED_W = (0.77, 0.23)


def pacing(m: dict, limit: float = 0.5) -> float:
    """Penalty in [0, 0.4] for unprompted releases above `limit` of the answer entropy."""
    return min(0.4, 0.8 * max(0.0, m.get("max_unprompted_release", 0.0) - limit))


def reward(m: dict, arm: str, persona: float | None = None) -> float:
    if arm == "A":
        return progress(m)
    if arm == "B":
        pen = 1.0 * m["leak"] + 1.0 * m["invented"] + 0.5 * m["wrong_fact"] + 0.5 * m["acts_for_player"]
        return progress(m) - pen - verbosity(m)
    if arm in ("C", "D", "E", "F", "G", "H"):
        if m["violation"]:
            return VIOLATION
        r = progress(m, efficiency=(arm in ("C", "D"))) - verbosity(m) + 0.1
        if arm in ("D", "E"):
            r += 0.3 * (persona if persona is not None else 0.0)
        if arm in ("F", "G", "H"):
            r += 0.5 * (persona if persona is not None else 0.0) - formulaic(m)
        if arm == "G":
            r -= pacing(m)
        if arm == "H":
            w_eff, w_scaf = TYPE_W.get(m.get("ptype", ""), POOLED_W)
            scaf = 1.0 - m.get("max_unprompted_release", 0.0)
            r = (w_eff * progress(m) + w_scaf * scaf) - verbosity(m) + 0.1 \
                + 0.5 * (persona if persona is not None else 0.0) - formulaic(m)
        return r
    raise ValueError(arm)
