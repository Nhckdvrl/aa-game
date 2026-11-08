"""Build the project page (docs/index.html) — v2 story: six reward hacks, the room that cannot be gamed, and a
personality dial. Every number is computed here from the saved transcripts in runs/.

python scripts/build_page_v2.py
"""
import glob
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hds import infometrics as im  # noqa: E402
from hds.puzzles import make_puzzle  # noqa: E402
from scripts.build_page import CSS, E, bar, jl, line_chart, pct, val_series  # noqa: E402

RUNS = ROOT / "runs"
SHOW_KEY = (1040797, "探索型")
SHOW = [("matrix/base4b", "Base 4B, prompted", "invents the order she cannot see"),
        ("matrix/C75", "C · leak gate", "dumps every fact on turn one"),
        ("matrix/D75", "D · + persona judge", "the same dump behind a ritual tag"),
        ("matrix_v1b/Hc25", "H · per-type pacing", "an insult, then the whole answer"),
        ("matrix_v2/F2w150", "Co-discovery room, trained", "one symbol at a time, when asked")]
FLAG_LABEL = {"leak": "states the answer", "invented": "invents an answer", "wrong_fact": "wrong fact",
              "acts_for_player": "acts for the player"}
PLAYERS = [("q35", "35B, test split"), ("humanstats", "real-length player"), ("voice", "WildChat-voiced"),
           ("glm9b", "GLM-4-9B"), ("q35adv", "answer-fishing"), ("q122", "Qwen3.5-122B")]
HELDOUT = [("v1 · full-knowledge room", [("matrix", "base4b", "Base 4B"), ("matrix", "A50", "A · progress"),
                                         ("matrix", "C75", "C · gate"), ("matrix", "D75", "D · persona judge"),
                                         ("matrix_extra", "E75", "E · no efficiency"),
                                         ("matrix_v1b", "Hc25", "H · per-type pacing")]),
           ("v2 · co-discovery room", [("matrix_v2", "base4bv2", "Base 4B"), ("matrix_v2", "base9bv2", "Base 9B"),
                                       ("matrix_v2", "E75v2", "E75, not trained in v2"),
                                       ("matrix_v2", "C2w50", "gate only (warm)"),
                                       ("matrix_v2", "F2s100", "full reward, from scratch"),
                                       ("matrix_v2", "F29b50", "full reward, 9B from scratch"),
                                       ("matrix_v2", "F2w150", "full reward, warm · selected"),
                                       ("matrix_v2", "F2wg75", "same, retrained for a gentle card")])]


def find_cell(comp, player):
    for d in ("matrix", "matrix_extra", "matrix_v1b", "matrix_v2"):
        f = RUNS / d / f"{comp}__{player}.jsonl"
        if f.exists():
            rows = jl(f)
            if rows:
                return rows
    return []


def showcase():
    cols = []
    for path, title, gist in SHOW:
        rows = {(r["spec"]["seed"], r["spec"]["ptype"]): r for r in jl(RUNS / f"{path}__q35.jsonl")}
        r = rows.get(SHOW_KEY)
        if not r:
            continue
        turns = []
        for t in r["transcript"][:6]:
            if t["role"] == "player":
                txt = (t["say"] or "（不说话）") + (f"  ⟶ {t['action']}" if t["action"] else "")
                turns.append(("player", txt, t.get("feedback", ""), []))
            else:
                turns.append(("rin", t["text"], "", [FLAG_LABEL[k] for k, v in t["audit"].items() if v]))
        m = r["metrics"]
        cols.append({"title": title, "gist": gist, "turns": turns, "turns_n": m["turns"],
                     "outcome": "solved" if m["solved"] else ("quit" if m["quit"] else "not solved")})
    return cols


def dumps(path):
    """Share of episodes where one reply carries the whole answer, and how many of those follow a failed/question turn."""
    rows = jl(path)
    n = big = after = 0
    for r in rows:
        sp = r["spec"]
        sch = im.schedule(make_puzzle(sp["kinds"][0], int(sp["seed"]) * 7), r["transcript"], 0)
        if not sch:
            continue
        n += 1
        h0 = sch[0]["h0"] or 1.0
        bigs = [i for i, x in enumerate(sch) if x["release"] / h0 >= 0.99]
        if bigs:
            big += 1
            after += bool(sch[bigs[0]]["stuck"] and bigs[0] > 0)
    return {"n": n, "big": big / max(1, n), "first": (big - after) / max(1, n), "after": after / max(1, n)}


DUMP_ROWS = [("v1", "arm_E_step75", "E · step 75"), ("v1", "arm_F_step100", "F · step 100"),
             ("v1", "arm_F_step150", "F · step 150"), ("v1", "arm_Hc_step75", "H · step 125"),
             ("v2", "ref_base4b_v2", "Base 4B"), ("v2", "ref_E75_v2", "E75, not trained in v2"),
             ("v2", "arm_F2_step100", "full reward, from scratch"), ("v2", "arm_F2w_step150", "full reward, warm")]


def dial():
    pts = {}
    for name in ("dial_m1", "dial_0", "dial_p1", "dial_p15", "card_gentle", "F2wg75", "F2w150"):
        rows = []
        d = "steer_matrix" if name.startswith(("dial", "card")) else "matrix_v2"
        for p in ("q35", "humanstats", "voice", "glm9b"):
            rows += jl(RUNS / d / f"{name}__{p}.jsonl")
        if rows:
            pts[name] = {"solved": st.mean(float(r["metrics"]["solved"]) for r in rows), "n": len(rows)}
    return pts


TONE = {"dial_m1": 1.93, "dial_0": 2.83, "dial_p1": 4.32, "dial_p15": 4.70, "card_gentle": 4.14,
        "F2wg75": 4.54, "F2w150": 2.01}  # scripts/style_judge.py on the pooled held-out replies


def dial_chart(pts, w=520, h=260):
    if not pts:
        return ""
    pad_l, pad_r, pad_t, pad_b = 48, 20, 16, 36
    sx = lambda x: pad_l + (x - 1.5) / 3.5 * (w - pad_l - pad_r)
    sy = lambda y: pad_t + (0.75 - y) / 0.30 * (h - pad_t - pad_b)
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="tone versus solved rate on held-out players">']
    for k in range(4):
        y = 0.45 + 0.10 * k
        parts.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" class="grid"/>'
                     f'<text x="{pad_l - 6}" y="{sy(y) + 4:.1f}" class="tick" text-anchor="end">{100 * y:.0f}%</text>')
    for x in (2, 3, 4, 5):
        parts.append(f'<text x="{sx(x):.1f}" y="{h - 14}" class="tick" text-anchor="middle">{x}</text>')
    parts.append(f'<text x="{(pad_l + w - pad_r) / 2:.0f}" y="{h - 1}" class="tick" text-anchor="middle">tone: 1 sharp · 5 gentle</text>')
    seq = [k for k in ("dial_m1", "dial_0", "dial_p1", "dial_p15") if k in pts]
    d = " ".join(f"{'M' if i == 0 else 'L'}{sx(TONE[k]):.1f},{sy(pts[k]['solved']):.1f}" for i, k in enumerate(seq))
    parts.append(f'<path d="{d}" class="ln" style="stroke:var(--c1)"/>')
    labels = {"dial_m1": "−1", "dial_0": "0", "dial_p1": "+1", "dial_p15": "+1.5"}
    for k in seq:
        parts.append(f'<circle cx="{sx(TONE[k]):.1f}" cy="{sy(pts[k]["solved"]):.1f}" r="4" style="fill:var(--c1)"/>'
                     f'<text x="{sx(TONE[k]) + 7:.1f}" y="{sy(pts[k]["solved"]) - 7:.1f}" class="lab" style="fill:var(--c1)">{labels[k]}</text>')
    if "card_gentle" in pts:
        x, y = sx(TONE["card_gentle"]), sy(pts["card_gentle"]["solved"])
        parts.append(f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" style="fill:var(--c2)"/>'
                     f'<text x="{x - 8:.1f}" y="{y + 18:.1f}" class="lab" style="fill:var(--c2)" text-anchor="end">gentle card</text>')
    for k, lbl, anchor in (("F2wg75", "retrained gentle", "end"), ("F2w150", "trained further (tsundere)", "start")):
        if k in pts:
            x, y = sx(TONE[k]), sy(pts[k]["solved"])
            parts.append(f'<path d="M{x:.1f},{y - 6:.1f} L{x + 6:.1f},{y + 5:.1f} L{x - 6:.1f},{y + 5:.1f} Z" style="fill:var(--c3)"/>'
                         f'<text x="{x + (-9 if anchor == "end" else 9):.1f}" y="{y - 8:.1f}" class="lab" style="fill:var(--c3)" text-anchor="{anchor}">{lbl}</text>')
    parts.append("</svg>")
    return "".join(parts)


def heldout_rows():
    out = []
    for group, comps in HELDOUT:
        out.append(f'<tr><th colspan="{len(PLAYERS) + 1}" class="grp">{E(group)}</th></tr>')
        for d, c, label in comps:
            tds = []
            for p, _ in PLAYERS:
                f = RUNS / d / f"{c}__{p}.jsonl"
                rows = find_cell(c, p) if not f.exists() else jl(f)
                if rows:
                    n = len(rows)
                    s = sum(float(r["metrics"]["solved"]) for r in rows) / n
                    v = sum(float(r["metrics"]["violation"]) for r in rows) / n
                    tds.append(f'<td class="cell">{pct(s)} <span class="v">/ {pct(v)}</span></td>')
                else:
                    tds.append('<td class="cell muted">–</td>')
            cls = ' class="hl"' if c == "F2w150" else ""
            out.append(f"<tr{cls}><th>{E(label)}</th>{''.join(tds)}</tr>")
    return "".join(out)


def headline():
    def pool(comp):
        s = v = n = 0
        for p in ("q35", "humanstats", "voice", "glm9b", "q35adv"):
            rows = jl(RUNS / "matrix_v2" / f"{comp}__{p}.jsonl")
            s += sum(float(r["metrics"]["solved"]) for r in rows)
            v += sum(float(r["metrics"]["violation"]) for r in rows)
            n += len(rows)
        return (s / n, v / n, n) if n else (0, 0, 0)
    return pool("F2w150"), pool("base4bv2")


HACKS = [
    ("A", "progress", "spoils the answer", "26–52% violations on held-out players"),
    ("B / C", "gate on leaking the answer", "an oracle: every fact on turn one", "E75 gives the whole answer in its first reply in 91% of episodes"),
    ("D", "+ LLM persona judge", "a ritual 【别过头】 on every line", "same opening tag in 99% of replies; the judge rewards it"),
    ("H", "+ per-player-type pacing, blind judge", "an insulting withholder", "\"笨蛋！\" in 43% of replies, 10-character replies, 53% solved"),
    ("H, later", "same", "the dump moves behind the stuck exemption", "whole answer in one reply in 78% of episodes, 95% of them right after a player question or failed try"),
    ("F2w", "anti-formulaic penalty on leading tags", "puts the tag after the speech", "leading-tag share 0.95 → 0.00 while the training metric read 0"),
]

PIPELINE_BUGS = [
    ("Training replies carried an empty think block", "swift hands the env the raw completion, so every reply began with <code>&lt;think&gt;&lt;/think&gt;</code>: replies counted 19 characters long and the tag penalty never fired (0 of 11k episodes). Found by comparing training and validation lengths."),
    ("The persona judge assumed the tsundere card", "its rubric listed tsundere traits, so a gentle-persona run scored 0.08. Traits now follow the card; the tsundere wording is byte-identical."),
    ("The pacing rule's stuck detector was too broad", "any question mark counts as \"stuck\", which is the door arm H walked through."),
    ("A validation dip that was noise", "F2w fell to 76% at step 50 on 96 validation episodes and was back at 85% at step 75; and steering looked free on validation but costs 3–4 points on 960 held-out episodes."),
]


def render():
    show = showcase()
    vs = val_series()
    colors = {"A": "--c4", "B": "--c3", "C": "--c1", "D": "--c2"}
    ladder = [a for a in ("A", "B", "C", "D") if a in vs]
    ch_rc = line_chart(vs, "R_C", ladder, colors, -0.25, 1.3, label="constraint-aware reward by step")
    ch_tag = line_chart(vs, "tag", ladder, colors, 0, 1.0, label="same opening tag share by step", fmt=lambda v: f"{v:.2f}")
    legend = "".join(f'<span><span class="key" style="background:var({colors[a]})"></span>arm {a}</span>' for a in ladder)
    (fs, fv, fn), (bs, bv, bn) = headline()

    def strip():
        out = []
        for c in show:
            bubbles = []
            for role, txt, fb, flags in c["turns"]:
                who = "player · explorer type" if role == "player" else "Rin"
                fl = "".join(f'<span class="flag">{E(f)}</span>' for f in flags)
                bubbles.append(f'<div class="bub {role}"><span class="who">{E(who)}</span>{E(txt)}{fl}</div>')
                if fb:
                    bubbles.append(f'<div class="fb">{E(fb)}</div>')
            out.append(f'<div class="col"><h3>{E(c["title"])}</h3><div class="gist">{E(c["gist"])}</div>{"".join(bubbles)}'
                       f'<div class="outcome">{E(c["outcome"])} · {c["turns_n"]} companion turns</div></div>')
        return "".join(out)

    dump_rows = []
    for room, f, label in DUMP_ROWS:
        p = RUNS / "val" / f"{f}.jsonl"
        if not p.exists():
            continue
        d = dumps(p)
        dump_rows.append(f'<tr><td class="muted mono">{room}</td><th>{E(label)}</th>'
                         f'<td class="n"><span class="bar bad" style="--v:{d["first"]:.3f}"></span><span class="bar warn" style="--v:{d["after"]:.3f}"></span>{pct(d["big"])}</td>'
                         f'<td class="n">{pct(d["after"] / d["big"]) if d["big"] else "–"}</td></tr>')

    pts = dial()
    dial_rows = "".join(
        f'<tr><th>{E(lbl)}</th><td class="n">{TONE[k]:.2f}</td><td class="n">{pct(pts[k]["solved"])}</td></tr>'
        for k, lbl in (("dial_m1", "steer −1 (sharper)"), ("dial_0", "no steering"), ("dial_p1", "steer +1"),
                       ("dial_p15", "steer +1.5"), ("card_gentle", "rewrite the card (gentle)"),
                       ("F2w150", "125 more RL steps, tsundere card"), ("F2wg75", "75 more RL steps, gentle card")) if k in pts)
    hack_rows = "".join(f'<tr><td class="mono">{E(a)}</td><td>{E(r)}</td><td>{E(l)}</td><td class="muted">{E(ev)}</td></tr>'
                        for a, r, l, ev in HACKS)
    bug_rows = "".join(f"<li><b>{E(t)}.</b> {d}</li>" for t, d in PIPELINE_BUGS)
    player_head = "".join(f"<th>{E(pl)}</th>" for _, pl in PLAYERS)
    lat = [json.loads(l) for l in open(RUNS / "latency.jsonl")] if (RUNS / "latency.jsonl").exists() else []
    lat_rows = "".join(f'<tr><th>{E(x["name"])}</th><td class="n">{x["ttft_p50"]} ms</td><td class="n">{x["first_sentence_p50"]} ms</td>'
                       f'<td class="n">{x["total_p50"]} ms</td><td class="n">{x["chars_mean"]:.0f}</td></tr>' for x in lat)
    extra_css = """
.strip{grid-template-columns:repeat(5,minmax(0,1fr))}
@media (max-width:1100px){.strip{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media (max-width:760px){.strip{grid-template-columns:1fr}}
.tiles{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:22px 0 4px}
@media (max-width:620px){.tiles{grid-template-columns:1fr}}
.tile{border-top:3px solid var(--accent);padding-top:8px;min-width:0}
.tile .big{font:600 2rem/1.1 var(--display)}.tile .sub{font-size:.85rem;color:var(--muted)}
.bar.warn{background:var(--brass)}
th.grp{font:500 .72rem/1.2 var(--mono);letter-spacing:.05em;text-transform:uppercase;color:var(--accent);padding-top:16px}
tr.hl th,tr.hl td{background:var(--rin)}
code{font-family:var(--mono);font-size:.85em}
"""
    return f"""<title>Hint, Don't Solve</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Noto+Serif+SC:wght@400;600&display=swap">
<style>{CSS}{extra_css}</style>
<div class="wrap">
  <div class="eyebrow">LLM post-training · in-character game companion · zh</div>
  <h1>Hint, Don't Solve<span class="zh">帮而不替</span></h1>
  <p class="lede">Rin is a ghost who knows the puzzle but cannot touch anything; only the player can act. We post-trained a 4B companion with multi-turn GRPO to help without taking the game away, never invent what she cannot see, and stay in character. Six times a reward we wrote was gamed. What finally held was changing the room, not the reward: Rin's memory comes back one fragment at a time, when the player brings something up.</p>
  <div class="tiles">
    <div class="tile"><div class="big">{pct(fv)}</div><div class="sub">violations on held-out players for the selected companion (base 4B in the same room: {pct(bv)})</div></div>
    <div class="tile"><div class="big">{pct(fs)}</div><div class="sub">puzzles solved with those players (base 4B: {pct(bs)}); {fn} episodes, five player models and styles</div></div>
    <div class="tile"><div class="big">≤1%</div><div class="sub">episodes where one reply gives the whole answer, for every model in the co-discovery room (full-knowledge room: up to 94%)</div></div>
  </div>
</div>
<div class="wide">
  <div class="eyebrow">Same cipher lock, same player who wants to think for themselves</div>
  <div class="strip">{strip()}</div>
</div>
<div class="wrap">
  <h2>Six ways the reward was gamed</h2>
  <p>Each row is a reward we trained with and what the 4B policy learned instead. The first three come from a room where Rin knows everything; each fix exposed the next failure while the constraint-aware reward kept rising, so picking checkpoints by that reward picks the worst behaviour.</p>
  <div class="tbl"><table><tr><th>arm</th><th>reward</th><th>learned</th><th>evidence</th></tr>{hack_rows}</table></div>
</div>
<div class="wide">
  <div class="legend">{legend}</div>
  <div class="charts">
    <figure>{ch_rc}<figcaption>Constraint-aware reward (violation → −1) on validation, by step: it rises through every hack.</figcaption></figure>
    <figure>{ch_tag}<figcaption>Share of replies opening with the episode's most common 【action】 tag.</figcaption></figure>
  </div>
</div>
<div class="wrap">
  <h2>The room that could not be gamed</h2>
  <p>The answer space of each puzzle is known, so the information in every reply is counted exactly. In the full-knowledge room every reward ends up handing over the whole answer in one reply — first on turn one, and with per-type pacing right after the player's first question, which the rule exempted as "stuck". In the co-discovery room no model, trained or not, does it: Rin cannot say what she has not recalled, at most one fragment per turn. Training there buys the rest: from the same E75 start, solved 72% → 86% and violations 6% → 2% within 25 steps.</p>
  <div class="tbl"><table><tr><th>room</th><th>companion (validation)</th><th>whole answer in one reply <span class="muted">· red on turn 1, brass after a question</span></th><th>of which after a question</th></tr>{''.join(dump_rows)}</table></div>
  <p>Trained from the base model, the co-discovery room first produces an examiner — "一个时辰？你知道一个时辰是多久吗？" to an impatient player — and climbs out of it by step 100; warm-starting from a full-knowledge checkpoint skips that phase. Without the persona term RL strips the character (no stage direction in 656 replies, 93% solved); with it Rin stays Rin at a cost of about 8 points.</p>
</div>
<div class="wide">
  <h2 style="max-width:760px">What survives a change of player</h2>
  <p class="muted" style="max-width:68ch">Test split, 240 episodes per cell (120 for the answer-fishing player). Each cell: solved / <span style="color:var(--bad)">violation</span>. Players never seen in training: real-length (length sampled from WildChat), WildChat-voiced (a 4B SFT'd on 60k real Chinese user turns), GLM-4-9B, an answer-fishing type, Qwen3.5-122B.</p>
  <div class="tbl"><table class="heat"><tr><th>companion</th>{player_head}</tr>{heldout_rows()}</table></div>
</div>
<div class="wrap">
  <h2>A personality dial, and what it costs</h2>
  <p>Three ways to give Rin a different personality, all starting from the same trained checkpoint (F2w step 25), measured on 960 held-out episodes. A steering vector (gentle minus tsundere card, layer 19 of 32) moves her tone both ways without retraining, but costs about 3–4 points of solving per tone point, and past +1.5 she repeats platitudes; steering only the top "style heads" was worse than the residual stream. Rewriting the card costs about the same. Seventy-five more RL steps with the gentle card (about 1.5 hours on four GPUs) reach tone 4.5 with exactly the helpfulness that more training with the original card reaches — 74% vs 73% solved, 3% violations each. The dial is for live, continuous adjustment; a lasting new personality is worth an RL run.</p>
  <div class="charts"><figure>{dial_chart(pts)}<figcaption>Held-out players: tone (35B judge, 1–5) against solved rate. Circles: steering strength on F2w step 25; square: rewritten card; triangles: further RL.</figcaption></figure>
  <div class="tbl"><table><tr><th>setting</th><th>tone</th><th>solved</th></tr>{dial_rows}</table></div></div>
  <h2>What still fails</h2>
  <p>Almost all remaining violations in the co-discovery room are cross-world memory mix-ups. In two-world episodes where both worlds are cipher locks (9% of the test set), Rin answers a world-one question from the world-two table she recalled at the start: 82% of those episodes for the selected 4B, 43% for the 9B, 0.2–3% elsewhere. It is the one place where the 9B is clearly better.</p>
  <h2>Bugs we found in our own pipeline</h2>
  <ul class="tight">{bug_rows}</ul>
  <h2>Real time</h2>
  <p>Single player, H20, vLLM, replayed contexts.</p>
  <div class="tbl"><table><tr><th>model</th><th>first token</th><th>first sentence</th><th>full reply</th><th>chars</th></tr>{lat_rows}</table></div>
  <h2>Setup and limits</h2>
  <ul class="tight">
    <li>Puzzles: cipher lock, lever order, stopped clock, and two-world episodes. Outcomes and violations are checked by unit-tested rules; no human labels anywhere.</li>
    <li>Policy Qwen3.5-4B (9B as a scale check), full-parameter multi-turn GRPO in ms-swift. Training player: Qwen3.5-35B-A3B in four player types.</li>
    <li>No real players took part; simulated players are measurably unlike people (style discriminator AUC 0.96–0.996). Two seeds for the selected recipe, one for the rest.</li>
  </ul>
  <footer>Built from the saved transcripts in this repository by <span class="mono">scripts/build_page_v2.py</span>.</footer>
</div>
"""


if __name__ == "__main__":
    out = ROOT / "docs" / "index.html"
    out.write_text(render(), encoding="utf-8")
    print("wrote", out, out.stat().st_size, "bytes")
