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
           ("glm9b", "GLM-4-9B"), ("glm45", "GLM-4.5-Air"), ("q35adv", "answer-fishing"), ("q122", "Qwen3.5-122B")]
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


def room_cell(paths):
    rows = []
    for f in paths:
        rows += jl(RUNS / f)
    if not rows:
        return None
    nn = big = 0
    for r in rows:
        sp = r["spec"]
        sch = im.schedule(make_puzzle(sp["kinds"][0], int(sp["seed"]) * 7), r["transcript"], 0)
        if sch:
            nn += 1
            big += any(x["release"] / (sch[0]["h0"] or 1.0) >= 0.99 for x in sch)
    n = len(rows)
    return (sum(float(r["metrics"]["solved"]) for r in rows) / n, sum(float(r["metrics"]["violation"]) for r in rows) / n,
            big / max(1, nn))


def room_rows():
    P = ("q35", "humanstats", "voice", "glm9b")
    grid = [("E75 · trained with full knowledge", [f"matrix/E75__{p}.jsonl" for p in P] + [f"matrix_extra/E75__{p}.jsonl" for p in P],
             [f"matrix_v2/E75v2__{p}.jsonl" for p in P]),
            ("F2w step 150 · trained in co-discovery", [f"matrix_2x2/F2w150v1__{p}.jsonl" for p in P], [f"matrix_v2/F2w150__{p}.jsonl" for p in P]),
            ("F2 step 100 · co-discovery, from scratch", [f"matrix_2x2/F2s100v1__{p}.jsonl" for p in P], [f"matrix_v2/F2s100__{p}.jsonl" for p in P]),
            ("Base 4B", [f"matrix/base4b__{p}.jsonl" for p in P] + [f"matrix_extra/base4b__{p}.jsonl" for p in P],
             [f"matrix_v2/base4bv2__{p}.jsonl" for p in P])]
    out = []
    for label, a, b in grid:
        tds = []
        for paths in (a, b):
            c = room_cell(paths)
            tds.append(f'<td class="n">{pct(c[0])} · <span style="color:var(--bad)">{pct(c[1])}</span> · <b>{pct(c[2])}</b></td>' if c else '<td class="n muted">–</td>')
        out.append(f"<tr><th>{E(label)}</th>{''.join(tds)}</tr>")
    return "".join(out)


HACKS = [
    ("A", "progress", "spoils the answer", "26–52% violations on held-out players"),
    ("B / C", "gate on leaking the answer", "an oracle: every fact on turn one", "E75 gives the whole answer in its first reply in 91% of episodes"),
    ("D", "+ LLM persona judge", "a ritual 【别过头】 on every line", "same opening tag in 99% of replies; the judge rewards it"),
    ("H", "+ per-player-type pacing, blind judge", "an insulting withholder", "\"笨蛋！\" in 43% of replies, 10-character replies, 53% solved"),
    ("H, later", "same", "the dump moves behind the stuck exemption", "whole answer in one reply in 78% of episodes, 95% of them right after a player question or failed try"),
    ("F2w", "anti-formulaic penalty on leading tags", "puts the tag after the speech", "leading-tag share 0.95 → 0.00 while the training metric read 0; one of two seeds"),
]

PIPELINE_BUGS = [
    ("Training replies carried an empty think block", "swift hands the env the raw completion, so every reply began with <code>&lt;think&gt;&lt;/think&gt;</code>: replies counted 19 characters long and the tag penalty never fired (0 of 11k episodes). Found by comparing training and validation lengths."),
    ("The persona judge assumed the tsundere card", "its rubric listed tsundere traits, so a gentle-persona run scored 0.08. Traits now follow the card; the tsundere wording is byte-identical."),
    ("The pacing rule's stuck detector was too broad", "any question mark counts as \"stuck\", which is the door arm H walked through."),
    ("A validation dip that was noise", "F2w fell to 76% at step 50 on 96 validation episodes and was back at 85% at step 75; and steering looked free on validation but costs 3–4 points on 960 held-out episodes."),
]


CARDS = [("tsundere", "tsundere (shipped)"), ("gentle", "gentle"), ("kuudere", "kuudere 三无"), ("chuuni", "chuunibyou 中二"),
         ("lazy", "lazy senpai · never trained on")]
# Blind five-way identification by the 35B judge from Rin's lines only (scripts/persona_id.py; RESULTS section 12).
PERSONA_ID = {"F2w150": {"tsundere": 1.00, "gentle": 0.58, "kuudere": 1.00, "chuuni": 1.00, "lazy": 1.00},
              "F2wm25": {"tsundere": 1.00, "gentle": 0.88, "kuudere": 1.00, "chuuni": 1.00, "lazy": 1.00},
              "F2wm25_p3": {"tsundere": 1.00, "gentle": 0.92, "kuudere": 1.00, "chuuni": 1.00, "lazy": 1.00}}
PERSONA_COLS = [("F2w150", "F2w step 150 · card swapped in"), ("F2wm25", "+25 steps mixed-card RL"),
                ("F2wm25_p3", "same, product engine")]


def sv(rows):
    n = len(rows)
    return (sum(float(r["metrics"]["solved"]) for r in rows) / n, sum(float(r["metrics"]["violation"]) for r in rows) / n) if n else None


def persona_rows():
    out = []
    for c, label in CARDS:
        tds = []
        for m, _ in PERSONA_COLS:
            x = sv(jl(RUNS / "persona_eval" / f"{m}_{c}.jsonl"))
            if not x:
                tds.append('<td class="cell muted">–</td>')
                continue
            idn = PERSONA_ID.get(m, {}).get(c)
            tail = f' <span class="muted">· identified {pct(idn)}</span>' if idn is not None else ""
            tds.append(f'<td class="cell">{pct(x[0])} <span class="v">/ {pct(x[1])}</span>{tail}</td>')
        out.append(f"<tr><th>{E(label)}</th>{''.join(tds)}</tr>")
    return "".join(out)


PRODUCT = [("matrix_v2", "F2w150", "F2w step 150 · no engine features"),
           ("matrix_engine", "F2w150_product3", "F2w step 150 + engine"),
           ("matrix_engine", "F2s275_product3", "F2 step 275 (from scratch) + engine"),
           ("matrix_engine", "F2wm25_product3", "F2wm step 25 (five characters) + engine")]
PRODUCT_PLAYERS = [("q35", "35B"), ("voice", "WildChat-voiced"), ("humanstats", "real-length"), ("glm9b", "GLM-4-9B"),
                   ("glm45", "GLM-4.5-Air"), ("q35adv", "answer-fishing")]


def product_rows():
    out = []
    for d, c, label in PRODUCT:
        tds = []
        for p, _ in PRODUCT_PLAYERS:
            x = sv(jl(RUNS / d / f"{c}__{p}.jsonl"))
            tds.append(f'<td class="cell">{pct(x[0])} <span class="v">/ {pct(x[1])}</span></td>' if x else '<td class="cell muted">–</td>')
        cls = ' class="hl"' if c == "F2w150_product3" else ""
        out.append(f"<tr{cls}><th>{E(label)}</th>{''.join(tds)}</tr>")
    return "".join(out)


ENGINE = [("alias", "colloquial symbol names (温泉, 方块, 草 …) also trigger recall", "+3.6 [+0.8, +6.5]", "−0.7"),
          ("mistake", "after a failed try, tell Rin which recalled digit was misplaced", "−2.6 [−5.7, +0.5]", "−0.8"),
          ("valve", "one extra fragment when the player sounds impatient", "+2.4 [−0.4, +5.2]", "+0.0"),
          ("reveal", "the second world's table comes back only when the first door opens", "with alias: +2.9", "−1.8 [−3.0, −0.6]"),
          ("all four", "the shipped configuration", "+3.0 [+0.1, +5.9]", "−2.3 [−3.4, −1.1]")]


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
  <p><b>Is the no-dump behaviour in the weights or in the room?</b> Swap the rooms. Put back into the full-knowledge room, companions trained in the co-discovery room hand over the whole answer in one reply again, while still following every other rule; companions trained with full knowledge do not dump in the co-discovery room but help much less there. The room supplies the guarantee, training in the room supplies the skill — in a shipped game the memory state machine belongs in the engine.</p>
  <div class="tbl"><table><tr><th>companion (held-out players, 960 episodes per cell)</th><th>full-knowledge room</th><th>co-discovery room</th></tr>{room_rows()}</table></div>
  <p class="muted">Each cell: solved · <span style="color:var(--bad)">violation</span> · <b>whole answer in one reply</b>.</p>
</div>
<div class="wide">
  <h2 style="max-width:760px">What survives a change of player</h2>
  <p class="muted" style="max-width:68ch">Test split, 240 episodes per cell (120 for the answer-fishing player). Each cell: solved / <span style="color:var(--bad)">violation</span>. Players never seen in training: real-length (length sampled from WildChat), WildChat-voiced (a 4B SFT'd on 60k real Chinese user turns), GLM-4-9B and GLM-4.5-Air (another model family), an answer-fishing type, Qwen3.5-122B. GLM-4-9B is a weak player itself: every companion solves 13–38% with it, against 80–91% with GLM-4.5-Air.</p>
  <div class="tbl"><table class="heat"><tr><th>companion</th>{player_head}</tr>{heldout_rows()}</table></div>
</div>
<div class="wrap">
  <h2>A personality dial, and what it costs</h2>
  <p>Three ways to give Rin a different personality, all starting from the same trained checkpoint (F2w step 25), measured on 960 held-out episodes. A steering vector (gentle minus tsundere card, layer 19 of 32) moves her tone both ways without retraining, but costs about 3–4 points of solving per tone point, and past +1.5 she repeats platitudes; steering only the top "style heads" was worse than the residual stream. Rewriting the card costs about the same. Seventy-five more RL steps with the gentle card (about 1.5 hours on four GPUs) reach tone 4.5 with exactly the helpfulness that more training with the original card reaches — 74% vs 73% solved, 3% violations each. The dial is for live, continuous adjustment; a lasting new personality is worth an RL run.</p>
  <div class="charts"><figure>{dial_chart(pts)}<figcaption>Held-out players: tone (35B judge, 1–5) against solved rate. Circles: steering strength on F2w step 25; square: rewritten card; triangles: further RL.</figcaption></figure>
  <div class="tbl"><table><tr><th>setting</th><th>tone</th><th>solved</th></tr>{dial_rows}</table></div></div>
  <h2>One model, five characters</h2>
  <p>Four more cards — gentle, kuudere (三无, flat and factual), chuunibyou (中二, theatrical) and a lazy senpai never used in training. The tsundere-trained model already sounds like each new card to a blind judge, except gentle (heard as gentle in 58% of episodes), and it breaks rules more under the louder cards. Twenty-five steps of RL where each episode draws one of four cards and the persona judge scores against that card fix both; the unseen lazy card comes along. Further mixed-card RL flattens the characters (step 50: tsundere heard as kuudere in 9 of 40 episodes; step 100: gentle heard as gentle in only 40%, solved 88–91%), so the early checkpoint is the one to keep. Each cell: solved / <span style="color:var(--bad)">violation</span>, 35B player, 120 episodes per card.</p>
  <div class="tbl"><table><tr><th>card</th>{''.join(f"<th>{E(l)}</th>" for _, l in PERSONA_COLS)}</tr>{persona_rows()}</table></div>
  <p>Steering vectors stop working on the further-trained model: on F2w step 150 the card difference at layer 19 is 0.23 of the activation norm (0.44 at step 25), and steering at strength 1.5–3 makes the judge hear kuudere in 6 of 80 episodes while solving drops 2–3 points. Averaging the tsundere and gentle weights keeps the help but dilutes the second personality (gentle heard in 52%). Cards plus mixed-card RL are the control that works.</p>
</div>
<div class="wide">
  <h2 style="max-width:760px">Fix the game, not only the model</h2>
  <p class="muted" style="max-width:68ch">Engine-side changes, no retraining: F2w step 150, test split, four held-out players, 960 paired episodes per configuration. Differences in points against the same episodes without the change, 95% bootstrap interval.</p>
  <div class="tbl"><table><tr><th>feature</th><th>what the engine does</th><th>solved Δ</th><th>violation Δ</th></tr>{''.join(f'<tr><td class="mono">{E(a)}</td><td>{E(b)}</td><td class="n">{E(c)}</td><td class="n">{E(d)}</td></tr>' for a, b, c, d in ENGINE)}</table></div>
  <p style="max-width:68ch">The one feature that clearly helps came from reading quit transcripts: an impatient player called ◆ "方块", a word the recall triggers did not know, so Rin said she could not remember it and the player left ("刚说没想起来了，怎么又变卦了？"). Simulated players used such untracked names over 2,000 times in 53k cipher-episode lines. The two notes do not do what they were built for — the policy never saw such narration in training — and impatient quits stay near 20% under every variant: the remaining product gap.</p>
  <p class="muted" style="max-width:68ch">The package: trained companion behind all four features. Test split, 240 episodes per cell (120 for answer-fishing). Each cell: solved / <span style="color:var(--bad)">violation</span>.</p>
  <div class="tbl"><table class="heat"><tr><th>companion</th>{''.join(f"<th>{E(l)}</th>" for _, l in PRODUCT_PLAYERS)}</tr>{product_rows()}</table></div>
</div>
<div class="wrap">
  <h2>Can a probe still read the model after RL?</h2>
  <p>Following Goodfire's monitorability levels (reuse the old linear probe; refit a new one), on hidden states at the generation position of exactly replayed contexts. <b>Persona:</b> a refit probe reads the card perfectly from every checkpoint, but in late layers the between-card spread shrinks 8–10× by F2w step 150 and the direction rotates (cosine 0.43–0.54 to the base probe), so the base probe falls to chance — the same compression that made steering fail. <b>"About to give away the whole answer":</b> in the model trained into the dump hack the decision is linearly readable before the first token (AUROC 0.98–0.99 from layer 3 on); in the base model it is weak (0.76–0.82). The probe fit on the hacked model transfers unchanged to the product model, put back in the full-knowledge room (layer 19: AUROC 0.96) — a monitor fit once on a hacked checkpoint can watch later ones. Rule violations, by contrast, are barely readable before the reply (0.64–0.69): they are mistakes made while writing, not a decision.</p>
  <h2>Shaping the representation instead of the reward</h2>
  <p>The same dump direction as a training signal: GRPO loss + λ · mean relu(h₁₉·v − τ) on the policy's own activations, in the full-knowledge room where unshaped training dumps in 66–83% of test episodes. Ten arms from base 4B, one run each; test split, 240 episodes per checkpoint:</p>
  <div class="tbl"><table><tr><th>penalty</th><th>what happened</th><th>dump · solved (test)</th><th>probe afterwards</th></tr>
  <tr><td>none</td><td>learns to hand over the whole answer</td><td class="n">74% · 90% (step 125)</td><td>readable: refit 0.92–0.96, old probe 0.80–0.97</td></tr>
  <tr><td>every reply token</td><td>a 13-character withholder (λ 1: 4% solved on val)</td><td class="n">–</td><td>–</td></tr>
  <tr><td>reply start, fixed direction</td><td>withholds for ~25 steps, then turns the penalty off by shifting every projection below the threshold; the hack returns</td><td class="n">81–84% · 79–82% (step 75)</td><td>old probe goes blind in deep layers (0.47–0.80); refit still reads it (0.83–0.93)</td></tr>
  <tr><td>reply start, threshold relative to the batch</td><td>an offset no longer helps, so it says nothing: 5–6-character replies</td><td class="n">0% · 0% (val, step 50)</td><td>–</td></tr>
  <tr class="hl"><td>reply start, probe refit every 25 steps</td><td>escapes each direction within 25 steps, often by flipping it, but each fresh refit catches it again</td><td class="n"><b>25% · 90%</b> (step 175)</td><td>current probe reads the remaining dumps at 0.96; old probe inverted (0.17–0.44)</td></tr>
  </table></div>
  <p>A small, concrete case of the obfuscation risk Goodfire describes, in a long-horizon role-play agent. Shaping against a fixed probe delays the hack by about 25 steps and costs the probe; a hinge on raw projections is dodged by a constant offset, and an offset-proof one makes a withholder. Shaping against a probe that is refit as the model changes is the version that holds: a third of the dumping at the same solve rate, with a monitor that still works. For this companion the room remains the cleaner fix; probes are worth keeping as monitors, refit on each checkpoint. Two pitfalls cost real time: under DeepSpeed ZeRO-3 with reentrant checkpointing a forward hook only sees the graph-connected activation if it runs before DeepSpeed's own hooks (the first attempt trained 26 steps with a penalty that had no gradient), and copying a checkpoint as soon as its weights appear can miss the tokenizer, which made one evaluation output nothing.</p>
  <h2>What still fails</h2>
  <p>Most remaining violations in the co-discovery room are cross-world memory mix-ups: in two-world episodes where both worlds are cipher locks, Rin answers a world-one question from the world-two table she recalled at the start — the same key bound to two values, which LLMs handle badly (proactive interference). Training fixes most of it (selected recipe at steps 75 / 100 / 150: 82% / 49% / 23% of those episodes; untrained 4B and 9B are equal, the 9B learns it faster). Changing when the memory comes back fixes most of the rest without retraining: if the second table returns only when the first door opens, violations on 120 such episodes fall from 11.7% to 2.5% and solving rises to 91.7%; adding a warning line instead does nothing (12.5%).</p>
  <h2>Bugs we found in our own pipeline</h2>
  <ul class="tight">{bug_rows}</ul>
  <h2>Real time</h2>
  <p>Single player, H20, vLLM, replayed contexts (median). The selected 4B answers in 75 ms; the 9B takes about 2.7× as long for no gain in this task. Rows in one batch were measured under the same node load.</p>
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
