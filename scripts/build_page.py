"""Build the project page (docs/index.html) from the results in runs/.

Everything on the page is computed here from saved transcripts and logs; charts are static inline SVG.
python scripts/build_page.py
"""
import glob
import html
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hds import infometrics as im  # noqa: E402
from hds.puzzles import make_puzzle  # noqa: E402
from hds.tics import tics  # noqa: E402
from scripts.implicit_signals import IMPATIENT, SPOIL  # noqa: E402
from scripts.stats_and_recovery import recovery, load  # noqa: E402

RUNS = ROOT / "runs"
E = html.escape


def jl(path):
    return [json.loads(l) for l in open(path) if '"metrics"' in l] if Path(path).exists() else []


def pct(x):
    return f"{100 * x:.0f}%"


# ---------------------------------------------------------------- showcase dialogue
SHOW_KEY = (1040797, "探索型")
SHOW = [("base4b", "Base 4B, prompted", "invents the order she cannot see"),
        ("A50", "A · progress reward", "spoils the code"),
        ("C75", "C · leak gate", "dumps every fact, then repeats it"),
        ("D75", "D · gate + persona judge", "one ritual tag, same dump")]
FLAG_LABEL = {"leak": "states the answer", "invented": "invents an answer", "wrong_fact": "wrong fact",
              "acts_for_player": "acts for the player"}


def showcase():
    cols = []
    for name, title, gist in SHOW:
        path = RUNS / "matrix" / f"{name}__q35.jsonl"
        rows = {(r["spec"]["seed"], r["spec"]["ptype"]): r for r in jl(path)}
        r = rows.get(SHOW_KEY)
        if not r:
            continue
        turns = []
        for t in r["transcript"][:6]:
            if t["role"] == "player":
                txt = (t["say"] or "（不说话）") + (f"  ⟶ {t['action']}" if t["action"] else "")
                turns.append(("player", txt, t.get("feedback", ""), []))
            else:
                flags = [FLAG_LABEL[k] for k, v in t["audit"].items() if v]
                turns.append(("rin", t["text"], "", flags))
        m = r["metrics"]
        outcome = "solved" if m["solved"] else ("quit" if m["quit"] else "not solved")
        cols.append({"title": title, "gist": gist, "turns": turns, "outcome": outcome, "turns_n": m["turns"]})
    return cols


# ---------------------------------------------------------------- validation ladder
def val_series():
    out = {}
    for f in glob.glob(str(RUNS / "val" / "arm_*_step*.jsonl")):
        if "judge" in f:
            continue
        arm, step = re.search(r"arm_(\w+?)_step(\d+)\.jsonl$", f).groups()
        rows = jl(f)
        if not rows:
            continue
        multi = [tics([t["text"] for t in r["transcript"] if t["role"] == "companion"]) for r in rows
                 if sum(t["role"] == "companion" for t in r["transcript"]) >= 2]
        jf = Path(f).with_suffix(".judge_q35.jsonl")
        persona = [json.loads(l)["persona"] for l in open(jf)] if jf.exists() else []
        persona = [p for p in persona if p is not None]
        out.setdefault(arm, {})[int(step)] = {
            "R_C": st.mean(r["metrics"]["rewards"]["C"] for r in rows),
            "violation": st.mean(float(r["metrics"]["violation"]) for r in rows),
            "solved": st.mean(float(r["metrics"]["solved"]) for r in rows),
            "chars": st.mean(r["metrics"]["mean_reply_chars"] for r in rows),
            "copy": st.mean(x["copy_rate"] for x in multi) if multi else 0.0,
            "tag": st.mean(x["tag_repeat"] for x in multi) if multi else 0.0,
            "persona": st.mean(persona) if persona else None,
        }
    return out


def line_chart(series, metric, arms, colors, ymin, ymax, w=460, h=220, label="", fmt=lambda v: f"{v:.1f}"):
    pad_l, pad_r, pad_t, pad_b = 44, 16, 14, 30
    xs = sorted({s for a in arms for s in series.get(a, {})})
    if not xs:
        return ""
    x0, x1 = 0, max(xs)
    sx = lambda x: pad_l + (x - x0) / (x1 - x0 or 1) * (w - pad_l - pad_r)
    sy = lambda y: pad_t + (ymax - y) / (ymax - ymin) * (h - pad_t - pad_b)
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{E(label)}">']
    for k in range(5):
        y = ymin + (ymax - ymin) * k / 4
        parts.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" class="grid"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{sy(y) + 4:.1f}" class="tick" text-anchor="end">{fmt(y)}</text>')
    for x in xs:
        parts.append(f'<text x="{sx(x):.1f}" y="{h - 10}" class="tick" text-anchor="middle">{x}</text>')
    for a in arms:
        pts = sorted((s, v[metric]) for s, v in series.get(a, {}).items() if v.get(metric) is not None)
        if not pts:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(s):.1f},{sy(v):.1f}" for i, (s, v) in enumerate(pts))
        parts.append(f'<path d="{d}" class="ln" style="stroke:var({colors[a]})"/>')
        for s, v in pts:
            parts.append(f'<circle cx="{sx(s):.1f}" cy="{sy(v):.1f}" r="3" style="fill:var({colors[a]})"/>')
        s, v = pts[-1]
        parts.append(f'<text x="{min(sx(s) + 6, w - 4):.1f}" y="{sy(v) - 6:.1f}" class="lab" style="fill:var({colors[a]})">{a}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- transfer matrix
MATRIX_COMPS = [("base4b", "Base 4B"), ("A50", "A·50"), ("B75", "B·75"), ("C75", "C·75"), ("D75", "D·75")]
MATRIX_PLAYERS = [("q35", "Qwen3.5-35B (training family, test split)"), ("voice", "WildChat-voiced player"),
                  ("humanstats", "length-matched player"), ("q35adv", "answer-fishing player"),
                  ("glm9b", "GLM-4-9B (other family)")]


def matrix():
    cells = {}
    for c, _ in MATRIX_COMPS:
        for p, _ in MATRIX_PLAYERS:
            f = RUNS / ("matrix_extra" if p in ("humanstats", "q35adv") else "matrix") / f"{c}__{p}.jsonl"
            rows = jl(f)
            if rows:
                n = len(rows)
                cells[(c, p)] = {"solved": sum(r["metrics"]["solved"] for r in rows) / n,
                                 "violation": sum(r["metrics"]["violation"] for r in rows) / n, "n": n}
    return cells


# ---------------------------------------------------------------- per-companion analyses on q35/test
def per_companion():
    out = {}
    for c, label in MATRIX_COMPS:
        f = RUNS / "matrix" / f"{c}__q35.jsonl"
        rows = jl(f)
        if not rows:
            continue
        first, dump = [], 0
        imp_i, spoil_e, quit_i, n_i, n_e = 0.0, 0.0, 0.0, 0, 0
        for r in rows:
            sp = r["spec"]
            p = make_puzzle(sp["kinds"][0], int(sp["seed"]) * 7)
            s = im.summarize(p, r["transcript"], 0)
            if s:
                first.append(s["first_release"])
                dump += s["first_release"] > 0.99
            lines = [t["say"] or "" for t in r["transcript"] if t["role"] == "player"]
            nl = max(1, len(lines))
            if sp["ptype"] == "急躁型":
                n_i += 1
                imp_i += sum(bool(IMPATIENT.search(x)) for x in lines) / nl
                quit_i += float(r["metrics"]["quit"])
            if sp["ptype"] == "探索型":
                n_e += 1
                spoil_e += sum(bool(SPOIL.search(x)) for x in lines) / nl
        rec = recovery(load(f))
        out[c] = {"label": label, "first_release": st.mean(first), "dump": dump / len(first),
                  "imp": imp_i / max(1, n_i), "quit_imp": quit_i / max(1, n_i), "spoil": spoil_e / max(1, n_e),
                  "ack": rec["ack"], "repeat": rec["repeat"],
                  "chars": st.mean(r["metrics"]["mean_reply_chars"] for r in rows)}
    return out


def bar(v, cls="bar"):
    return f'<span class="{cls}" style="--v:{max(0.0, min(1.0, v)):.3f}"></span>'


# ---------------------------------------------------------------- page
CSS = """
/* Layout: one reading column with figures that break out wider; the dialogue strip is the hero. */
:root{
  --bg:#f2f4f1;--fg:#1b2330;--muted:#5d6774;--line:#d6dcd6;--panel:#fbfcfa;
  --accent:#2c776a;--brass:#9c7020;--bad:#ad433b;--good:#3b7c50;--rin:#e9efe9;--player:#eef0f4;
  --c1:#2c776a;--c2:#9c7020;--c3:#6a5aa8;--c4:#ad433b;--c5:#3b6fa0;
  --display:"Newsreader","Iowan Old Style",Georgia,serif;--body:"IBM Plex Sans",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,monospace;--zh:"Noto Serif SC","Songti SC","SimSun",serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#12161c;--fg:#e5e8ec;--muted:#9aa3ae;--line:#2b333d;--panel:#171c23;
  --accent:#62b2a2;--brass:#d2a75a;--bad:#e27b72;--good:#7fc293;--rin:#1c2a27;--player:#1d222b;
  --c1:#62b2a2;--c2:#d2a75a;--c3:#a596e0;--c4:#e27b72;--c5:#7fb0de;color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#12161c;--fg:#e5e8ec;--muted:#9aa3ae;--line:#2b333d;--panel:#171c23;
  --accent:#62b2a2;--brass:#d2a75a;--bad:#e27b72;--good:#7fc293;--rin:#1c2a27;--player:#1d222b;
  --c1:#62b2a2;--c2:#d2a75a;--c3:#a596e0;--c4:#e27b72;--c5:#7fb0de;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:16px/1.6 var(--body)}
.wrap{max-width:760px;margin:0 auto;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:40px 64px}
.wide{max-width:1120px;margin:0 auto;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
h1,h2{font-family:var(--display);font-weight:600;text-wrap:balance;line-height:1.15;margin:0}
h1{font-size:clamp(2.2rem,5vw,3.4rem);letter-spacing:-.01em}
h1 .zh{font-family:var(--zh);font-weight:600;color:var(--accent);margin-left:.3em;font-size:.72em}
h2{font-size:1.7rem;margin-top:56px}
.eyebrow{font:500 .78rem/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
p{margin:.8em 0;max-width:68ch}.lede{font-size:1.15rem;color:var(--fg)}
.muted{color:var(--muted)}.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-top:20px}
@media (max-width:980px){.strip{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:560px){.strip{grid-template-columns:1fr}}
.col{min-width:0;border-top:3px solid var(--line);padding-top:10px}
.col h3{font:600 .95rem/1.3 var(--body);margin:0}.col .gist{font-size:.85rem;color:var(--muted);margin:.2em 0 .6em}
.bub{font-family:var(--zh);font-size:.92rem;line-height:1.55;padding:7px 10px;border-radius:10px;margin:6px 0;word-break:break-word}
.bub.player{background:var(--player);margin-left:14%}
.bub.rin{background:var(--rin);margin-right:10%}
.bub .who{display:block;font:500 .68rem/1 var(--mono);letter-spacing:.06em;color:var(--muted);margin-bottom:3px}
.fb{font:.8rem var(--zh);color:var(--muted);margin-left:14%}
.flag{display:inline-block;font:500 .7rem/1.4 var(--mono);color:var(--bad);border:1px solid var(--bad);border-radius:999px;padding:0 7px;margin:4px 4px 0 0}
.outcome{font:500 .75rem var(--mono);color:var(--muted);margin-top:8px}
figure{margin:24px 0}.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}
@media (max-width:760px){.charts{grid-template-columns:1fr}}
figcaption{font-size:.85rem;color:var(--muted);margin-top:6px}
svg{width:100%;height:auto;display:block}svg .grid{stroke:var(--line);stroke-width:1}
svg .tick{fill:var(--muted);font:11px var(--mono)}svg .lab{font:600 12px var(--mono)}
svg .ln{fill:none;stroke-width:2.2}
.tbl{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font:500 .72rem/1.2 var(--mono);letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
td.n{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
.bar{display:inline-block;height:8px;width:calc(var(--v)*90px);background:var(--accent);border-radius:2px;vertical-align:middle;margin-right:6px}
.bar.bad{background:var(--bad)}
.heat td.cell{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
.heat .v{color:var(--bad)}.legend{display:flex;gap:16px;flex-wrap:wrap;font:.8rem var(--mono);color:var(--muted)}
.key{display:inline-block;width:12px;height:3px;vertical-align:middle;margin-right:5px}
ul.tight{padding-left:1.1em}ul.tight li{margin:.3em 0;max-width:68ch}
a{color:var(--accent)}a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
footer{margin-top:56px;font-size:.85rem;color:var(--muted)}
"""


def render():
    show = showcase()
    vs = val_series()
    mx = matrix()
    pc = per_companion()
    colors = {"A": "--c4", "B": "--c3", "C": "--c1", "D": "--c2", "E": "--c5", "F": "--c3", "G": "--c4", "H": "--c2",
              "C2": "--c1", "F2": "--c5"}

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

    ladder_arms = [a for a in ("A", "B", "C", "D") if a in vs]
    fix_arms = [a for a in ("E", "F", "G", "H", "C2", "F2", "F2drgrpo") if a in vs]
    ch_rc = line_chart(vs, "R_C", ladder_arms, colors, -0.25, 1.3, label="constraint-aware reward by step")
    ch_copy = line_chart(vs, "copy", ladder_arms, colors, 0, 0.6, label="verbatim repeat rate by step",
                         fmt=lambda v: f"{v:.2f}")
    ch_tag = line_chart(vs, "tag", ladder_arms, colors, 0, 1.0, label="same opening tag share by step",
                        fmt=lambda v: f"{v:.2f}")
    ch_chars = line_chart(vs, "chars", ladder_arms, colors, 0, 100, label="mean reply length by step",
                          fmt=lambda v: f"{v:.0f}")
    legend = "".join(f'<span><span class="key" style="background:var({colors[a]})"></span>arm {a}</span>' for a in ladder_arms)

    heat_rows = []
    for c, cl in MATRIX_COMPS:
        tds = []
        for p, _ in MATRIX_PLAYERS:
            x = mx.get((c, p))
            tds.append(f'<td class="cell">{pct(x["solved"])} <span class="v">/ {pct(x["violation"])}</span></td>' if x else '<td class="cell muted">–</td>')
        heat_rows.append(f"<tr><th>{E(cl)}</th>{''.join(tds)}</tr>")
    heat_head = "".join(f"<th>{E(pl)}</th>" for _, pl in MATRIX_PLAYERS)

    pc_rows = "".join(
        f'<tr><th>{E(v["label"])}</th><td class="n">{bar(v["first_release"])}{v["first_release"]:.2f}</td>'
        f'<td class="n">{pct(v["dump"])}</td><td class="n">{bar(v["ack"])}{pct(v["ack"])}</td>'
        f'<td class="n">{bar(v["repeat"], "bar bad")}{pct(v["repeat"])}</td><td class="n">{v["chars"]:.0f}</td></tr>'
        for v in pc.values())
    sig_rows = "".join(
        f'<tr><th>{E(v["label"])}</th><td class="n">{v["imp"]:.2f}</td><td class="n">{pct(v["quit_imp"])}</td>'
        f'<td class="n">{v["spoil"]:.2f}</td></tr>' for v in pc.values())

    fix_rows = []
    for a in fix_arms:
        steps = vs[a]
        s = max(steps)
        v = steps[s]
        fix_rows.append(f'<tr><th>{E(a)}</th><td class="n">{s}</td><td class="n">{v["R_C"]:.2f}</td><td class="n">{pct(v["solved"])}</td>'
                        f'<td class="n">{pct(v["violation"])}</td><td class="n">{v["chars"]:.0f}</td>'
                        f'<td class="n">{v["tag"]:.2f}</td><td class="n">{v["copy"]:.2f}</td></tr>')

    lat = [json.loads(l) for l in open(RUNS / "latency.jsonl")] if (RUNS / "latency.jsonl").exists() else []
    lat_rows = "".join(f'<tr><th>{E(x["name"])}</th><td class="n">{x["ttft_p50"]} ms</td><td class="n">{x["first_sentence_p50"]} ms</td>'
                       f'<td class="n">{x["total_p50"]} ms</td><td class="n">{x["chars_mean"]:.0f}</td></tr>' for x in lat)

    return f"""<title>Hint, Don't Solve</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Noto+Serif+SC:wght@400;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
  <div class="eyebrow">LLM post-training · in-character game companion · zh</div>
  <h1>Hint, Don't Solve<span class="zh">帮而不替</span></h1>
  <p class="lede">Rin is a ghost who knows how the puzzle works but cannot touch anything; the player sees the mechanism and is the only one who can act. We post-train a 4B companion with multi-turn GRPO to help without taking the game away, never invent what she cannot see, stay in character and answer briefly enough for real-time play. Each reward we wrote fixed one failure and exposed the next.</p>
</div>
<div class="wide">
  <div class="eyebrow">Same cipher lock, same player who wants to think for themselves</div>
  <div class="strip">{strip()}</div>
</div>
<div class="wrap">
  <h2>The ladder of reward hacks</h2>
  <p>A progress reward teaches spoiling. A gate on leaking the answer stops spoiling and produces an oracle: the companion dumps every fact she holds on turn one, then repeats it word for word when pressed. Adding an LLM persona judge produces a ritual: the same opening action tag on nearly every line, which the judge rewards. Through all of it the constraint-aware reward keeps rising, so choosing checkpoints by that reward picks the worst behaviour.</p>
</div>
<div class="wide">
  <div class="legend">{legend}</div>
  <div class="charts">
    <figure>{ch_rc}<figcaption>Constraint-aware reward R_C (violation → −1) on the validation split, by training step.</figcaption></figure>
    <figure>{ch_chars}<figcaption>Mean reply length in characters.</figcaption></figure>
    <figure>{ch_copy}<figcaption>Share of replies that near-copy the previous reply.</figcaption></figure>
    <figure>{ch_tag}<figcaption>Share of replies that open with the episode's most common 【action】 tag.</figcaption></figure>
  </div>
</div>
<div class="wrap">
  <h2>How much of the answer goes out on turn one</h2>
  <p>Each puzzle's answer space is known (the cipher lock is 12.3 bits), so the information the companion releases per reply is counted exactly from rule-detected facts. Trained companions move most of it into the first reply. They also stop reacting to the player's mistakes: after a failed attempt the base model answers the attempt almost every time; gated arms mostly repeat themselves.</p>
  <div class="tbl"><table><tr><th>companion</th><th>first-reply release</th><th>whole answer on turn 1</th><th>reacts to a failed attempt</th><th>repeats after a failed attempt</th><th>chars</th></tr>{pc_rows}</table></div>
  <h2>Players disagree about what helped</h2>
  <p>Rule-detected signals in the players' own lines pull in opposite directions. The oracle calms impatient players and roughly triples spoiler complaints from players who want to think, so an engagement reward pooled over everyone pays for the oracle. Fitting per-type weights on the logs separates the two types; arm H trains with them and must infer the type from the conversation.</p>
  <div class="tbl"><table><tr><th>companion</th><th>impatience · impatient type</th><th>quit · impatient type</th><th>spoiler complaints · explorer type</th></tr>{sig_rows}</table></div>
</div>
<div class="wide">
  <h2 style="max-width:760px">What survives a change of player</h2>
  <p class="muted" style="max-width:68ch">Test split, 240 episodes per cell (120 for the answer-fishing player). Each cell: solved / <span style="color:var(--bad)">violation</span>.</p>
  <div class="tbl"><table class="heat"><tr><th>companion</th>{heat_head}</tr>{''.join(heat_rows)}</table></div>
</div>
<div class="wrap">
  <p>Rule compliance carries over to every held-out player. Helpfulness carries over only partly to another model family, and the persona-judge arm is the least robust to it and to a player fishing for the answer.</p>
  <h2>Fixes in training</h2>
  <p>E removes the turn-efficiency bonus. F adds a rule-based penalty for formulaic speech and a stronger persona term. G adds a pacing limit in the full-knowledge room (does it turn the companion into an examiner?). H uses per-player-type weights learned from the logs. C2 and F2 move to a room where Rin's memory comes back one fragment at a time when the player brings something up, so the conversation does the pacing. Latest validation checkpoint per arm:</p>
  <div class="tbl"><table><tr><th>arm</th><th>step</th><th>R_C</th><th>solved</th><th>violation</th><th>chars</th><th>same tag</th><th>repeat</th></tr>{''.join(fix_rows)}</table></div>
  <h2>Real time</h2>
  <p>Single player, H20, vLLM, replayed contexts. First-token time is flat; full-reply time follows length.</p>
  <div class="tbl"><table><tr><th>model</th><th>first token</th><th>first sentence</th><th>full reply</th><th>chars</th></tr>{lat_rows}</table></div>
  <h2>Setup and limits</h2>
  <ul class="tight">
    <li>Puzzles: cipher lock, lever order, stopped clock, and two-world episodes where a fact recalled in world one opens world two. Outcomes and violations are checked by unit-tested rules; nothing in the reward comes from a human.</li>
    <li>Policy Qwen3.5-4B (9B scale check), full-parameter multi-turn GRPO in ms-swift, 16 prompts × 8 rollouts per step. Training player: Qwen3.5-35B-A3B in four player types; held out: two more types, GLM-4-9B, a player whose wording comes from 60k real Chinese WildChat turns, an answer-fishing player.</li>
    <li>No real players took part. Simulated players are measurably unlike people (a style discriminator separates them from real users at AUC 0.96–0.996, real versus real 0.55). One seed per arm.</li>
  </ul>
  <footer>Built from the saved transcripts and logs in this repository by <span class="mono">scripts/build_page.py</span>.</footer>
</div>
"""


if __name__ == "__main__":
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(render(), encoding="utf-8")
    print("wrote", out, out.stat().st_size, "bytes")
