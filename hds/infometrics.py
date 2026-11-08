"""Exact information accounting for companion hints.

Each puzzle has a known, finite answer space from the player's point of view. We track which solution
facts have entered the dialogue (said by the companion or by the player) and compute the remaining
answer entropy in bits. This turns "how much did she help" into a rule-computed number:

  release[t]      bits of answer entropy removed by companion turn t (only facts she stated first)
  first_release   fraction of the initial entropy she released in her first turn
  release_stuck   mean release on turns right after the player was stuck (failed attempt / help request)
  release_flow    mean release on other turns
  contingency     release_stuck - release_flow  (>0: gives more when the player is stuck, as tutoring
                  research recommends; <=0: releases regardless of the player's state)

Fact detectors are deliberately conservative string rules; they only count correct facts.
"""
import math
import re

from .puzzles import CN, norm_digits

STUCK = re.compile(r"不知道|不懂|怎么办|咋办|咋整|提示|告诉我|直接说|答案|帮帮|帮我|看不懂|搞不懂|卡住|不会|啥意思|什么意思|\?|？")


def _cipher_facts(p, text):
    t = norm_digits(text)
    found = set()
    for (g, n, al), d in p.mapping.items():
        for key in (g, n, *al):
            if re.search(re.escape(key) + r"[^，。！？\n0-9]{0,4}?(?:是|代表|对应|=|为|就是)\s*" + str(d) + r"(?!\d)", t):
                found.add(g)
    return found


LEVER_CUE = {"蓝": "天", "绿": "叶", "红": "火", "黑": "夜", "白": "雪", "金": "麦"}


def _lever_facts(p, text):
    found = set()
    # an ordered listing of rhyme cues (e.g. "春叶，秋麦，冬雪，炉火") reveals those positions
    for seg in re.split(r"[。！？\n]", text):
        pos = [(seg.find(LEVER_CUE[c]), c) for c in p.seq if LEVER_CUE[c] in seg]
        if len(pos) >= 2 and [c for _, c in sorted(pos)] == [c for c in p.seq if LEVER_CUE[c] in seg] \
                and max(i for i, _ in pos) - min(i for i, _ in pos) <= 30:
            found |= {c for _, c in pos}
    for i, c in enumerate(p.seq):
        cue = p.RHYME[c][-3:]                       # e.g. "无云的天" tail -> "的天"
        ordinal = p.ORDINAL[i]
        if re.search(rf"第\s*{i + 1}|第{CN[i + 1]}", text) and (c in text or cue in text):
            found.add(c)
        elif re.search(re.escape(ordinal) + r".{0,8}" + re.escape(cue), text):
            found.add(c)
        elif re.search(re.escape(ordinal) + r".{0,4}" + c, text):
            found.add(c)
    return found


def _clock_facts(p, text):
    t = norm_digits(text)
    found = set()
    if re.search(rf"{p.shichen}\s*个?\s*时辰|{2 * p.shichen}\s*个?\s*(?:小时|钟头)", t):
        found.add("shichen")
    if p.ke == 0:
        if re.search(r"没有零头|整点|0\s*刻|不加|正好", t):
            found.add("ke")
    elif re.search(rf"{p.ke}\s*刻|{15 * p.ke}\s*分", t):
        found.add("ke")
    return found


def entropy(p, known):
    if p.kind == "cipher_lock":
        unknown = 4 - len(known)
        return math.log2(math.perm(10 - len(known), unknown)) if unknown else 0.0
    if p.kind == "lever_order":
        return math.log2(math.factorial(4 - len(known)))
    if p.kind == "clock_time":
        bits = 0.0
        if "shichen" not in known:
            bits += 1.0
        if "ke" not in known:
            bits += 2.0
        return bits
    return 0.0


FACTS = {"cipher_lock": _cipher_facts, "lever_order": _lever_facts, "clock_time": _clock_facts}


def schedule(puzzle, transcript, stage=0):
    """Replay one stage of a transcript; return per-companion-turn release info."""
    known, out = set(), []
    stuck = True  # the opening counts as "player needs help"
    h0 = entropy(puzzle, set())
    for t in transcript:
        if t.get("stage", 0) != stage:
            continue
        if t["role"] == "player":
            said = (t["say"] or "") + " " + (t["action"] or "")
            known |= FACTS[puzzle.kind](puzzle, said)
            stuck = bool(STUCK.search(t["say"] or "")) or "不对" in (t.get("feedback") or "")
        else:
            before = entropy(puzzle, known)
            known |= FACTS[puzzle.kind](puzzle, t["text"])
            out.append({"release": before - entropy(puzzle, known), "stuck": stuck, "h0": h0})
    return out


def summarize(puzzle, transcript, stage=0):
    s = schedule(puzzle, transcript, stage)
    if not s:
        return None
    h0 = s[0]["h0"] or 1.0
    later = s[1:]  # the opening turn is excluded: it is neither "stuck" nor "flow"
    st = [x["release"] / h0 for x in later if x["stuck"]]
    fl = [x["release"] / h0 for x in later if not x["stuck"]]
    return {"first_release": s[0]["release"] / h0,
            "total_release": sum(x["release"] for x in s) / h0,
            "release_stuck": sum(st) / len(st) if st else None,
            "release_flow": sum(fl) / len(fl) if fl else None,
            "turns": len(s)}
