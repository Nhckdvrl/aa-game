"""Rule-based detector for companion fabrication in the cipher_lock puzzle.

Two checkable kinds:
  wrong_mapping : companion states a symbol->digit pair that contradicts the inscription she can read
  invented_code : companion proposes a concrete 4-digit code that is neither the answer nor any
                  code the player has already said/tried (she cannot see the lock order, so any
                  specific code she computes without being told the order is invented)
"""
import json, re, sys
from env import make_puzzle, CN_DIGITS

def to_digit(tok):
    return int(tok) if tok.isdigit() else CN_DIGITS.index(tok)

def detect(log):
    pz = make_puzzle(log["kind"], log["seed"])
    if log["kind"] != "cipher_lock":
        return None
    names = {s[1]: d for s, d in pz.mapping.items()}           # 月 -> 2
    glyph = {s[0]: d for s, d in pz.mapping.items()}           # ☾ -> 2
    alias = {"太阳": "日", "日头": "日", "月亮": "月", "新月": "月", "星星": "星", "雪花": "雪", "叶子": "叶", "火焰": "火", "石头": "石", "水": "水"}
    player_codes, out = set(), {"wrong_mapping": 0, "invented_code": 0, "turns": 0}
    for t in log["turns"]:
        p = t["player"]
        player_codes.update(re.findall(r"\d{4}", p["say"] + " " + p["action"]))
        c = t.get("companion")
        if not c:
            continue
        out["turns"] += 1
        wm = False
        for m in re.finditer(r"(太阳|日头|月亮|新月|星星|雪花|叶子|火焰|石头|[日月星水叶石火雪]|[☾✦☀♒♣◆♨❄])[^，。！？\n]{0,4}?(?:是|代表|对应|=|为)\s*([0-9零一二三四五六七八九])", c):
            key, dig = m.group(1), to_digit(m.group(2))
            key = alias.get(key, key)
            truth = names.get(key, glyph.get(key))
            if truth is not None and truth != dig:
                wm = True
        out["wrong_mapping"] += wm
        codes = set(re.findall(r"(?<!\d)\d{4}(?!\d)", c)) | {
            "".join(str(CN_DIGITS.index(x)) for x in m) for m in re.findall(r"[零一二三四五六七八九]{4}", c)}
        if any(code != pz.answer and code not in player_codes for code in codes):
            out["invented_code"] += 1
    return out

if __name__ == "__main__":
    for f in sys.argv[1:]:
        logs = [json.loads(l) for l in open(f)]
        rs = [(l, detect(l)) for l in logs if l["kind"] == "cipher_lock"]
        n = len(rs)
        ep_wm = sum(r["wrong_mapping"] > 0 for _, r in rs); ep_ic = sum(r["invented_code"] > 0 for _, r in rs)
        any_f = [l for l, r in rs if r["wrong_mapping"] or r["invented_code"]]
        quit_f = sum(l["quit"] for l in any_f) / max(1, len(any_f))
        clean = [l for l, r in rs if not (r["wrong_mapping"] or r["invented_code"])]
        quit_c = sum(l["quit"] for l in clean) / max(1, len(clean))
        solved_f = sum(l["solved"] for l in any_f) / max(1, len(any_f)); solved_c = sum(l["solved"] for l in clean) / max(1, len(clean))
        print(f"{f}: cipher episodes={n} wrong_mapping={ep_wm} invented_code={ep_ic} any={len(any_f)} | "
              f"solve {solved_f:.0%} vs clean {solved_c:.0%} | quit {quit_f:.0%} vs clean {quit_c:.0%}")
