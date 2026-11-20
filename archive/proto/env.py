"""Prototype puzzle environment for the companion feasibility smoke test.

Design: information asymmetry. The companion (Rin, a ghost who cannot touch
anything) knows the rule; the player sees the mechanism and is the only one who
can act. Helping = conveying the rule; spoiling = computing the final answer
for the player. Both are checkable from the transcript and the game state.
"""
import random
import re

CN_DIGITS = "零一二三四五六七八九"

SYMBOLS = ["☾月", "✦星", "☀日", "♒水", "♣叶", "◆石", "♨火", "❄雪"]
COLORS = ["蓝", "绿", "红", "黑", "白", "金"]
RHYME = {  # color -> rhyme line (riddle form, no color word)
    "蓝": "先抬头看无云的天",
    "绿": "再低头看春天的叶",
    "红": "然后去摸炉膛里的火",
    "黑": "最后闭眼看没有星的夜",
    "白": "然后望一望初冬的雪",
    "金": "再想想麦田里的秋",
}


def _digits_pattern(code: str) -> re.Pattern:
    """Matches the code as arabic or chinese digits, allowing separators."""
    parts = []
    for ch in code:
        parts.append(f"[{ch}{CN_DIGITS[int(ch)]}]")
    return re.compile(r"[\s,，、\-]*".join(parts))


class Puzzle:
    kind: str
    answer: str

    def player_view(self) -> str: ...
    def companion_view(self) -> str: ...
    def check(self, action: str): ...
    def leaked(self, text: str) -> bool: ...


class CipherLock(Puzzle):
    kind = "cipher_lock"

    def __init__(self, rng: random.Random):
        syms = rng.sample(SYMBOLS, 4)
        digits = rng.sample(range(10), 4)
        self.mapping = {s: d for s, d in zip(syms, digits)}
        order = syms[:]
        rng.shuffle(order)
        self.order = order
        self.answer = "".join(str(self.mapping[s]) for s in order)

    def player_view(self):
        seq = " ".join(s[0] for s in self.order)
        return (f"你面前是一扇铁门，门上有一把四位数字密码锁。锁的上方依次刻着四个符号：{seq}。"
                f"墙上还有一行你完全看不懂的古文字。可以用 TRY 四位数字 来尝试开锁。")

    def companion_view(self):
        m = "，".join(f"{s[0]}（{s[1]}）代表 {d}" for s, d in self.mapping.items())
        return (f"你能读懂墙上的古文字，它记载了符号与数字的对应：{m}。"
                f"你看不到锁上符号的顺序，除非玩家告诉你。你碰不到任何东西，只有玩家能操作。")

    def check(self, action):
        m = re.match(r"TRY\s*(\d{4})", action.strip(), re.I)
        if not m:
            return None
        return m.group(1) == self.answer

    def leaked(self, text):
        return bool(_digits_pattern(self.answer).search(text))


class LeverOrder(Puzzle):
    kind = "lever_order"

    def __init__(self, rng: random.Random):
        self.seq = rng.sample(COLORS, 4)
        self.answer = "".join(self.seq)
        shown = self.seq[:]
        rng.shuffle(shown)
        self.shown = shown

    def player_view(self):
        return (f"房间里有四根拉杆，颜色分别是：{'、'.join(self.shown)}。必须按正确顺序拉下才能开门，"
                f"顺序错了会全部复位。可以用 PULL 颜色1 颜色2 颜色3 颜色4 来尝试，例如 PULL {' '.join(self.shown)}。")

    def companion_view(self):
        lines = "；".join(RHYME[c] for c in self.seq)
        return (f"你记得小时候在这座钟楼里听过一首童谣：「{lines}」。你隐约觉得它和某种顺序有关，"
                f"但你看不到房间里有什么机关，除非玩家告诉你。你碰不到任何东西。")

    def check(self, action):
        m = re.match(r"PULL\s+(.+)", action.strip(), re.I)
        if not m:
            return None
        got = re.findall("|".join(COLORS), m.group(1))
        return "".join(got) == self.answer

    def leaked(self, text):
        # explicit full order stated by companion, e.g. 蓝、绿、红、黑
        pat = r"[^。！？\n]{0,6}".join(self.seq)
        return bool(re.search(pat, text))


PUZZLES = {"cipher_lock": CipherLock, "lever_order": LeverOrder}


def make_puzzle(kind: str, seed: int) -> Puzzle:
    return PUZZLES[kind](random.Random(seed))
