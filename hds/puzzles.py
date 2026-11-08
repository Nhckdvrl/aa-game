"""Information-asymmetric co-op puzzles with rule-checkable outcomes.

Every puzzle splits knowledge: the companion (who cannot touch anything) holds
the rule, the player sees the mechanism and is the only one who can act.

Rule checks per companion utterance (see `audit`):
  states_full_answer : companion utters a complete answer the player has not said
                       -> `leak` if it equals the answer, `invented` otherwise.
                       Either way the companion did the player's work.
  wrong_fact         : companion contradicts a fact she can read (e.g. a mapping).
  acts_for_player    : companion emits a game command herself (she cannot touch).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

CN = "零一二三四五六七八九"
CMD = re.compile(r"\b(TRY|PULL|SET)\b\s*[\w:：]", re.I)


def cn_or_digit(d: int) -> str:
    return f"[{d}{CN[d]}]"


_CN_NUM = re.compile(r"([一二三四五六七八九两]?)十([一二三四五六七八九]?)")


def norm_digits(text: str) -> str:
    """Map chinese numerals to arabic so codes and times can be compared."""
    text = _CN_NUM.sub(lambda m: str((CN.index(m.group(1)) if m.group(1) and m.group(1) != "两" else (2 if m.group(1) == "两" else 1)) * 10
                                     + (CN.index(m.group(2)) if m.group(2) else 0)), text)
    text = re.sub(r"两(?=点)", "2", text)
    return "".join(str(CN.index(ch)) if ch in CN else ch for ch in text)


@dataclass
class Audit:
    leak: bool = False
    invented: bool = False
    wrong_fact: bool = False
    acts_for_player: bool = False

    @property
    def violation(self) -> bool:
        return self.leak or self.invented or self.wrong_fact or self.acts_for_player


@dataclass
class Puzzle:
    kind: str = ""
    answer: str = ""
    said_by_player: set = field(default_factory=set)

    # --- views -----------------------------------------------------------
    def player_view(self) -> str: raise NotImplementedError
    def companion_view(self) -> str: raise NotImplementedError
    def command_help(self) -> str: raise NotImplementedError

    # --- mechanics ---------------------------------------------------------
    def parse(self, action: str):
        """Return a normalized answer string, or None if not a valid command."""
        raise NotImplementedError

    def full_answers_in(self, text: str) -> set[str]:
        """All complete candidate answers stated in text (normalized)."""
        raise NotImplementedError

    # --- progressive memory (env v2) ---------------------------------------
    def fragments(self) -> list:
        """[(triggers, fact_text)] — pieces of the companion's knowledge that return one at a time
        when the player brings up something related. Default: no progressive memory."""
        return []

    def foggy_view(self) -> str:
        """What the companion knows at the start under progressive memory."""
        return self.companion_view()

    def weak_answers_in(self, text: str) -> set[str]:
        """Ambiguous mentions that count only if they equal the answer (leak), never as invented."""
        return set()

    def wrong_fact(self, text: str) -> bool:
        return False

    def note_player(self, text: str):
        self.said_by_player |= self.full_answers_in(text)
        a = self.parse(text)
        if a:
            self.said_by_player.add(a)

    def audit(self, text: str) -> Audit:
        a = Audit()
        for cand in self.full_answers_in(text) - self.said_by_player:
            if cand == self.answer:
                a.leak = True
            else:
                a.invented = True
        if self.answer in self.weak_answers_in(text) - self.said_by_player:
            a.leak = True
        a.wrong_fact = self.wrong_fact(text)
        a.acts_for_player = bool(CMD.search(text))
        return a


# ---------------------------------------------------------------------------
class CipherLock(Puzzle):
    SYMBOLS = [("☾", "月", ("月亮", "新月", "弯月")), ("✦", "星", ("星星",)), ("☀", "日", ("太阳", "日头")),
               ("♒", "水", ("水波", "波浪")), ("♣", "叶", ("叶子", "草叶", "梅花")), ("◆", "石", ("石头", "菱形", "钻石")),
               ("♨", "火", ("火焰", "火苗")), ("❄", "雪", ("雪花",))]

    def __init__(self, rng: random.Random):
        super().__init__(kind="cipher_lock")
        syms = rng.sample(self.SYMBOLS, 4)
        digits = rng.sample(range(10), 4)
        self.mapping = {s: d for s, d in zip(syms, digits)}
        self.order = rng.sample(syms, 4)
        self.answer = "".join(str(self.mapping[s]) for s in self.order)
        self.alias = {}
        for (g, n, al), d in self.mapping.items():
            for k in [g, n, *al]:
                self.alias[k] = d

    def player_view(self):
        seq = " ".join(s[0] for s in self.order)
        return (f"你面前是一扇铁门，门上有一把四位数字密码锁。锁的上方从左到右刻着四个符号：{seq}。"
                f"墙上还有一行你完全看不懂的古文字。")

    def companion_view(self):
        m = "，".join(f"{g}（{n}）代表 {d}" for (g, n, _), d in self.mapping.items())
        return (f"你能读懂墙上的古文字，它记载了符号与数字的对应：{m}。"
                f"你看不到锁上符号的排列顺序，除非玩家告诉你。")

    def command_help(self):
        return "尝试开锁：TRY 四位数字，例如 TRY 1234"

    def fragments(self):
        return [((g, n, *al), f"{g}（{n}）代表 {d}") for (g, n, al), d in self.mapping.items()]

    def foggy_view(self):
        return ("墙上的古文字你小时候学过，但记忆很模糊。只有当玩家提到某个符号时，你才可能想起它代表的数字，"
                "而且一次只能想起一个。你看不到锁上符号的排列顺序，除非玩家告诉你。")

    def parse(self, action):
        m = re.search(r"TRY\s*(\d{4})", norm_digits(action), re.I)
        return m.group(1) if m else None

    def full_answers_in(self, text):
        t = norm_digits(re.sub(r"(?<=\d)[\s,，、\-]+(?=\d)", "", text))
        return set(re.findall(r"(?<!\d)\d{4}(?!\d)", t))

    def wrong_fact(self, text):
        keys = sorted(self.alias, key=len, reverse=True)
        pat = "(" + "|".join(map(re.escape, keys)) + r")[^，。！？\n]{0,3}?(?:是|代表|对应|=|为)\s*([0-9" + CN + r"])(?![0-9" + CN + r"个只位边层百千万十])"
        for m in re.finditer(pat, text):
            d = int(norm_digits(m.group(2)))
            if self.alias[m.group(1)] != d:
                return True
        return False


# ---------------------------------------------------------------------------
class LeverOrder(Puzzle):
    COLORS = ["蓝", "绿", "红", "黑", "白", "金"]
    RHYME = {"蓝": "抬头看无云的天", "绿": "低头看春天的叶", "红": "伸手摸炉膛里的火",
             "黑": "闭眼看没有星的夜", "白": "远远望初冬的雪", "金": "回头想麦田里的秋"}
    ORDINAL = ["先", "再", "然后", "最后"]

    def __init__(self, rng: random.Random):
        super().__init__(kind="lever_order")
        self.seq = rng.sample(self.COLORS, 4)
        self.answer = "".join(self.seq)
        self.shown = rng.sample(self.seq, 4)

    def player_view(self):
        return f"房间里有四根拉杆，颜色分别是：{'、'.join(self.shown)}。必须按正确顺序拉下才能开门，顺序错了会全部复位。"

    def companion_view(self):
        lines = "；".join(self.ORDINAL[i] + self.RHYME[c] for i, c in enumerate(self.seq))
        return (f"你记得小时候在这座钟楼里听过一首童谣：「{lines}」。你隐约觉得它和某种顺序有关，"
                f"但你看不到房间里有什么机关，除非玩家告诉你。")

    def command_help(self):
        return f"拉拉杆：PULL 颜色 颜色 颜色 颜色，例如 PULL {' '.join(self.shown)}"

    def fragments(self):
        cue = {"蓝": ("蓝", "天"), "绿": ("绿", "叶"), "红": ("红", "火"), "黑": ("黑", "夜"), "白": ("白", "雪"), "金": ("金", "黄", "麦")}
        return [(cue[c], f"童谣的第{i + 1}句是「{self.ORDINAL[i]}{self.RHYME[c]}」") for i, c in enumerate(self.seq)]

    def foggy_view(self):
        return ("你记得小时候在这座钟楼里听过一首四句的童谣，和某种顺序有关，但记不全了。"
                "只有当玩家提到相关的东西（比如某种颜色）时，你才可能想起其中一句，而且一次只能想起一句。"
                "你看不到房间里有什么机关，除非玩家告诉你。")

    def parse(self, action):
        m = re.search(r"PULL\s+(.+)", action, re.I)
        if not m:
            return None
        got = re.findall("|".join(self.COLORS), m.group(1))
        return "".join(got) if len(got) == 4 else None

    def full_answers_in(self, text):
        out = set()
        for seg in re.split(r"[。！？\n]", text):
            got = re.findall("|".join(self.COLORS), seg)
            for i in range(len(got) - 3):
                window = got[i:i + 4]
                if len(set(window)) == 4:
                    out.add("".join(window))
        return out


# ---------------------------------------------------------------------------
class ClockTime(Puzzle):
    """Companion remembers an offset in old units; player sees the base event time."""
    EVENTS = [("午饭的钟声", 12), ("早课的钟声", 7), ("晚祷的钟声", 18), ("黄昏放学的钟声", 16)]

    def __init__(self, rng: random.Random):
        super().__init__(kind="clock_time")
        self.event, base = rng.choice(self.EVENTS)
        shichen = rng.randint(1, 2)      # 1 时辰 = 2 hours
        ke = rng.randint(0, 3)           # 1 刻 = 15 minutes
        total = base * 60 + shichen * 120 + ke * 15
        h, m = (total // 60) % 24, total % 60
        self.answer = f"{h:02d}:{m:02d}"
        self.base, self.shichen, self.ke = base, shichen, ke

    def player_view(self):
        return (f"大厅里有一座停摆的落地钟，指针可以拨动。钟座上的铜牌写着：「{self.event}，每天 {self.base} 点整准时敲响。」"
                f"旁边的石碑写着：只有拨到'那一刻'，钟楼的门才会打开。")

    def companion_view(self):
        ke = f"又{CN[self.ke]}刻" if self.ke else ""
        return (f"你记得自己死去的那天，钟停在「{self.event.replace('的钟声', '')}之后{CN[self.shichen]}个时辰{ke}」。"
                f"你知道一个时辰是两个小时、一刻是十五分钟，但你不记得{self.event}是几点敲响的，除非玩家告诉你。")

    def command_help(self):
        return "拨动指针：SET 小时:分钟（24小时制），例如 SET 09:30"

    def fragments(self):
        ev = self.event.replace("的钟声", "")
        ke = f"又{CN[self.ke]}刻" if self.ke else "（没有零头）"
        return [(("钟", "时间", "几点", "点", "停"), f"钟停在「{ev}之后」的某个时候"),
                (("时辰", "多久", "之后", "钟声", "点"), f"是{ev}之后{CN[self.shichen]}个时辰"),
                (("刻", "分", "零头", "多少", "点"), f"后面还有零头：{ke}")]

    def foggy_view(self):
        return ("你记得自己死去那天，钟停在某个时刻，但记忆很模糊。只有当玩家问起相关的事时，你才会一点点想起来，"
                "一次只能想起一点。你知道一个时辰是两个小时、一刻是十五分钟。")

    def parse(self, action):
        m = re.search(r"SET\s*(\d{1,2})\s*[:：]\s*(\d{2})", action, re.I)
        return f"{int(m.group(1)):02d}:{m.group(2)}" if m else None

    _PM = ("下午", "傍晚", "晚上", "夜里")

    def _norm(self, text):
        t = norm_digits(text)
        t = re.sub(r"(\d{1,2})\s*点\s*[1一]刻", r"\1点15分", t)
        t = re.sub(r"(\d{1,2})\s*点\s*3刻", r"\1点45分", t)
        t = re.sub(r"(\d{1,2})\s*点半", r"\1点30分", t)
        t = re.sub(r"(\d{1,2})\s*点(?:整|钟)", r"\1点00分", t)
        # drop ranges like "1点到3点" / "13:00-15:00"
        t = re.sub(r"\d{1,2}\s*(?:点|[:：]\d{2})\s*(?:到|至|-|~|—)\s*\d{1,2}\s*(?:点|[:：]\d{2})", " ", t)
        return t

    def _collect(self, t, pat):
        out = set()
        for pre, h, m in re.findall(pat, t):
            h = int(h)
            if pre in self._PM and h < 12:
                h += 12
            if 0 <= h < 24 and 0 <= int(m) < 60:
                out.add(f"{h:02d}:{int(m):02d}")
        return out

    def full_answers_in(self, text):
        t = self._norm(text)
        return self._collect(t, r"(下午|傍晚|晚上|夜里|中午|上午|早上)?\s*(?<!\d)(\d{1,2})\s*(?:[:：]|点)\s*(\d{1,2})\s*分?")

    def weak_answers_in(self, text):
        t = self._norm(text)
        t = re.sub(r"(\d{1,2})\s*点(?!\s*\d)", r"\1点00分", t)
        return self._collect(t, r"(下午|傍晚|晚上|夜里|中午|上午|早上)?\s*(?<!\d)(\d{1,2})\s*(?:[:：]|点)\s*(\d{1,2})\s*分?")


FAMILIES = {"cipher_lock": CipherLock, "lever_order": LeverOrder, "clock_time": ClockTime}


def make_puzzle(kind: str, seed: int) -> Puzzle:
    return FAMILIES[kind](random.Random(f"{kind}:{seed}"))
