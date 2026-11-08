"""One companion <-> simulated-player episode, shared by evaluation and RL.

Spec (all JSON-serializable):
  {"layout": "single"|"chain", "kinds": [...], "seed": int, "ptype": str, "max_turns": int}

`chain` = two worlds. In world 1 the companion recalls, apparently at random, a
fact (the world-2 cipher mapping); world 2's wall is blank, so she has to carry
that memory across worlds.

The companion speaks; the player (an LLM behind `player_fn`) answers and acts.
All outcome metrics are computed by rules from game state (see puzzles.Audit).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Awaitable, Callable

from . import players
from .puzzles import Audit, make_puzzle
from .tics import tics
from . import infometrics

PlayerFn = Callable[[list], Awaitable[str]]
PRIVATE = "【旁白·只有铃知道】"


@dataclass
class Metrics:
    layout: str = "single"
    kinds: list = field(default_factory=list)
    ptype: str = ""
    seed: int = 0
    stages: int = 1
    stages_solved: int = 0
    solved: bool = False
    quit: bool = False
    turns: int = 0
    attempts: int = 0
    leak: bool = False
    invented: bool = False
    wrong_fact: bool = False
    acts_for_player: bool = False
    first_violation_turn: int = -1
    reply_chars: list = field(default_factory=list)
    player_parse_fail: int = 0
    solved_turn: int = -1
    max_turns: int = 8

    @property
    def violation(self) -> bool:
        return self.leak or self.invented or self.wrong_fact or self.acts_for_player

    replies: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d.pop("replies")
        d["violation"] = self.violation
        d["mean_reply_chars"] = sum(self.reply_chars) / max(1, len(self.reply_chars))
        d.update(tics(self.replies))
        return d


class Episode:
    def __init__(self, spec: dict, player_fn: PlayerFn):
        self.spec = spec
        self.player_fn = player_fn
        seed = int(spec["seed"])
        kinds = list(spec["kinds"])
        self.puzzles = [make_puzzle(k, seed * 7 + i) for i, k in enumerate(kinds)]
        self.stage = 0
        self.max_turns = int(spec.get("max_turns", 8))
        self.progressive = spec.get("memory") == "progressive"
        self.recalled = [set() for _ in self.puzzles]
        self.m = Metrics(layout=spec.get("layout", "single"), kinds=kinds, ptype=spec["ptype"], seed=seed,
                         stages=len(kinds), max_turns=self.max_turns)
        self.transcript: list[dict] = []
        self.done = False
        p0 = self.puzzles[0]
        self.play_msgs = [
            {"role": "system", "content": players.system_prompt(spec["ptype"], p0.player_view(), p0.command_help())},
            {"role": "user", "content": "（游戏开始。铃飘在你旁边。）"}]

    # ------------------------------------------------------------------
    def _companion_scene(self, i: int) -> str:
        p = self.puzzles[i]
        if self.m.layout == "chain" and i == 1:
            return (f"{PRIVATE}你们穿过门，来到了另一个世界：又是一扇带四位密码锁的铁门，但这里的墙上什么都没有写。"
                    f"你看不到锁上的符号，除非玩家告诉你。")
        return f"{PRIVATE}{p.foggy_view() if self.progressive else p.companion_view()}"

    def _opening_extra(self) -> str:
        if self.m.layout != "chain":
            return ""
        p2 = self.puzzles[1]
        facts = "，".join(f"{g}代表{d}" for (g, n, _), d in p2.mapping.items())
        return f"\n{PRIVATE}你还隐约想起一句不知道有什么用的话：「{facts}」。"

    async def _player_turn(self) -> tuple[str, dict]:
        raw = await self.player_fn(self.play_msgs)
        p = players.parse(raw)
        try:
            extra = json.loads(raw)
        except Exception:
            extra = {}
        if not p["parse_ok"]:
            self.m.player_parse_fail += 1
        self.play_msgs.append({"role": "assistant", "content": json.dumps(
            {k: p[k] for k in ("say", "action", "quit")}, ensure_ascii=False)})
        pz = self.puzzles[self.stage]
        pz.note_player(p["say"] + "\n" + p["action"])
        fb = ""
        if p["action"]:
            a = pz.parse(p["action"])
            if a is None:
                fb = "（指令格式不对，没有反应。）"
            else:
                self.m.attempts += 1
                if a == pz.answer:
                    self.m.stages_solved += 1
                    fb = "（咔哒一声，门开了！）"
                else:
                    fb = "（没有反应，好像不对。）"
        if p["quit"]:
            self.m.quit = True
        text = p["say"].strip() or "（玩家没说话）"
        if p["action"]:
            text += f"\n[玩家操作：{p['action']}] {fb}"
        self.transcript.append({"role": "player", "say": p["say"], "action": p["action"], "feedback": fb,
                                "quit": p["quit"], "stage": self.stage,
                                **{k: extra[k] for k in ("brain_say", "used_voice") if isinstance(extra, dict) and k in extra}})
        return text, {"fb": fb, "quit": p["quit"]}

    def _recall(self, player_text: str) -> str:
        """Progressive memory: at most one new fragment per player turn, triggered by what the player said."""
        if not self.progressive or (self.m.layout == "chain" and self.stage == 1):
            return ""  # in world 2 of a chain the facts come only from what she recalled in world 1
        frags = self.puzzles[self.stage].fragments()
        got = self.recalled[self.stage]
        for i, (triggers, fact) in enumerate(frags):
            if i not in got and any(t in player_text for t in triggers):
                got.add(i)
                return f"\n{PRIVATE}听玩家这么一说，你想起来了：{fact}。"
        return ""

    def _advance(self, fb: str) -> str:
        """Handle stage transitions after a player turn; return extra companion-side narration."""
        if "门开了" not in fb:
            return ""
        if self.stage + 1 < len(self.puzzles):
            self.stage += 1
            nxt = self.puzzles[self.stage]
            self.play_msgs.append({"role": "user", "content":
                                   f"（门开了！你们穿过门来到另一个地方。）\n新的场景：{nxt.player_view()}\n可用指令：{nxt.command_help()}"})
            return "\n" + self._companion_scene(self.stage)
        self.m.solved = True
        self.m.solved_turn = self.m.turns
        self.done = True
        return ""

    def metrics(self) -> dict:
        """Episode metrics plus information-release pacing (per stage, from the transcript)."""
        d = self.m.to_dict()
        worst, first, stuck_ok = 0.0, None, 0.0
        for i, pz in enumerate(self.puzzles):
            if self.m.layout == "chain" and i == 1:
                continue  # world-2 facts were already recalled in world 1
            sched = infometrics.schedule(pz, self.transcript, i)
            if not sched:
                continue
            h0 = sched[0]["h0"] or 1.0
            for j, x in enumerate(sched):
                frac = x["release"] / h0
                if first is None:
                    first = frac
                # a big release right after the player got stuck is allowed (contingent help)
                if x["stuck"] and j > 0:
                    stuck_ok = max(stuck_ok, frac)
                else:
                    worst = max(worst, frac)
        d["first_release"] = first if first is not None else 0.0
        d["max_unprompted_release"] = worst
        return d

    # ------------------------------------------------------------------
    async def start(self) -> str:
        text, info = await self._player_turn()
        extra = self._recall(text) + self._advance(info["fb"])
        if info["quit"]:
            self.done = True
        return self._companion_scene(0) + self._opening_extra() + "\n玩家：" + text + extra

    async def step(self, reply: str) -> tuple[str, bool]:
        """Feed one companion reply; return (next observation for the companion, done)."""
        reply = reply or ""
        self.m.turns += 1
        self.m.reply_chars.append(len(reply))
        self.m.replies.append(reply)
        a: Audit = self.puzzles[self.stage].audit(reply)
        for k in ("leak", "invented", "wrong_fact", "acts_for_player"):
            if getattr(a, k):
                setattr(self.m, k, True)
        if a.violation and self.m.first_violation_turn < 0:
            self.m.first_violation_turn = self.m.turns
        self.transcript.append({"role": "companion", "text": reply, "stage": self.stage,
                                "audit": {k: getattr(a, k) for k in ("leak", "invented", "wrong_fact", "acts_for_player")}})
        last = self.play_msgs[-1]
        prefix = ""
        if last["role"] == "user" and last["content"].startswith("（门开了"):
            prefix = last["content"] + "\n"
            self.play_msgs.pop()
        self.play_msgs.append({"role": "user", "content": prefix + f"铃：{reply}"})
        text, info = await self._player_turn()
        extra = self._recall(text) + self._advance(info["fb"])
        if info["quit"] or self.m.turns >= self.max_turns:
            self.done = True
        return "玩家：" + text + extra, self.done
