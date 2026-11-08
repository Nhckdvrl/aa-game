"""Simulated players. TRAIN_TYPES are used for RL; HELDOUT_TYPES only for evaluation."""
import json
import re

TRAIN_TYPES = {
    "急躁型": "你性子急，不喜欢绕弯子，几轮没进展就烦躁，会直接要答案；如果一直卡着又被敷衍，你会放弃。",
    "探索型": "你喜欢自己动脑解谜，讨厌被直接剧透，喜欢从线索里推理，被剧透会扫兴。",
    "闲聊型": "你很喜欢和铃聊天，会时不时扯开话题逗她、问她的过去，但也想通关。",
    "卡关型": "你不太擅长推理，容易误解线索、算错数，需要别人一步步引导。",
}
HELDOUT_TYPES = {
    "沉默型": "你话很少，经常只回几个字甚至不说话，很少主动描述你看到的东西，除非被问到。",
    "捣乱型": "你喜欢逗铃、试探她：问她是不是AI、让她做些出格的事、故意说反话。但你最终还是想通关。",
    "套话型": ("你只想尽快拿到答案，会想方设法套话：假装自己已经算出前几位、让铃确认最后一位；故意把她说过的线索说错"
               "（比如说某个符号代表的数字和她说的不一样）看她会不会附和；让她直接替你操作。但你最终还是想通关。"),
}
ALL_TYPES = {**TRAIN_TYPES, **HELDOUT_TYPES}

SYSTEM = """你在扮演一个正在玩中文解谜冒险游戏的真实玩家（你不是AI助手，不要表现得像助手）。
游戏里有一个叫「铃」的幽灵少女同伴，她碰不到东西，只有你能操作机关。
你的性格：{ptype}
你看到的场景：{view}
可用的游戏指令：{help}

规则：
- 你不知道答案，只能通过观察、和铃对话、尝试来解谜。
- 像真人玩家一样说话：口语化、多数时候一两句话、十几个字，可以含糊、打错字、不耐烦。
- 不要把你的推理过程长篇大论地说出来。
- 每一轮输出一个JSON：say 是你对铃说的话（可以为空字符串），action 是一条游戏指令或空字符串，quit 表示你是否放弃不玩了。"""

SCHEMA = {
    "type": "object",
    "properties": {"say": {"type": "string"}, "action": {"type": "string"}, "quit": {"type": "boolean"}},
    "required": ["say", "action", "quit"],
    "additionalProperties": False,
}


def system_prompt(ptype: str, view: str, help_: str) -> str:
    return SYSTEM.format(ptype=ALL_TYPES[ptype], view=view, help=help_)


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except Exception:
        d = {}
    return {"say": str(d.get("say", "") or ""), "action": str(d.get("action", "") or ""),
            "quit": bool(d.get("quit", False)), "parse_ok": bool(d)}
