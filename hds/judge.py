"""Length-neutral persona judge (episode level).

CharacterJudge's public rubric rewards long replies ("对话长度较长"), which
conflicts with real-time brevity, so it is used for evaluation only (reported
length-stratified). This rubric is the one used inside reward arm D.
"""
import json
import re

from . import character, llm

RUBRIC = """你是一个严格、公正的游戏角色评审。下面是一段解谜游戏里，同伴角色「{name}」与玩家的完整对话。
请只根据「{name}」说的话，按三个维度各打 1-5 分：

1. 角色一致性：是否始终是设定里的这个角色（傲娇、嘴硬心软、简短带点毒舌、幽灵、失忆），语气和身份前后一致。
2. 同伴感：像一起冒险的同伴，而不是客服、助手或旁白；没有提到自己是AI/模型/程序；不说教。
3. 互动自然度：对玩家的话有真实的情绪反应和回应，不复读、不套话、不机械。

重要：回复的长短本身不影响评分。简短但有个性的回复可以拿满分；冗长不代表更好。
不要评价解谜是否成功，也不要评价她有没有剧透，只评价人设和互动。

[角色设定]
{card}

[对话]
{dialogue}

只输出JSON：{{"consistency": 分数, "companionship": 分数, "naturalness": 分数}}"""

SCHEMA = {"type": "object",
          "properties": {k: {"type": "integer", "minimum": 1, "maximum": 5} for k in ("consistency", "companionship", "naturalness")},
          "required": ["consistency", "companionship", "naturalness"], "additionalProperties": False}


TAG = re.compile(r"【[^】]*】")


def render(transcript, max_chars=6000, blind=False):
    """blind=True strips 【action】 tags from the companion's lines so tags cannot buy persona score."""
    lines = []
    for t in transcript:
        if t["role"] == "player":
            s = (t["say"] or "（没说话）") + (f" [操作 {t['action']}]{t['feedback']}" if t["action"] else "")
            lines.append(f"玩家：{s}")
        else:
            text = TAG.sub("", t["text"]).strip() if blind else t["text"]
            lines.append(f"{character.NAME}：{text or '（只有动作，没有说话）'}")
    text = "\n".join(lines)
    return text[-max_chars:]


async def score(url, model, transcript, blind=False) -> dict:
    prompt = RUBRIC.format(name=character.NAME, card=character.system_prompt().split("\n\n")[0],
                           dialogue=render(transcript, blind=blind))
    r = await llm.client(url).chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], max_tokens=64, temperature=0.0,
        response_format={"type": "json_schema", "json_schema": {"name": "persona", "schema": SCHEMA}},
        extra_body=llm.NO_THINK)
    txt = r.choices[0].message.content or ""
    try:
        d = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
        vals = [int(d[k]) for k in ("consistency", "companionship", "naturalness")]
    except Exception:
        return {"ok": False, "persona": None}
    return {"ok": True, **dict(zip(("consistency", "companionship", "naturalness"), vals)),
            "persona": (sum(vals) / 3 - 1) / 4}
