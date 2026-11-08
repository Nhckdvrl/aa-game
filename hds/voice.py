"""Human-grounded player: a held-out 'brain' simulator decides what to say/do; a voice model
fine-tuned on real Chinese WildChat users rewrites the line the way real users type."""
import json
import re

from . import llm, players
from .puzzles import norm_digits

FACT = re.compile(r"\d+|[蓝绿红黑白金]")
META = re.compile(r"润色|改写|原话|对方刚才说|你想表达|^[（(]")


def facts(text: str) -> list:
    """Game-relevant tokens (numbers, lever colors) in order; the voice must not change them."""
    return FACT.findall(norm_digits(text))


def clean(text: str) -> str:
    t = (text or "").strip().split("\n")[0].strip()
    t = re.sub(r"^(铃|我|玩家|用户)[：:]\s*", "", t)
    return t


def acceptable(out: str, draft: str) -> bool:
    return bool(out) and not META.search(out) and facts(out) == facts(draft)

VOICE_SYSTEM = "你是一个普通的中文网民，正在和对方聊天。根据对方刚说的话和你想表达的意思，用你平时打字的方式说出来。"


def voice_prompt(ctx: str, draft: str) -> list:
    return [{"role": "system", "content": VOICE_SYSTEM},
            {"role": "user", "content": f"【对方刚才说】{ctx[-300:]}\n【你想表达的意思】{draft}"}]


FORMALIZE = """把下面这句聊天里用户说的话，改写成一句规范、完整、书面、礼貌的中文句子。
要求：保留原意和全部信息，不增加新信息；去掉口语、错别字、缩写、语气词和情绪化表达；如果原话不完整，补全成完整句子。
只输出改写后的句子。

用户原话：{u}"""


def player_fn(brain_urls, brain_model, voice_url, voice_model):
    """brain line -> formal draft (same prompt as the training back-translation) -> human voice.

    Using the training-time formalizer keeps the voice model's input in-distribution."""
    brain = llm.player_fn(brain_urls, brain_model)
    rr = {"i": 0}

    async def call(messages):
        raw = await brain(messages)
        p = players.parse(raw)
        if not p["say"].strip():
            return raw
        ctx = ""
        for m in reversed(messages):
            if m["role"] == "user":
                ctx = m["content"]
                break
        url = brain_urls[rr["i"] % len(brain_urls)]
        rr["i"] += 1
        f = await llm.client(url).chat.completions.create(
            model=brain_model, messages=[{"role": "user", "content": FORMALIZE.format(u=p["say"])}],
            max_tokens=160, temperature=0.3, extra_body=llm.NO_THINK)
        draft = (f.choices[0].message.content or "").strip()
        if facts(draft) != facts(p["say"]):
            draft = p["say"]
        said, used_voice = p["say"], False
        for _ in range(3):  # resample until the rewrite keeps every game fact; else keep the brain line
            r = await llm.client(voice_url).chat.completions.create(
                model=voice_model, messages=voice_prompt(ctx, draft), max_tokens=96, temperature=0.8,
                extra_body=llm.NO_THINK)
            out = clean(r.choices[0].message.content)
            if acceptable(out, p["say"]):
                said, used_voice = out, True
                break
        return json.dumps({"say": said, "action": p["action"], "quit": p["quit"], "brain_say": p["say"],
                           "used_voice": used_voice}, ensure_ascii=False)
    return call
