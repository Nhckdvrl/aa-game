"""OpenAI-compatible clients for vLLM servers, safe across event loops (swift opens a new loop per rollout)."""
import asyncio
import random
import time

from openai import AsyncOpenAI

from .players import SCHEMA

_clients: dict = {}
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def client(url: str) -> AsyncOpenAI:
    loop = asyncio.get_running_loop()
    key = (id(loop), url)
    c = _clients.get(key)
    if c is None or c[0] is not loop:
        c = (loop, AsyncOpenAI(base_url=url, api_key="x", timeout=180, max_retries=3))
        _clients[key] = c
    return c[1]


def player_fn(urls: list[str], model: str, temperature: float = 1.0):
    """Round-robin over simulator replicas; output constrained to the player JSON schema."""
    state = {"i": random.randrange(len(urls))}  # random phase so concurrent episodes spread over replicas

    async def call(messages):
        url = urls[state["i"] % len(urls)]
        state["i"] += 1
        r = await client(url).chat.completions.create(
            model=model, messages=messages, max_tokens=256, temperature=temperature,
            response_format={"type": "json_schema", "json_schema": {"name": "player", "schema": SCHEMA}},
            extra_body=NO_THINK)
        return r.choices[0].message.content or ""
    return call


async def companion_reply(url: str, model: str, messages, max_tokens: int = 200, temperature: float = 0.7):
    t0 = time.perf_counter()
    r = await client(url).chat.completions.create(model=model, messages=messages, max_tokens=max_tokens,
                                                  temperature=temperature, extra_body=NO_THINK)
    return (r.choices[0].message.content or "").strip(), time.perf_counter() - t0
