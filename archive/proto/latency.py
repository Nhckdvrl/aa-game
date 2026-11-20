import asyncio, time, statistics as st, sys
from openai import AsyncOpenAI
sys.path.insert(0, ".")
from run_episodes import CHARACTER
c = AsyncOpenAI(base_url=sys.argv[1], api_key="x"); model = sys.argv[2]
hist = [{"role": "system", "content": CHARACTER + "\n\n当前场景：你能读懂墙上的古文字。"}]
for i in range(6):
    hist += [{"role": "user", "content": f"铃，这是第{i}个房间吧？你怎么看？"}, {"role": "assistant", "content": "【别过头】哼，才不告诉你呢……好吧，你先看看墙上。"}]
async def one(q):
    msgs = hist + [{"role": "user", "content": q}]
    t0 = time.perf_counter(); ttft = fs = None; text = ""
    s = await c.chat.completions.create(model=model, messages=msgs, max_tokens=120, temperature=0.8, stream=True,
                                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    async for ch in s:
        d = ch.choices[0].delta.content if ch.choices else None
        if d:
            now = time.perf_counter() - t0
            ttft = ttft or now; text += d
            if fs is None and any(p in text.split("】")[-1] for p in "。！？…"): fs = now
    return ttft, fs or (time.perf_counter() - t0), time.perf_counter() - t0, len(text)
async def main():
    qs = ["门打不开啊，怎么办？", "你会害怕吗？", "这个符号是什么意思？", "我们走吧。"] * 5
    r = [await one(q) for q in qs]  # sequential = single user
    for name, i in [("ttft", 0), ("first_sentence", 1), ("total", 2)]:
        v = sorted(x[i] * 1000 for x in r); print(f"{name:15s} p50={st.median(v):6.0f}ms p90={v[int(.9*len(v))-1]:6.0f}ms")
    print("chars", st.mean(x[3] for x in r))
asyncio.run(main())
