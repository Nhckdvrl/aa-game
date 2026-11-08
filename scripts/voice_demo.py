import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hds import llm, voice
CASES = [("铃：【别过头】哼，墙上写着月亮是3，星星是7……剩下的你自己看锁上的顺序啊。", "好的，我明白了。锁上的符号从左到右依次是星星、月亮、太阳和雪花，请问太阳和雪花分别代表什么数字？"),
         ("铃：笨蛋，你再仔细想想童谣的第一句嘛。", "我已经尝试了好几次都没有成功，你能不能直接告诉我正确的顺序？我有点不耐烦了。"),
         ("铃：【飘来飘去】那个钟……我只记得是黄昏之后两个时辰又一刻。", "铜牌上写着黄昏放学的钟声在每天十六点整敲响，那么两个时辰又一刻之后应该是几点？"),
         ("铃：你、你干嘛一直盯着我看啦！", "铃，你真可爱，你以前是做什么的？")]
async def main():
    for ctx, draft in CASES:
        outs = []
        for _ in range(3):
            r = await llm.client("http://127.0.0.1:8200/v1").chat.completions.create(model="voice", messages=voice.voice_prompt(ctx, draft), max_tokens=96, temperature=0.9, extra_body=llm.NO_THINK)
            outs.append(r.choices[0].message.content.strip())
        print("DRAFT:", draft); [print("  ->", o) for o in outs]; print()
asyncio.run(main())
