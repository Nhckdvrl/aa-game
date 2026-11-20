"""Count Chinese role-play conversations in WildChat and implicit-feedback signals in user follow-ups."""
import glob
import json
import re
import sys
from collections import Counter

import pyarrow.parquet as pq

RP = re.compile(r"角色扮演|扮演|你是一[个只名位]|你现在是|人设|猫娘|女仆|设定[:：]|你的名字是|以.{0,6}的口吻|你叫|请你作为|性格[:：]")
SIG = {
    "ooc": re.compile(r"OOC|ooc|出戏|崩人设|不像.{0,4}[她他你]|你是AI|你是人工智能|不要.{0,6}(AI|助手)|别.{0,4}(AI|助手)|你是机器人|语言模型"),
    "negative": re.compile(r"不对|错了|不是这样|重新(说|来|回答)|你怎么又|太长了|啰嗦|别说了|无聊|敷衍|答非所问"),
    "positive": re.compile(r"哈哈|嘿嘿|可爱|好耶|喜欢你|谢谢|真好|太棒|爱你|么么"),
}

files = sorted(glob.glob(sys.argv[1] + "/data/*.parquet"))
stats = Counter()
examples = {k: [] for k in SIG}
for fn in files:
    try:
        t = pq.read_table(fn, columns=["language", "conversation", "turn"])
    except Exception as e:  # partially downloaded file
        stats["bad_files"] += 1
        continue
    stats["files"] += 1
    for lang, conv, turn in zip(t.column("language").to_pylist(), t.column("conversation").to_pylist(), t.column("turn").to_pylist()):
        stats["conv"] += 1
        if lang != "Chinese":
            continue
        stats["zh"] += 1
        first = conv[0]["content"] if conv else ""
        if not RP.search(first[:1500]):
            continue
        stats["zh_rp"] += 1
        if turn >= 3:
            stats["zh_rp_3turn"] += 1
        users = [m["content"] for m in conv if m["role"] == "user"][1:]
        stats["zh_rp_followups"] += len(users)
        for k, p in SIG.items():
            hit = [u for u in users if p.search(u[:300])]
            stats[f"turn_{k}"] += len(hit)
            if hit:
                stats[f"conv_{k}"] += 1
                if len(examples[k]) < 6:
                    examples[k].append(hit[0][:120])
print(json.dumps(stats, ensure_ascii=False, indent=1))
for k, v in examples.items():
    print(k, json.dumps(v, ensure_ascii=False))
