"""How human-like is a player simulator's language?

Compares simulator player utterances (from eval jsonl files) with real Chinese user
follow-ups (WildChat val): length statistics, and the held-out AUC of a char 1-3gram
naive-Bayes discriminator (real vs simulated). AUC 0.5 = indistinguishable from humans.

python scripts/sim_realism.py --real hgsim/wildchat_zh_followups_val.jsonl runs/a.jsonl [runs/b.jsonl ...]
"""
import argparse
import json
import math
import random
from collections import Counter


# Topic-free abstraction: keep function words, particles and punctuation; replace content.
FUNC = set("的了吗吧呢啊嘛啦呀哦哈嗯额呃喂诶唉哎咋啥么嘞咯呗噢吼我你他她它们这那就都也还又不没别是在有要会能可以给把被让对和跟与或但而所因为如果然后再先才只很太最更好行算吧怎么什么哪谁为何几多少一个些点下")
def abstract(s):
    out = []
    for ch in s.strip():
        if ch in FUNC or not ch.isalnum():
            out.append(ch)
        elif ch.isdigit():
            out.append("0")
        elif ch.isascii():
            out.append("a")
        else:
            out.append("X")
    return "".join(out)


def grams(s):
    s = abstract(s)
    return [s[i:i + n] for n in (1, 2, 3) for i in range(len(s) - n + 1)]


def train_nb(pos, neg):
    cp, cn = Counter(), Counter()
    for s in pos:
        cp.update(grams(s))
    for s in neg:
        cn.update(grams(s))
    vocab = set(cp) | set(cn)
    tp, tn, V = sum(cp.values()), sum(cn.values()), len(vocab)
    def score(s):
        return sum(math.log((cp[g] + 1) / (tp + V)) - math.log((cn[g] + 1) / (tn + V)) for g in grams(s))
    return score


def auc(pos_scores, neg_scores):
    allv = sorted([(v, 1) for v in pos_scores] + [(v, 0) for v in neg_scores])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if allv[k][1] == 1)
        i = j
    P, N = len(pos_scores), len(neg_scores)
    return (rank_sum - P * (P + 1) / 2) / (P * N)


def stats(xs):
    L = sorted(len(x) for x in xs)
    return f"n={len(L)} p50={L[len(L)//2]} p90={L[9*len(L)//10]} <=15={100*sum(l<=15 for l in L)/len(L):.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    rng = random.Random(0)
    real = [json.loads(l)["user"] for l in open(a.real)]
    rng.shuffle(real)
    real = real[: a.n]
    print(f"{'real WildChat':28s} {stats(real)}")
    for f in a.runs:
        sim = [t["say"] for l in open(f) if '"transcript"' in l for t in json.loads(l)["transcript"]
               if t["role"] == "player" and t["say"].strip()]
        rng.shuffle(sim)
        sim = sim[: a.n]
        k = min(len(sim), len(real)) // 2
        score = train_nb(sim[:k], real[:k])
        A = auc([score(s) for s in sim[k:2 * k]], [score(s) for s in real[k:2 * k]])
        print(f"{f.split('/')[-1][:28]:28s} {stats(sim)} discriminator_AUC={A:.3f}")


if __name__ == "__main__":
    main()
