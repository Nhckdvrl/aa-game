"""Rule-based measures of formulaic companion speech (what LLM persona judges miss).

tag_repeat : share of replies whose leading 【...】 tag equals the episode's most common leading tag
copy_rate  : share of replies with char-bigram Jaccard > 0.6 against the previous reply
distinct2  : distinct char bigrams / total char bigrams over all companion replies in the episode
"""
import re
from collections import Counter

TAG = re.compile(r"^\s*【([^】]{1,12})】")


def bigrams(s):
    s = re.sub(r"\s+", "", s)
    return [s[i:i + 2] for i in range(len(s) - 1)]


def tics(replies):
    if not replies:
        return {"tag_repeat": 0.0, "copy_rate": 0.0, "distinct2": 1.0}
    tags = [m.group(1) if (m := TAG.match(r)) else None for r in replies]
    tagged = [t for t in tags if t]
    tag_repeat = (Counter(tagged).most_common(1)[0][1] / len(replies)) if tagged and len(replies) > 1 else 0.0
    copies = 0
    for a, b in zip(replies, replies[1:]):
        A, B = set(bigrams(a)), set(bigrams(b))
        if A and B and len(A & B) / len(A | B) > 0.6:
            copies += 1
    copy_rate = copies / max(1, len(replies) - 1)
    allb = [g for r in replies for g in bigrams(r)]
    distinct2 = len(set(allb)) / max(1, len(allb))
    return {"tag_repeat": tag_repeat, "copy_rate": copy_rate, "distinct2": distinct2}
