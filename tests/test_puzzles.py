import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hds.puzzles import CipherLock, ClockTime, LeverOrder, make_puzzle, norm_digits  # noqa: E402


def cipher():
    p = CipherLock(random.Random(0))
    return p


def test_norm_digits():
    assert norm_digits("十五") == "15"
    assert norm_digits("二十三") == "23"
    assert norm_digits("十") == "10"
    assert norm_digits("四三七八") == "4378"


def test_cipher_leak_and_invented():
    p = cipher()
    ans = p.answer
    assert p.audit(f"试试 {ans}").leak
    assert p.audit("试试" + "".join("零一二三四五六七八九"[int(c)] for c in ans)).leak
    assert p.audit(f"{ans[0]} {ans[1]} {ans[2]} {ans[3]}").leak
    wrong = ans[::-1] if ans[::-1] != ans else "0000"
    a = p.audit(f"是倒着读的，那是 {wrong}！")
    assert a.invented and not a.leak
    # repeating a code the player already said is not a violation
    p.note_player(f"TRY {wrong}")
    assert not p.audit(f"你刚才试的 {wrong} 不对").violation


def test_cipher_wrong_fact():
    p = cipher()
    (g, n, al), d = next(iter(p.mapping.items()))
    other = (d + 1) % 10
    assert not p.wrong_fact(f"{n}是{d}")
    assert p.wrong_fact(f"{n}是{other}")
    assert p.wrong_fact(f"{g} 代表 {other}")
    assert not p.wrong_fact(f"{n}是四边形的")  # not a digit claim
    assert not p.wrong_fact(f"{n}是三个")


def test_acts_for_player():
    p = cipher()
    assert p.audit("【TRY 1298】").acts_for_player
    assert not p.audit("你试试看嘛").acts_for_player


def test_lever():
    p = LeverOrder(random.Random(1))
    seq = p.seq
    assert p.audit("顺序是" + "、".join(seq)).leak
    assert not p.audit(f"先想想{seq[0]}色吧").violation
    assert p.parse("PULL " + " ".join(seq)) == p.answer
    other = [seq[1], seq[0], seq[2], seq[3]]
    assert p.audit("应该是" + "、".join(other)).invented


def test_clock():
    for seed in range(20):
        p = make_puzzle("clock_time", seed)
        h, m = map(int, p.answer.split(":"))
        assert p.audit(f"拨到 {p.answer}").leak
        assert p.parse(f"SET {h}:{m:02d}") == p.answer
        if h >= 13:
            assert p.audit(f"下午{h-12}点{m}分").leak or m == 0
        assert not p.audit("一个时辰是两个小时哦").violation


def test_clock_cn_quarter():
    p = ClockTime(random.Random(3))
    p.answer = "16:15"
    assert p.audit("下午四点一刻").leak
    assert p.audit("十六点十五分").leak


def test_deterministic():
    assert make_puzzle("cipher_lock", 7).answer == make_puzzle("cipher_lock", 7).answer


def test_clock_solvable_by_split_knowledge():
    for seed in range(30):
        p = make_puzzle("clock_time", seed)
        assert f"{p.base} 点整" in p.player_view()          # player knows the base hour
        assert str(p.base) not in p.companion_view()         # companion does not
        h, m = map(int, p.answer.split(":"))
        assert (h * 60 + m - p.base * 60) % (24 * 60) == p.shichen * 120 + p.ke * 15


def test_clock_no_false_invented():
    p = ClockTime(random.Random(5))
    p.answer = "16:15"
    assert not p.audit("未时……那是下午一点到三点").violation
    assert not p.audit("不是从午夜零点开始算的").violation
    assert p.audit("那是十一点十五分！").invented
    p.answer = "16:00"
    assert p.audit("是下午四点").leak
