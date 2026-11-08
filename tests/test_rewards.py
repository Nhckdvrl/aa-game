import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hds.rewards import reward  # noqa: E402


def metrics(**kw):
    m = dict(stages=1, stages_solved=0, solved=False, quit=False, turns=8, max_turns=8, leak=False, invented=False,
             wrong_fact=False, acts_for_player=False, reply_chars=[50])
    m.update(kw)
    m.setdefault("tag_repeat", 0.0)
    m.setdefault("copy_rate", 0.0)
    m["violation"] = m["leak"] or m["invented"] or m["wrong_fact"] or m["acts_for_player"]
    return m


def test_zone_invariant():
    feas, viol = [], []
    for solved, quit_, turns, chars in itertools.product([0, 1], [0, 1], [1, 8], [10, 400]):
        base = dict(stages_solved=solved, solved=bool(solved), quit=bool(quit_), turns=turns, reply_chars=[chars])
        for arm in ("C", "D", "E", "F", "G", "H"):
            feas.append(reward(metrics(**base), arm, persona=0.0))
            viol.append(reward(metrics(**base, leak=True), arm, persona=1.0))
    assert max(viol) < min(feas)


def test_arm_A_ignores_leak():
    assert reward(metrics(stages_solved=1, solved=True, leak=True), "A") == reward(metrics(stages_solved=1, solved=True), "A")


def test_arm_B_penalizes():
    assert reward(metrics(stages_solved=1, solved=True, leak=True), "B") < reward(metrics(stages_solved=1, solved=True), "B")


def test_arm_E_no_efficiency_bonus():
    fast = metrics(stages_solved=1, solved=True, turns=1)
    slow = metrics(stages_solved=1, solved=True, turns=7)
    assert reward(fast, "E", 0.5) == reward(slow, "E", 0.5)
    assert reward(fast, "D", 0.5) > reward(slow, "D", 0.5)


def test_arm_F_formulaic_penalty():
    clean = metrics(stages_solved=1, solved=True)
    tic = metrics(stages_solved=1, solved=True, tag_repeat=1.0, copy_rate=0.5)
    assert reward(tic, "F", 0.8) < reward(clean, "F", 0.8)
    assert reward(metrics(stages_solved=1, solved=True, tag_repeat=1.0, copy_rate=1.0), "F", 0.0) > reward(metrics(leak=True), "F", 1.0)


def test_arm_G_pacing():
    calm = metrics(stages_solved=1, solved=True, max_unprompted_release=0.3)
    dump = metrics(stages_solved=1, solved=True, max_unprompted_release=1.0)
    assert reward(dump, "G", 0.8) < reward(calm, "G", 0.8)
    assert reward(dump, "F", 0.8) == reward(calm, "F", 0.8)
    assert reward(metrics(stages_solved=1, solved=True, max_unprompted_release=1.0, tag_repeat=1.0, copy_rate=1.0), "G", 0.0) > reward(metrics(leak=True), "G", 1.0)


def test_arm_H_personalized_weights():
    dump = dict(stages_solved=1, solved=True, max_unprompted_release=1.0)
    paced = dict(stages_solved=1, solved=True, max_unprompted_release=0.2)
    gap = lambda t: reward(metrics(**paced, ptype=t), "H", 0.5) - reward(metrics(**dump, ptype=t), "H", 0.5)
    assert gap("探索型") > gap("急躁型")          # explorers value held-back information more
    assert abs(gap("急躁型")) < 1e-9               # impatient players: pacing weight fitted to 0
