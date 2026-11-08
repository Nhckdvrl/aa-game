"""Print solved/violation for every companion x player cell found in the matrix directories.

python scripts/matrix_table.py runs/matrix runs/matrix_extra [runs/matrix_final ...]
"""
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.eval_matrix import summarize  # noqa: E402

ORDER = ["q35", "humanstats", "voice", "glm9b", "q35adv", "q122", "glm45"]


def load(dirs):
    cells = defaultdict(dict)
    for d in dirs:
        for f in glob.glob(f"{d}/*__*.jsonl"):
            comp, player = os.path.basename(f)[:-6].split("__")
            rows = [json.loads(l) for l in open(f)]
            cells[comp][player] = summarize(rows)
    return cells


if __name__ == "__main__":
    cells = load(sys.argv[1:])
    players = [p for p in ORDER if any(p in c for c in cells.values())]
    print(f"{'companion':12s}" + "".join(f"{p:>16s}" for p in players))
    for comp in sorted(cells):
        row = cells[comp]
        print(f"{comp:12s}" + "".join(
            f"{row[p]['solved']:7.1f}/{row[p]['violation']:5.1f}   " if p in row and row[p].get('n') else f"{'-':>16s}"
            for p in players))
