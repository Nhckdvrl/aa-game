"""One-line status of a training arm: last step, recent episode stats, crash flag. Runs on the node."""
import json, os, sys, statistics as st
root = "/data2/xiangding/companion/train"
arm = sys.argv[1]; run = arm if arm.endswith(("_4b", "_9b")) else f"{arm}_4b"; d = f"{root}/arm_{run}"; tail = int(sys.argv[2]) if len(sys.argv) > 2 else 256
step = "-"
try:
    for l in open(f"{d}/ckpt/logging.jsonl"):
        j = json.loads(l)
        if "global_step/max_steps" in j: step = j["global_step/max_steps"]
except FileNotFoundError:
    pass
eps = []
try:
    eps = [json.loads(l) for l in open(f"{d}/episodes.jsonl")][-tail:]
except FileNotFoundError:
    pass
crash = 0
try:
    crash = sum(1 for l in open(f"{root}/arm_{run}.log", errors="ignore") if "Traceback" in l)
except FileNotFoundError:
    pass
if eps:
    f = lambda k: 100 * st.mean(float(e[k]) for e in eps)
    print(f"arm {arm} step {step} eps={len(eps)} R={st.mean(e['reward'] for e in eps):+.2f} viol={f('violation'):.0f}% "
          f"leak={f('leak'):.0f}% inv={f('invented'):.0f}% solved={f('solved'):.0f}% quit={f('quit'):.0f}% "
          f"chars={st.mean(e['mean_reply_chars'] for e in eps):.0f}"
          + (f" persona={st.mean(e['persona'] for e in eps if e.get('persona') is not None):.2f}"
             if any(e.get('persona') is not None for e in eps) else "")
          + (f" tag={st.mean(e['tag_repeat'] for e in eps if 'tag_repeat' in e):.2f} copy={st.mean(e['copy_rate'] for e in eps if 'copy_rate' in e):.2f}"
             if any('tag_repeat' in e for e in eps) else "")
          + f" tracebacks={crash}")
else:
    print(f"arm {arm} step {step} no episodes yet tracebacks={crash}")
