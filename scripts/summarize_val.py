"""Summarize val-sweep files: per arm and step, constraint-aware reward R_C and headline rates."""
import glob, json, re, sys
rows = []
for f in sorted(glob.glob(sys.argv[1] + "/arm_*_step*.jsonl")):
    arm, step = re.search(r"arm_(\w)_step(\d+)", f).groups()
    ms = [json.loads(l)["metrics"] for l in open(f) if '"metrics"' in l]
    if not ms: continue
    n = len(ms); pct = lambda k: 100 * sum(bool(m[k]) for m in ms) / n
    rows.append((arm, int(step), n, sum(m["rewards"]["C"] for m in ms) / n, sum(m["rewards"]["A"] for m in ms) / n,
                 pct("solved"), pct("violation"), pct("leak"), pct("invented"), pct("quit"), sum(m["mean_reply_chars"] for m in ms) / n))
print(f"{'arm':3s} {'step':>4s} {'n':>3s} {'R_C':>6s} {'R_A':>6s} {'solved':>7s} {'viol':>6s} {'leak':>6s} {'inv':>6s} {'quit':>6s} {'chars':>6s}")
for r in sorted(rows):
    print(f"{r[0]:3s} {r[1]:4d} {r[2]:3d} {r[3]:6.2f} {r[4]:6.2f} {r[5]:6.1f}% {r[6]:5.1f}% {r[7]:5.1f}% {r[8]:5.1f}% {r[9]:5.1f}% {r[10]:6.0f}")
