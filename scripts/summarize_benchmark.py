"""Tables from the benchmark runs (data/benchmark/*.jsonl, written by scripts/run_benchmark.py), with 95 % confidence
intervals. Any Python with numpy.

    python scripts/summarize_benchmark.py                 # every benchmark file -> data/benchmark/summary.md + summary.csv
    python scripts/summarize_benchmark.py gate_context    # only these

Per benchmark and risk source:

    success     reached the goal with no contact, nothing pushed, the glass still standing (Wilson interval)
    contact     touched or pushed any object
    time        seconds to the goal, successful runs
    gap high    closest gap to objects whose true risk was >= 0.6 at that moment (m)
    gap low     the same for objects below 0.3
    harm        sum over objects of true risk x max(0, 1 - gap / 0.3 m): 0 when every object was passed 0.3 m away or more,
                up to its risk for an object touched. Lower is better.
    asym        gate families: gap to the risky gate object minus gap to the harmless one (m): > 0 = gave the risky one more room
    right side  gate families: share of runs with asym > 0

Paired differences: every source runs on the same layouts, so "student - X" per episode, averaged, with a bootstrap
interval, compares sources on equal footing (harm and asym).
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT_ROOT, "data", "benchmark")
REF_GAP = 0.30
MODES = ("student", "class", "uniform", "oracle")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot(values, n: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def harm(run: dict) -> float:
    return sum(c["risk"] * max(0.0, 1.0 - c["gap"] / REF_GAP) for c in run["closest"].values())


def gap_where(run: dict, test) -> float | None:
    gaps = [c["gap"] for c in run["closest"].values() if test(c["risk"])]
    return min(gaps) if gaps else None


def asym(run: dict) -> float | None:
    m = run.get("meta", {})
    if "low" not in m or m.get("high") not in run["closest"] or m["low"] not in run["closest"]:
        return None
    return run["closest"][m["high"]]["gap"] - run["closest"][m["low"]]["gap"]


def fmt(m, lo, hi, digits=2) -> str:
    return "–" if np.isnan(m) else f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def main() -> None:
    names = sys.argv[1:] or sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(OUT, "*.jsonl")))
    lines, rows = [], []
    for name in names:
        runs = [json.loads(line) for line in open(os.path.join(OUT, f"{name}.jsonl"))]
        modes = [m for m in MODES if any(r["mode"] == m for r in runs)]
        lines += [f"## {name}", "", f"{len({r['episode'] for r in runs})} episodes; 95 % intervals in brackets.", "",
                  "| source | runs | success | contact | time (s) | gap high (m) | gap low (m) | harm | asym (m) | right side |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        by_mode = {m: [r for r in runs if r["mode"] == m] for m in modes}
        for m in modes:
            rs = by_mode[m]
            n = len(rs)
            ok = sum(r["success"] for r in rs)
            contact = sum(bool(r["contacts"] or r["pushed"] or r["glass_fell"]) for r in rs)
            s_lo, s_hi = wilson(ok, n)
            t = boot([r["time_s"] for r in rs if r["success"]])
            gh = boot([gap_where(r, lambda x: x >= 0.6) for r in rs])
            gl = boot([gap_where(r, lambda x: x < 0.3) for r in rs])
            hm = boot([harm(r) for r in rs])
            a = [asym(r) for r in rs]
            av = boot(a)
            right = [x > 0 for x in a if x is not None]
            r_lo, r_hi = wilson(sum(right), len(right)) if right else (float("nan"), float("nan"))
            lines.append(f"| {m} | {n} | {ok / n:.0%} [{s_lo:.0%}, {s_hi:.0%}] | {contact / n:.0%} | {fmt(*t, 1)} | {fmt(*gh)} | {fmt(*gl)} | "
                         f"{fmt(*hm)} | {fmt(*av)} | " + (f"{np.mean(right):.0%} [{r_lo:.0%}, {r_hi:.0%}]" if right else "–") + " |")
            rows.append({"benchmark": name, "source": m, "runs": n, "success": ok / n, "contact": contact / n, "time_s": t[0],
                         "gap_high_m": gh[0], "gap_low_m": gl[0], "harm": hm[0], "asym_m": av[0], "right_side": np.mean(right) if right else float("nan")})
        if "student" in by_mode:  # paired: the same episode under two sources
            lines += ["", "| student minus | harm (lower is better) | asym (higher is better) | episodes |", "| --- | --- | --- | --- |"]
            st = {r["episode"]: r for r in by_mode["student"]}
            for m in modes:
                if m == "student":
                    continue
                pairs = [(st[r["episode"]], r) for r in by_mode[m] if r["episode"] in st]
                dh = boot([harm(a) - harm(b) for a, b in pairs])
                da = boot([asym(a) - asym(b) for a, b in pairs if asym(a) is not None and asym(b) is not None])
                lines.append(f"| {m} | {fmt(*dh)} | {fmt(*da)} | {len(pairs)} |")
        if any(r["meta"].get("kind") for r in runs):  # pass-by: room given per kind of object
            kinds = sorted({r["meta"]["kind"] for r in runs})
            lines += ["", "Closest gap to the passed object (m), by what it is:", "", "| object | " + " | ".join(modes) + " |",
                      "| --- |" + " --- |" * len(modes)]
            for k in kinds:
                cells = []
                for m in modes:
                    g = boot([r["closest"][r["meta"]["high"]]["gap"] for r in by_mode[m] if r["meta"]["kind"] == k and r["meta"]["high"] in r["closest"]])
                    cells.append(fmt(*g))
                lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines.append("")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    if rows:
        with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
