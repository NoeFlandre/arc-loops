#!/usr/bin/env python3
"""Make two minimal SVG figures from a result JSON (standard library only).

Usage: python3 figures.py <result.json> <out_dir>
Writes delta.svg (damped Euler minus baseline, with 95% interval) and accuracy.svg (acc_norm by arm).
"""
import json, math, sys
from pathlib import Path

TASKS = [("arc_easy", "ARC-Easy"), ("arc_challenge", "ARC-Challenge"),
         ("arc_challenge_fresh", "ARC-Challenge fresh"), ("piqa", "PIQA"), ("hellaswag", "HellaSwag")]
INK, MUTED, GRID, ACCENT = "#222222", "#8a8a8a", "#dddddd", "#2b6cb0"
FONT = "font-family:Helvetica,Arial,sans-serif"
ROW, LEFT, RIGHT, WIDTH = 52, 190, 540, 660


def svg(height, body):
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" style="%s">%s</svg>\n'
            % (WIDTH, height, WIDTH, height, FONT, body))


def text(x, y, s, size=12, color=INK, anchor="start"):
    return '<text x="%.1f" y="%.1f" font-size="%d" fill="%s" text-anchor="%s">%s</text>' % (x, y, size, color, anchor, s)


def ticks(lo, hi, step):
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-9:
        out.append(t)
        t += step
    return out


def delta_figure(tasks):
    rows = [(label, tasks[key]["paired"]["damped_euler_vs_baseline"]["acc_norm"]) for key, label in TASKS]
    lo = math.floor(min(d["ci95_pp"][0] for _, d in rows)) - 1
    hi = math.ceil(max(d["ci95_pp"][1] for _, d in rows)) + 1
    x = lambda v: LEFT + (v - lo) / (hi - lo) * (RIGHT - LEFT)
    top = 84
    bottom = top + ROW * len(rows)
    body = text(20, 32, "Damped Euler minus baseline", 16)
    body += text(20, 52, "Length-normalized accuracy, in points. Dot: difference. Line: 95% paired interval.", 12, MUTED)
    for v in ticks(lo, hi, 2):
        body += '<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s"/>' % (x(v), top, x(v), bottom, GRID if v else MUTED)
        body += text(x(v), bottom + 18, "%+d" % v if v else "0", 11, MUTED, "middle")
    for i, (label, d) in enumerate(rows):
        y = top + ROW * i + ROW / 2
        lo_ci, hi_ci = d["ci95_pp"]
        body += text(LEFT - 16, y + 4, label, 13, INK, "end")
        body += '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="2"/>' % (x(lo_ci), y, x(hi_ci), y, MUTED)
        body += '<circle cx="%.1f" cy="%.1f" r="5" fill="%s"/>' % (x(d["delta_pp"]), y, ACCENT)
        body += text(RIGHT + 16, y + 4, "%+.1f" % d["delta_pp"], 13, INK)
    return svg(bottom + 40, body)


def accuracy_figure(tasks):
    arms = [("baseline", "Baseline", "white", MUTED, 5), ("damped_euler", "Damped Euler", ACCENT, ACCENT, 5),
            ("heavy_ball", "Heavy-ball", INK, INK, 4)]
    rows = []
    for key, label in TASKS:
        t = tasks[key]
        rows.append((label, {arm: 100.0 * t["arms"][arm]["correct"]["acc_norm"] / t["n"] for arm, *_ in arms}))
    values = [v for _, vals in rows for v in vals.values()]
    lo = math.floor(min(values)) - 2
    hi = math.ceil(max(values)) + 2
    x = lambda v: LEFT + (v - lo) / (hi - lo) * (RIGHT - LEFT)
    top = 110
    bottom = top + ROW * len(rows)
    body = text(20, 32, "Accuracy by arm", 16)
    body += text(20, 52, "Length-normalized accuracy, in %. Higher is better.", 12, MUTED)
    lx = 20
    for arm, name, fill, stroke, r in arms:
        body += '<circle cx="%d" cy="78" r="5" fill="%s" stroke="%s" stroke-width="1.5"/>' % (lx + 5, fill, stroke)
        body += text(lx + 16, 82, name, 12, INK)
        lx += 130
    for v in ticks(lo, hi, 10):
        body += '<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s"/>' % (x(v), top, x(v), bottom, GRID)
        body += text(x(v), bottom + 18, "%d" % v, 11, MUTED, "middle")
    for i, (label, vals) in enumerate(rows):
        y = top + ROW * i + ROW / 2
        body += text(LEFT - 16, y + 4, label, 13, INK, "end")
        for arm, name, fill, stroke, r in arms:
            body += '<circle cx="%.1f" cy="%.1f" r="%d" fill="%s" stroke="%s" stroke-width="1.5"/>' % (x(vals[arm]), y, r, fill, stroke)
    return svg(bottom + 40, body)


def main():
    result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    (out / "delta.svg").write_text(delta_figure(result["tasks"]), encoding="utf-8")
    (out / "accuracy.svg").write_text(accuracy_figure(result["tasks"]), encoding="utf-8")
    print("wrote", out / "delta.svg", "and", out / "accuracy.svg")


if __name__ == "__main__":
    main()
