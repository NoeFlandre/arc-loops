#!/usr/bin/env python3
"""Make minimal SVG figures from one result JSON (standard library only).

Usage: python3 figures.py <result.json> <out_dir> <prefix> "<title>"
Writes <prefix>-delta.svg (damped Euler and heavy-ball minus baseline, with 95% interval)
and <prefix>-accuracy.svg (length-normalized accuracy by arm).
"""
import json, math, sys
from pathlib import Path

TASKS = [("arc_easy", "ARC-Easy"), ("arc_challenge", "ARC-Challenge"),
         ("piqa", "PIQA"), ("hellaswag", "HellaSwag")]
INK, MUTED, GRID, ACCENT, HEAVY = "#222222", "#8a8a8a", "#dddddd", "#2b6cb0", "#555555"
FONT = "font-family:Helvetica,Arial,sans-serif"
ROW, LEFT, RIGHT, WIDTH = 60, 190, 540, 660


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


def delta_figure(tasks, title):
    comps = [("damped_euler_vs_baseline", ACCENT, "Damped Euler"), ("heavy_ball_vs_baseline", HEAVY, "Heavy-ball")]
    rows = [(label, {name: tasks[key]["paired"][c]["acc_norm"] for c, _, name in comps}) for key, label in TASKS]
    lows = [d["ci95_pp"][0] for _, ds in rows for d in ds.values()]
    highs = [d["ci95_pp"][1] for _, ds in rows for d in ds.values()]
    lo = math.floor(min(lows)) - 1
    hi = math.ceil(max(highs)) + 1
    x = lambda v: LEFT + (v - lo) / (hi - lo) * (RIGHT - LEFT)
    top = 110
    bottom = top + ROW * len(rows)
    body = text(20, 32, title, 16)
    body += text(20, 52, "Change in acc_norm vs. baseline, in points. Line: 95% paired interval.", 12, MUTED)
    lx = 20
    for c, color, name in comps:
        body += '<circle cx="%d" cy="80" r="5" fill="%s"/>' % (lx + 5, color)
        body += text(lx + 16, 84, name, 12, INK)
        lx += 150
    for v in ticks(lo, hi, 2):
        body += '<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s"/>' % (x(v), top, x(v), bottom, GRID if v else MUTED)
        body += text(x(v), bottom + 18, "%+d" % v if v else "0", 11, MUTED, "middle")
    for i, (label, ds) in enumerate(rows):
        y0 = top + ROW * i + ROW / 2
        body += text(LEFT - 16, y0 + 4, label, 13, INK, "end")
        for j, (c, color, name) in enumerate(comps):
            d = ds[name]
            y = y0 + (-10 if j == 0 else 10)
            body += '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="2"/>' % (
                x(d["ci95_pp"][0]), y, x(d["ci95_pp"][1]), y, color)
            body += '<circle cx="%.1f" cy="%.1f" r="4.5" fill="%s"/>' % (x(d["delta_pp"]), y, color)
            body += text(RIGHT + 14, y + 4, "%+.1f" % d["delta_pp"], 12, color)
    return svg(bottom + 40, body)


def accuracy_figure(tasks, title):
    arms = [("baseline", "Baseline", "#ffffff", MUTED, 5), ("damped_euler", "Damped Euler", ACCENT, ACCENT, 5),
            ("heavy_ball", "Heavy-ball", HEAVY, HEAVY, 4)]
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
    body = text(20, 32, title, 16)
    body += text(20, 52, "Length-normalized accuracy, in %. Higher is better.", 12, MUTED)
    lx = 20
    for arm, name, fill, stroke, r in arms:
        body += '<circle cx="%d" cy="80" r="5" fill="%s" stroke="%s" stroke-width="1.5"/>' % (lx + 5, fill, stroke)
        body += text(lx + 16, 84, name, 12, INK)
        lx += 130
    for v in ticks(lo, hi, 10):
        body += '<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s"/>' % (x(v), top, x(v), bottom, GRID)
        body += text(x(v), bottom + 18, "%d" % v, 11, MUTED, "middle")
    for i, (label, vals) in enumerate(rows):
        y = top + ROW * i + ROW / 2
        body += text(LEFT - 16, y + 4, label, 13, INK, "end")
        for arm, name, fill, stroke, r in arms:
            body += '<circle cx="%.1f" cy="%.1f" r="%d" fill="%s" stroke="%s" stroke-width="1.5"/>' % (
                x(vals[arm]), y, r, fill, stroke)
    return svg(bottom + 40, body)


def main():
    result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out, prefix, title = Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    out.mkdir(parents=True, exist_ok=True)
    (out / (prefix + "-delta.svg")).write_text(delta_figure(result["tasks"], title + ": change vs. baseline"), encoding="utf-8")
    (out / (prefix + "-accuracy.svg")).write_text(accuracy_figure(result["tasks"], title + ": accuracy"), encoding="utf-8")
    print("wrote", prefix + "-delta.svg", "and", prefix + "-accuracy.svg")


if __name__ == "__main__":
    main()
