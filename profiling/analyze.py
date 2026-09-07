#!/usr/bin/env python
"""Summarize profiling/raw/*.csv (written by the PROFILE build of cis5650_boids)
and draw the README graphs.

    python profiling/analyze.py [--warmup 200] [--images images]

Per run: drop the first --warmup frames, then report mean / median / p99 / min /
stdev of the CUDA-event step time and of the wall-clock frame time, plus FPS
derived from the mean frame time and a "1% low" FPS from the slowest 1% of frames.

Outputs: profiling/summary.csv, profiling/summary.md, and PNGs in --images.
"""
import argparse
import glob
import os
import re
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LABEL_RE = re.compile(
    r"^(?P<mode>naive|scattered|coherent)_vis(?P<vis>[01])_bs(?P<bs>\d+)_cw(?P<cw>\d+)_n(?P<n>\d+)$")

# Colors: dataviz reference palette, light surface. One hue per implementation.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODE_ORDER = ["naive", "scattered", "coherent"]
MODE_COLOR = {"naive": "#2a78d6", "scattered": "#eb6834", "coherent": "#1baf7a"}
MODE_NAME = {"naive": "Naive", "scattered": "Scattered grid", "coherent": "Coherent grid"}


# ---------------------------------------------------------------- loading

def parse_header(path):
    meta = {}
    with open(path) as f:
        first = f.readline()
    if first.startswith("#"):
        for kv in first[1:].strip().split(","):
            k, _, v = kv.partition("=")
            meta[k.strip()] = v.strip()
    return meta


def worst_fraction_mean(a, frac=0.01):
    k = max(1, int(round(len(a) * frac)))
    return np.sort(a)[-k:].mean()


def load_run(path, warmup):
    label = os.path.splitext(os.path.basename(path))[0]
    m = LABEL_RE.match(label)
    if not m:
        print(f"skip {label}: name does not follow <mode>_vis<v>_bs<b>_cw<c>_n<N>")
        return None, None
    d = pd.read_csv(path, comment="#")
    if len(d) <= warmup + 10:
        print(f"skip {label}: only {len(d)} rows, warm-up is {warmup}")
        return None, None
    s = d.iloc[warmup:]
    step, frame = s.step_ms.to_numpy(), s.frame_ms.to_numpy()
    row = dict(
        label=label, mode=m["mode"], vis=int(m["vis"]), bs=int(m["bs"]), cw=int(m["cw"]), n=int(m["n"]),
        samples=len(s), total_steps=len(d),
        step_mean=step.mean(), step_median=np.median(step), step_p99=np.quantile(step, 0.99),
        step_min=step.min(), step_std=step.std(),
        frame_mean=frame.mean(), frame_median=np.median(frame), frame_p99=np.quantile(frame, 0.99),
        frame_min=frame.min(), frame_std=frame.std(),
        fps=1000.0 / frame.mean(),
        fps_low1=1000.0 / worst_fraction_mean(frame),
        step_fps=1000.0 / step.mean(),
    )
    meta = parse_header(path)
    for key, col in (("mode", "mode"), ("N", "n"), ("vis", "vis")):
        if key in meta and str(meta[key]) != str(row[col]):
            print(f"warning {label}: file header says {key}={meta[key]}, name says {row[col]}")
    return row, d


def load_all(raw_dir, warmup):
    rows, traces = [], {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.csv"))):
        row, d = load_run(path, warmup)
        if row is not None:
            rows.append(row)
            traces[row["label"]] = d
    if not rows:
        raise SystemExit(f"no usable CSVs in {raw_dir}")
    df = pd.DataFrame(rows).sort_values(["vis", "cw", "bs", "mode", "n"]).reset_index(drop=True)
    return df, traces


# ---------------------------------------------------------------- tables

def fmt_n(n):
    n = int(n)
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def sweep_table(df, vis, bs, cw):
    d = df[(df.vis == vis) & (df.bs == bs) & (df.cw == cw)]
    if d.empty:
        return None
    ns = sorted(d.n.unique())
    header = ["Boids"]
    modes = [m for m in MODE_ORDER if (d["mode"] == m).any()]
    for mode in modes:
        header += [f"{MODE_NAME[mode]} step ms", f"{MODE_NAME[mode]} FPS"]
    rows = []
    for n in ns:
        r = [fmt_n(n)]
        for mode in modes:
            x = d[(d["mode"] == mode) & (d.n == n)]
            if x.empty:
                r += ["", ""]
            else:
                x = x.iloc[0]
                r += [f"{x.step_mean:.3f}", f"{x.fps:.0f}"]
        rows.append(r)
    return md_table(header, rows)


def block_table(df, n, vis, cw):
    d = df[(df.n == n) & (df.vis == vis) & (df.cw == cw)]
    if d.bs.nunique() < 2:
        return None
    bss = sorted(d.bs.unique())
    header = ["Block size"]
    modes = [m for m in MODE_ORDER if (d["mode"] == m).any()]
    for mode in modes:
        header += [f"{MODE_NAME[mode]} step ms", f"{MODE_NAME[mode]} FPS"]
    rows = []
    for bs in bss:
        r = [str(bs)]
        for mode in modes:
            x = d[(d["mode"] == mode) & (d.bs == bs)]
            if x.empty:
                r += ["", ""]
            else:
                x = x.iloc[0]
                r += [f"{x.step_mean:.3f}", f"{x.fps:.0f}"]
        rows.append(r)
    return md_table(header, rows)


def cells_table(df, vis, bs):
    d = df[(df.vis == vis) & (df.bs == bs) & (df["mode"] != "naive")]
    if d.cw.nunique() < 2:
        return None
    ns = sorted(d.n.unique())
    header = ["Boids"]
    modes = [m for m in ("scattered", "coherent") if (d["mode"] == m).any()]
    for mode in modes:
        header += [f"{MODE_NAME[mode]} 8 cells ms", f"{MODE_NAME[mode]} 27 cells ms", "27 / 8"]
    rows = []
    for n in ns:
        r = [fmt_n(n)]
        for mode in modes:
            a = d[(d["mode"] == mode) & (d.n == n) & (d.cw == 2)]
            b = d[(d["mode"] == mode) & (d.n == n) & (d.cw == 1)]
            if a.empty or b.empty:
                r += ["", "", ""]
            else:
                a, b = a.iloc[0].step_mean, b.iloc[0].step_mean
                r += [f"{a:.3f}", f"{b:.3f}", f"{b / a:.2f}x"]
        rows.append(r)
    return md_table(header, rows)


def write_summary(df, out_dir, warmup):
    cols = ["label", "mode", "vis", "bs", "cw", "n", "samples", "total_steps",
            "step_mean", "step_median", "step_p99", "step_min", "step_std",
            "frame_mean", "frame_median", "frame_p99", "frame_min", "frame_std",
            "fps", "fps_low1", "step_fps"]
    df[cols].to_csv(os.path.join(out_dir, "summary.csv"), index=False, float_format="%.4f")

    parts = [f"# Profiling summary\n\nFirst {warmup} frames of every run dropped as warm-up. "
             "`step ms` is the CUDA-event time of one simulation step (mean). "
             "`FPS` is 1000 / mean wall-clock frame time of the main loop.\n"]
    for vis in (0, 1):
        for cw in sorted(df.cw.unique()):
            t = sweep_table(df, vis, 128, cw)
            if t:
                parts.append(f"## Boid count sweep, {'with' if vis else 'no'} visualization, "
                             f"block 128, cell width {cw}x\n\n{t}\n")
    for n in sorted(df.n.unique()):
        for vis in (0, 1):
            t = block_table(df, n, vis, 2)
            if t:
                parts.append(f"## Block size sweep, {fmt_n(n)} boids, "
                             f"{'with' if vis else 'no'} visualization\n\n{t}\n")
    t = cells_table(df, 0, 128)
    if t:
        parts.append("## 27 vs 8 neighbor cells (cell width 1x vs 2x rule distance), "
                     "no visualization, block 128\n\n" + t + "\n")

    header = ["label", "samples", "step mean", "step median", "step p99", "step min", "step std",
              "frame mean", "frame p99", "FPS", "1% low FPS"]
    rows = [[r.label, str(r.samples), f"{r.step_mean:.4f}", f"{r.step_median:.4f}", f"{r.step_p99:.4f}",
             f"{r.step_min:.4f}", f"{r.step_std:.4f}", f"{r.frame_mean:.4f}", f"{r.frame_p99:.4f}",
             f"{r.fps:.1f}", f"{r.fps_low1:.1f}"] for r in df.itertuples()]
    parts.append("## All runs (ms)\n\n" + md_table(header, rows) + "\n")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(parts))


# ---------------------------------------------------------------- plots

def style(ax, xlabel, ylabel, title=None):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, which="both")
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)


def n_axis(ax, ns):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(sorted(ns)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_n(v)))
    ax.xaxis.set_minor_formatter(NullFormatter())


def plain_log_y(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def legend(ax, **kw):
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, **kw)


def save(fig, path):
    # Retry: on Windows a thumbnailer or file watcher sometimes has the previous PNG
    # memory-mapped for a moment, and opening it for truncation fails with EINVAL/EACCES.
    for attempt in range(20):
        try:
            fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
            break
        except OSError as e:
            if e.errno not in (13, 22) or attempt == 19:
                raise
            time.sleep(0.25)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def plot_vs_n(df, images, value, ylabel, fname, vis_values=(0, 1)):
    d = df[(df.bs == 128) & (df.cw == 2) & (df.vis.isin(vis_values))]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    for mode in MODE_ORDER:
        for vis, ls, marker, suffix in ((0, "-", "o", ""), (1, "--", "s", ", with visualization")):
            s = d[(d["mode"] == mode) & (d.vis == vis)].sort_values("n")
            if s.empty:
                continue
            ax.plot(s.n, s[value], ls, color=MODE_COLOR[mode], lw=2, marker=marker, ms=5,
                    label=MODE_NAME[mode] + suffix)
    style(ax, "Boids", ylabel)
    n_axis(ax, d.n.unique())
    plain_log_y(ax)
    legend(ax)
    save(fig, os.path.join(images, fname))


def plot_block_size(df, images, n):
    d = df[(df.n == n) & (df.vis == 0) & (df.cw == 2)]
    if d.bs.nunique() < 2:
        return
    # Two separate images. FPS is what the assignment asks for. Step time relative to
    # block 128 is on a linear axis because the differences are tens of percent and
    # vanish on a log axis.
    stem = os.path.join(images, f"block-size-sweep-{fmt_n(n)}-boids")
    for value, fname in (("fps", stem + "-fps.png"), ("rel", stem + "-relative-step-time.png")):
        fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
        for mode in MODE_ORDER:
            s = d[d["mode"] == mode].sort_values("bs")
            if s.empty:
                continue
            if value == "fps":
                y = s.fps
            else:
                ref = s[s.bs == 128].step_mean
                if ref.empty:
                    continue
                y = s.step_mean / ref.iloc[0]
            ax.plot(s.bs, y, "-", color=MODE_COLOR[mode], lw=2, marker="o", ms=5, label=MODE_NAME[mode])
        if value == "fps":
            style(ax, "Block size (threads)", "FPS (from frame time)")
            plain_log_y(ax)
        else:
            style(ax, "Block size (threads)", "Step time relative to block 128")
            ax.axhline(1.0, color=AXIS, lw=1, zorder=0)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}x"))
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(sorted(d.bs.unique())))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        legend(ax)
        save(fig, fname)


def plot_cells(df, images):
    d = df[(df.vis == 0) & (df.bs == 128) & (df["mode"] != "naive")]
    if d.cw.nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    for mode in ("scattered", "coherent"):
        for cw, ls, marker, suffix in ((2, "-", "o", ", 8 cells (2x width)"), (1, "--", "s", ", 27 cells (1x width)")):
            s = d[(d["mode"] == mode) & (d.cw == cw)].sort_values("n")
            if s.empty:
                continue
            ax.plot(s.n, s.step_mean, ls, color=MODE_COLOR[mode], lw=2, marker=marker, ms=5,
                    label=MODE_NAME[mode] + suffix)
    style(ax, "Boids", "Step time, ms (CUDA events)")
    n_axis(ax, d.n.unique())
    plain_log_y(ax)
    legend(ax)
    save(fig, os.path.join(images, "27-vs-8-neighbor-cells-step-time.png"))


def plot_trace(traces, images, warmup, label=None):
    if label not in traces:
        prefer = [l for l in traces if l.startswith("coherent_vis0_bs128_cw2_n50000")]
        label = prefer[0] if prefer else sorted(traces)[0]
    d = traces[label]
    mode = LABEL_RE.match(label)["mode"]
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    ax.axvspan(0, warmup, color=GRID, alpha=0.6, lw=0)
    ax.plot(d.step, d.step_ms, "-", color=MODE_COLOR[mode], lw=1)
    style(ax, "Simulation step", "Step time, ms")
    ax.set_ylim(bottom=0)
    ax.text(warmup / 2, ax.get_ylim()[1] * 0.95, "warm-up\n(dropped)", ha="center", va="top", color=MUTED, fontsize=8)
    m = LABEL_RE.match(label)
    save(fig, os.path.join(images, f"per-step-time-trace-{m['mode']}-{fmt_n(m['n'])}.png"))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "raw"))
    ap.add_argument("--images", default=os.path.join(ROOT, "images"))
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--block-n", type=int, default=50000, help="boid count used for the block-size sweep")
    ap.add_argument("--trace", default=None, help="label of the run to draw as a per-step trace")
    args = ap.parse_args()

    df, traces = load_all(args.raw, args.warmup)
    os.makedirs(args.images, exist_ok=True)
    write_summary(df, HERE, args.warmup)
    print(f"{len(df)} runs -> profiling/summary.csv, profiling/summary.md")

    plot_vs_n(df, args.images, "fps", "FPS (1000 / mean frame time)", "framerate-vs-boid-count-block-128.png")
    plot_vs_n(df, args.images, "step_mean", "Step time, ms (CUDA events)", "step-time-vs-boid-count-block-128.png",
              vis_values=(0,))
    plot_block_size(df, args.images, args.block_n)
    plot_cells(df, args.images)
    plot_trace(traces, args.images, args.warmup, args.trace)


if __name__ == "__main__":
    main()
