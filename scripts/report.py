#!/usr/bin/env python3
"""Generate charts and markdown tables from the comparison CSVs.

Usage:
    .venv/bin/python scripts/report.py

Reads results/*.csv (written by `cargo run --release --features metal --bin
compare`) and writes results/charts/*.png plus results/tables.md.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CHARTS = RES / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 170,
        "font.size": 9.5,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
    }
)

LABELS = {
    "cpu_naive": "CPU naive (1 thread)",
    "cpu_flash_1t": "CPU flash, baseline (1 thread)",
    "cpu_cached_1t": "CPU flash, cached (1 thread)",
    "cpu_parallel": "CPU flash, parallel (all cores)",
    "gpu_flash": "GPU flash (Metal)",
    "gpu_naive": "GPU naive (S² materialized)",
}
COLORS = {
    "cpu_naive": "#8c8c8c",
    "cpu_flash_1t": "#1f77b4",
    "cpu_cached_1t": "#2ca02c",
    "cpu_parallel": "#d62728",
    "gpu_flash": "#7b52ab",
    "gpu_naive": "#ff7f0e",
}
SHORT = {
    "cpu_naive": "CPU naive 1T",
    "cpu_flash_1t": "CPU flash 1T",
    "cpu_cached_1t": "CPU cached 1T",
    "cpu_parallel": "CPU all cores",
    "gpu_flash": "GPU flash",
    "gpu_naive": "GPU naive",
}
MARKERS = {
    "cpu_naive": "v",
    "cpu_flash_1t": "o",
    "cpu_cached_1t": "s",
    "cpu_parallel": "^",
    "gpu_flash": "D",
    "gpu_naive": "P",
}


def load():
    timings = pd.read_csv(RES / "timings.csv")
    blocks = pd.read_csv(RES / "blocksize.csv")
    threads = pd.read_csv(RES / "threads.csv")
    transfer = pd.read_csv(RES / "transfer.csv")
    startup = pd.read_csv(RES / "startup.csv")
    return timings, blocks, threads, transfer, startup


def series(df, suite, impl, value="latency_ms", **filters):
    d = df[(df.suite == suite) & (df.implementation == impl)]
    for k, v in filters.items():
        d = d[d[k] == v]
    return d.set_index("seq")[value].sort_index()


def save(fig, name):
    path = CHARTS / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def fmt_ms(x):
    return f"{x:.3f}"


def chart_latency(df):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))
    ax = axes[0]
    impls = ["cpu_naive", "cpu_flash_1t", "cpu_cached_1t", "cpu_parallel", "gpu_flash", "gpu_naive"]
    for impl in impls:
        s = series(df, "scaling_d64", impl, dim=64)
        s = s[s > 0]
        if s.empty:
            continue
        ax.plot(
            s.index,
            s.values,
            marker=MARKERS[impl],
            color=COLORS[impl],
            label=LABELS[impl],
            linewidth=1.6,
            markersize=4,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length S")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Latency vs sequence length (D=64, B*H=8)")
    ax.legend(fontsize=7.5, loc="upper left")

    ax = axes[1]
    cpu1 = series(df, "scaling_d64", "cpu_flash_1t", dim=64)
    cpu10 = series(df, "scaling_d64", "cpu_parallel", dim=64)
    gpu = series(df, "scaling_d64", "gpu_flash", dim=64)
    gpu_naive = series(df, "scaling_d64", "gpu_naive", dim=64)
    common = cpu1.index.intersection(gpu.index)
    ax.plot(common, (cpu1 / gpu).loc[common], marker="D", color=COLORS["gpu_flash"],
            label="GPU flash vs CPU 1T")
    ax.plot(common, (cpu10 / gpu).loc[common], marker="^", color=COLORS["cpu_parallel"],
            label="GPU flash vs CPU all cores")
    nc = gpu_naive.index.intersection(gpu.index)
    if len(nc) > 1:
        ax.plot(nc, (gpu_naive / gpu).loc[nc], marker="P", color=COLORS["gpu_naive"],
                label="GPU flash vs GPU naive", linewidth=1.3)
    ax.axhline(1.0, color="k", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length S")
    ax.set_ylabel("speedup of GPU flash (x)")
    ax.set_title("Speedup: higher is better")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "01_latency_scaling.png")


def chart_throughput(df):
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for impl in ["cpu_flash_1t", "cpu_cached_1t", "cpu_parallel", "gpu_flash", "gpu_naive"]:
        s = series(df, "scaling_d64", impl, "tflops", dim=64)
        s = s[s > 0]
        if s.empty:
            continue
        ax.plot(
            s.index,
            s.values,
            marker=MARKERS[impl],
            color=COLORS[impl],
            label=LABELS[impl],
            linewidth=1.6,
            markersize=4,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length S")
    ax.set_ylabel("achieved TFLOP/s (useful FLOPs)")
    ax.set_title("Throughput vs sequence length (D=64, B*H=8)")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, "02_throughput.png")


def chart_causal(df):
    causal = df[df.suite == "causal"]
    noncausal = df[df.suite == "scaling_d64"]
    seqs = sorted(causal.seq.unique())
    impls = ["cpu_flash_1t", "cpu_parallel", "gpu_flash", "gpu_naive"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))

    ax = axes[0]
    width = 0.19
    x = np.arange(len(seqs))
    for i, impl in enumerate(impls):
        vals = [causal[(causal.seq == s) & (causal.implementation == impl)]["latency_ms"].mean() for s in seqs]
        ax.bar(x + (i - 1.5) * width, vals, width, color=COLORS[impl], label=LABELS[impl])
    ax.set_xticks(x)
    ax.set_xticklabels([f"S={s}" for s in seqs])
    ax.set_yscale("log")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Causal attention latency")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    cg = causal[causal.implementation == "gpu_flash"].set_index("seq")["latency_ms"].sort_index()
    ng = noncausal[noncausal.implementation == "gpu_flash"].set_index("seq")["latency_ms"].sort_index()
    c1 = causal[causal.implementation == "cpu_flash_1t"].set_index("seq")["latency_ms"].sort_index()
    n1 = noncausal[noncausal.implementation == "cpu_flash_1t"].set_index("seq")["latency_ms"].sort_index()
    cp = causal[causal.implementation == "cpu_parallel"].set_index("seq")["latency_ms"].sort_index()
    np_ = noncausal[noncausal.implementation == "cpu_parallel"].set_index("seq")["latency_ms"].sort_index()
    common = cg.index.intersection(ng.index).intersection(c1.index).intersection(n1.index)
    width = 0.26
    x = np.arange(len(common))
    for i, (num, den, label, color) in enumerate([
        (ng, cg, "GPU flash", COLORS["gpu_flash"]),
        (n1, c1, "CPU 1T", COLORS["cpu_flash_1t"]),
        (np_, cp, "CPU all cores", COLORS["cpu_parallel"]),
    ]):
        vals = (num.loc[common] / den.loc[common]).values
        ax.bar(x + (i - 1) * width, vals, width, color=color, label=label)
    ax.axhline(1.0, color="k", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S={s}" for s in common])
    ax.set_ylabel("non-causal / causal latency")
    ax.set_title("Causal masking saves work")
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    save(fig, "03_causal.png")


def chart_threads(threads_df, df):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for algorithm, color in [("cpu_flash", COLORS["cpu_cached_1t"]), ("cpu_naive", COLORS["cpu_naive"])]:
        sub = threads_df[threads_df.implementation.str.startswith(algorithm + "_p")]
        if sub.empty:
            continue
        g = sub[sub.seq == sub.seq.min()].groupby("threads")["latency_ms"].median().sort_index()
        ax = axes[0]
        ax.plot(g.index, g.values, marker="o", color=color, label=f"{algorithm} latency")
        ax = axes[1]
        ax.plot(g.index, g.iloc[0] / g.values, marker="o", color=color,
                label=f"{algorithm} speedup")
        ax.plot(g.index, g.index / g.index[0], linestyle="--", color=color, alpha=0.4,
                label=f"{algorithm} ideal")
    seq = threads_df.seq.min()
    gpu = df[(df.suite == "scaling_d64") & (df.seq == seq) & (df.implementation == "gpu_flash")]
    if len(gpu):
        axes[0].axhline(gpu.latency_ms.iloc[0], color=COLORS["gpu_flash"],
                        linestyle=":", linewidth=1.2, label="GPU flash (single call)")
        axes[1].axhline(
            threads_df[(threads_df.implementation == "cpu_flash_p1") & (threads_df.seq == seq)]["latency_ms"].iloc[0]
            / gpu.latency_ms.iloc[0],
            color=COLORS["gpu_flash"], linestyle=":", linewidth=1.2,
            label="GPU flash vs 1 CPU thread",
        )
    axes[0].set_xlabel("CPU threads")
    axes[0].set_ylabel("latency (ms)")
    axes[0].set_title(f"CPU thread scaling (S={seq}, D=64, B*H=8)")
    axes[0].legend(fontsize=7.5)
    axes[1].set_xlabel("CPU threads")
    axes[1].set_ylabel("speedup (x)")
    axes[1].set_title("Thread scaling efficiency")
    axes[1].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "04_thread_scaling.png")


def chart_blocks(blocks):
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    cpu = blocks[(blocks.backend == "cpu") & (blocks.dim == 64)]
    for impl, prefix, color in [
        ("cached", "cpu_cached_b", COLORS["cpu_cached_1t"]),
        ("parallel", "cpu_parallel_b", COLORS["cpu_parallel"]),
    ]:
        sub = cpu[cpu.implementation.str.startswith(prefix)].sort_values("block_size")
        if sub.empty:
            continue
        axes[0].plot(sub.block_size, sub.latency_ms, marker="o", color=color,
                     label=f"CPU {impl}")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("CPU block size")
    axes[0].set_ylabel("latency (ms)")
    axes[0].set_title("CPU block size sweep")
    axes[0].legend(fontsize=7.5)

    gpu = blocks[(blocks.backend == "gpu") & (blocks.dim == 64)]
    for br in sorted(gpu.query_block.unique()):
        sub = gpu[gpu.query_block == br].sort_values("block_size")
        axes[1].plot(sub.block_size, sub.latency_ms, marker="o",
                     label=f"br={br}")
    axes[1].set_xlabel("GPU keys per tile (bc)")
    axes[1].set_ylabel("per-call latency (ms)")
    axes[1].set_title("GPU K/V tile sweep (D=64)")
    axes[1].legend(fontsize=7.5)

    for dim, color in [(64, COLORS["gpu_flash"]), (128, COLORS["gpu_naive"])]:
        sub = blocks[(blocks.backend == "gpu") & (blocks.dim == dim)].sort_values("block_size")
        if sub.empty:
            continue
        axes[2].plot(sub.block_size, sub.max_abs_err, marker="o", color=color,
                     label=f"GPU D={dim}")
    axis_tol = 1e-4
    axes[2].axhline(axis_tol, color="k", linestyle="--", linewidth=0.8, label="test tolerance 1e-4")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("GPU keys per tile (bc)")
    axes[2].set_ylabel("max abs error")
    axes[2].set_title("Accuracy vs tile size")
    axes[2].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "05_block_sweep.png")


def chart_batch_heads(df):
    suite = df[df.suite == "batch_heads"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for impl in ["cpu_flash_1t", "cpu_parallel", "gpu_flash", "gpu_naive"]:
        sub = suite[suite.implementation == impl].copy()
        if sub.empty:
            continue
        sub["bh"] = sub.batch * sub.heads
        sub = sub.sort_values("bh")
        axes[0].plot(sub.bh, sub.latency_ms, marker=MARKERS[impl], color=COLORS[impl],
                     label=LABELS[impl])
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("B*H (heads in parallel)")
    axes[0].set_ylabel("latency (ms)")
    axes[0].set_title("Batch/head scaling (S=1024, D=64)")
    axes[0].legend(fontsize=7.5)

    gpu = suite[suite.implementation == "gpu_flash"].copy()
    cpu10 = suite[suite.implementation == "cpu_parallel"].copy()
    gpu["bh"] = gpu.batch * gpu.heads
    cpu10["bh"] = cpu10.batch * cpu10.heads
    gpu = gpu.groupby("bh")["latency_ms"].median().sort_index()
    cpu10 = cpu10.groupby("bh")["latency_ms"].median().sort_index()
    common = gpu.index.intersection(cpu10.index)
    axes[1].plot(common, cpu10.loc[common] / gpu.loc[common],
                 marker="D", color=COLORS["gpu_flash"], label="GPU vs CPU all cores")
    axes[1].axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("B*H")
    axes[1].set_ylabel("speedup (x)")
    axes[1].set_title("GPU speedup vs batch/heads")
    axes[1].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "06_batch_heads.png")


def chart_transfer(transfer, df):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    if not transfer.empty:
        t = transfer.sort_values("size_mib")
        axes[0].loglog(t.size_mib, t.upload_gbps, marker="o", label="host → device (memcpy)")
        axes[0].loglog(t.size_mib, t.download_gbps, marker="s", label="device → host")
        axes[0].loglog(t.size_mib, t.device_copy_gbps, marker="^", label="device → device (blit)")
        axes[0].set_xlabel("transfer size (MiB)")
        axes[0].set_ylabel("bandwidth (GB/s)")
        axes[0].set_title("Unified-memory copy bandwidth")
        axes[0].legend(fontsize=7.5)
    gpu = df[(df.suite == "scaling_d64") & (df.implementation == "gpu_flash")].sort_values("seq")
    if not gpu.empty:
        axes[1].plot(gpu.seq, gpu.latency_ms, marker="o", label="per-call (dispatch+sync)")
        axes[1].plot(gpu.seq, gpu.throughput_ms, marker="s", label="kernel only (amortized)")
        axes[1].plot(gpu.seq, gpu.e2e_ms, marker="^", label="end-to-end (upload+compute+download)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("sequence length S")
    axes[1].set_ylabel("latency (ms)")
    axes[1].set_title("GPU launch / transfer overhead")
    axes[1].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "07_transfer_overhead.png")


def chart_memory():
    s = np.array([128, 256, 512, 1024, 2048, 4096, 8192], dtype=float)
    bh, d = 8, 64
    naive_mib = bh * s * s * 4 / (1 << 20)
    flash_mib = 4 * bh * s * d * 4 / (1 << 20)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.loglog(s, naive_mib, marker="P", color=COLORS["gpu_naive"], label="naive scores (S²)")
    ax.loglog(s, flash_mib, marker="D", color=COLORS["gpu_flash"], label="flash Q/K/V/O + tiles")
    ax.axhline(8.88 * 1024, color="k", linestyle="--", linewidth=0.9,
               label="M4 maxBufferLength ≈ 8.9 GiB")
    ax.set_xlabel("sequence length S")
    ax.set_ylabel("GPU memory (MiB)")
    ax.set_title("GPU memory footprint (B*H=8, D=64)")
    ax.legend(fontsize=8)
    ax.annotate("≈16× smaller at S=4096", xy=(4096, flash_mib[-2]), xytext=(256, 8),
                arrowprops=dict(arrowstyle="->", color="0.4"), fontsize=8, color="0.3")
    fig.tight_layout()
    save(fig, "08_memory.png")


def chart_accuracy(df):
    names, errs, rels = [], [], []
    for impl in ["cpu_flash_1t", "cpu_cached_1t", "cpu_parallel", "gpu_flash", "gpu_naive"]:
        sub = df[(df.implementation == impl) & (df.suite.isin(["scaling_d64", "scaling_d128", "causal", "batch_heads", "edge"]))]
        if sub.empty or sub.max_abs_err.max() <= 0:
            continue
        names.append(LABELS[impl])
        errs.append(sub.max_abs_err.max())
        rels.append(sub.rel_l2.max())
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    x = np.arange(len(names))
    ax.bar(x - 0.18, errs, 0.36, label="max |error|", color="#4c78a8")
    ax.bar(x + 0.18, rels, 0.36, label="max relative L2", color="#f58518")
    ax.axhline(1e-4, color="k", linestyle="--", linewidth=0.9, label="1e-4 test tolerance")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("error (vs CPU naive)")
    ax.set_title("Numerical accuracy across all benchmarked shapes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "09_accuracy.png")


# ---------------------------------------------------------------------------
# Markdown tables
# ---------------------------------------------------------------------------
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def latency_table(df, suite, dim):
    sub = df[(df.suite == suite) & (df.dim == dim)]
    seqs = sorted(sub.seq.unique())
    impls = ["cpu_naive", "cpu_flash_1t", "cpu_cached_1t", "cpu_parallel", "gpu_flash", "gpu_naive"]
    headers = ["S"] + [SHORT[i] for i in impls]
    rows = []
    for s in seqs:
        vals = []
        for impl in impls:
            v = sub[(sub.seq == s) & (sub.implementation == impl)]["latency_ms"]
            vals.append(fmt_ms(v.iloc[0]) if len(v) and not np.isnan(v.iloc[0]) else "—")
        rows.append([s] + vals)
    return md_table(headers, rows)


def speedup_table(df, suite, dim):
    sub = df[(df.suite == suite) & (df.dim == dim)]
    seqs = sorted(sub.seq.unique())

    def val(s, impl, col="latency_ms"):
        v = sub[(sub.seq == s) & (sub.implementation == impl)][col]
        return float(v.iloc[0]) if len(v) and not np.isnan(v.iloc[0]) else np.nan

    headers = ["S", "GPU/CPU 1T", "GPU/CPU all cores", "GPU flash / GPU naive", "GPU flash TFLOP/s", "max abs err"]
    rows = []
    for s in seqs:
        g = val(s, "gpu_flash")
        c1 = val(s, "cpu_flash_1t")
        c10 = val(s, "cpu_parallel")
        gn = val(s, "gpu_naive")
        tf = val(s, "gpu_flash", "tflops")
        err = val(s, "gpu_flash", "max_abs_err")
        rows.append([
            s,
            f"{c1 / g:.1f}×" if g and not np.isnan(g) else "—",
            f"{c10 / g:.1f}×" if g and not np.isnan(g) else "—",
            f"{gn / g:.1f}×" if g and not np.isnan(g) and not np.isnan(gn) else "—",
            f"{tf:.2f}" if not np.isnan(tf) else "—",
            f"{err:.2e}" if not np.isnan(err) else "—",
        ])
    return md_table(headers, rows)


def causal_table(df):
    sub = df[df.suite == "causal"]
    seqs = sorted(sub.seq.unique())

    def val(s, impl, col="latency_ms"):
        v = sub[(sub.seq == s) & (sub.implementation == impl)][col]
        return float(v.iloc[0]) if len(v) else np.nan

    headers = ["S", "CPU 1T (ms)", "CPU all cores (ms)", "GPU flash (ms)", "GPU naive (ms)", "GPU/CPU all cores", "max abs err"]
    rows = []
    for s in seqs:
        g = val(s, "gpu_flash")
        c10 = val(s, "cpu_parallel")
        rows.append([
            s, fmt_ms(val(s, "cpu_flash_1t")), fmt_ms(c10), fmt_ms(g),
            fmt_ms(val(s, "gpu_naive")), f"{c10 / g:.1f}×", f"{val(s, 'gpu_flash', 'max_abs_err'):.2e}",
        ])
    return md_table(headers, rows)


def threads_table(threads):
    sub = threads[threads.seq == threads.seq.min()]
    g = sub[sub.implementation.str.startswith("cpu_flash_p")].groupby("threads")["latency_ms"].median().sort_index()
    gn = sub[sub.implementation.str.startswith("cpu_naive_p")].groupby("threads")["latency_ms"].median().sort_index()
    base = g.iloc[0] if len(g) else np.nan
    base_n = gn.iloc[0] if len(gn) else np.nan
    headers = ["threads", "flash ms", "flash speedup", "flash efficiency", "naive ms", "naive speedup"]
    rows = []
    for t in g.index:
        rows.append([
            t, fmt_ms(g[t]), f"{base / g[t]:.2f}×", f"{base / g[t] / t * 100:.0f}%",
            fmt_ms(gn[t]) if t in gn.index else "—",
            f"{base_n / gn[t]:.2f}×" if t in gn.index else "—",
        ])
    return md_table(headers, rows)


def blocks_table(blocks):
    cpu = blocks[(blocks.backend == "cpu") & (blocks.dim == 64)]
    gpu = blocks[(blocks.backend == "gpu") & (blocks.dim == 64)]
    headers = ["CPU block", "CPU cached 1T (ms)", "CPU all cores (ms)", "GPU bc", "GPU br", "GPU flash (ms)", "GPU err"]
    cpu_blocks = sorted(cpu.block_size.unique())
    rows = []
    for b in cpu_blocks:
        c1 = cpu[(cpu.block_size == b) & (cpu.implementation.str.startswith("cpu_cached"))]["latency_ms"]
        cp = cpu[(cpu.block_size == b) & (cpu.implementation.str.startswith("cpu_parallel"))]["latency_ms"]
        rows.append([b, fmt_ms(c1.iloc[0]) if len(c1) else "—", fmt_ms(cp.iloc[0]) if len(cp) else "—", "", "", "", ""])
    for _, r in gpu.sort_values(["block_size", "query_block"]).iterrows():
        rows.append(["", "", "", r.block_size, r.query_block, fmt_ms(r.latency_ms), f"{r.max_abs_err:.2e}"])
    return md_table(headers, rows)


def transfer_table(transfer):
    t = transfer.sort_values("size_mib")
    headers = ["size (MiB)", "host→device (GB/s)", "device→host (GB/s)", "device→device (GB/s)"]
    rows = []
    for _, r in t.iterrows():
        rows.append([
            f"{r.size_mib:.0f}",
            f"{r.upload_gbps:.1f}",
            f"{r.download_gbps:.1f}",
            f"{r.device_copy_gbps:.1f}",
        ])
    return md_table(headers, rows)


def accuracy_table(df):
    headers = ["implementation", "max abs error", "max relative L2", "worst case"]
    rows = []
    for impl in ["cpu_flash_1t", "cpu_cached_1t", "cpu_parallel", "gpu_flash", "gpu_naive"]:
        sub = df[df.implementation == impl]
        if sub.empty:
            continue
        idx = sub.max_abs_err.idxmax()
        worst = sub.loc[idx]
        rows.append([
            LABELS[impl],
            f"{sub.max_abs_err.max():.3e}",
            f"{sub.rel_l2.max():.3e}",
            f"S={int(worst.seq)}, D={int(worst.dim)}, causal={bool(worst.causal)}",
        ])
    return md_table(headers, rows)


def memory_table():
    headers = ["S", "naive scores (MiB)", "flash Q/K/V/O (MiB)", "ratio naive/flash"]
    rows = []
    for s in [512, 1024, 2048, 4096, 8192]:
        naive = 8 * s * s * 4 / (1 << 20)
        flash = 4 * 8 * s * 64 * 4 / (1 << 20)
        rows.append([s, f"{naive:.1f}", f"{flash:.1f}", f"{naive / flash:.1f}×"])
    return md_table(headers, rows)


def startup_table(startup):
    headers = ["item", "ms", "note"]
    rows = [
        [str(r["item"]), f"{float(r['value_ms']):.2f}", str(r["note"])]
        for _, r in startup.iterrows()
    ]
    return md_table(headers, rows)


def write_tables(df, threads, blocks, transfer, startup):
    parts = ["# Generated comparison tables", ""]
    meta = {}
    for line in (RES / "meta.txt").read_text().splitlines():
        key, _, value = line.partition("=")
        meta[key] = value
    env_rows = [
        ["GPU", meta.get("gpu_name", "?")],
        ["GPU unified memory", meta.get("gpu_unified_memory", "?")],
        ["threadgroup memory", meta.get("gpu_max_threadgroup_memory", "?") + " bytes"],
        ["max buffer", f"{int(meta.get('gpu_max_buffer_length', 0)) / (1 << 30):.2f} GiB"],
        ["CPU threads", meta.get("available_threads", "?")],
        ["Rust", meta.get("rustc", "?")],
    ]
    parts += ["## Environment", "", md_table(["item", "value"], env_rows), ""]
    parts += ["## Startup and one-time costs", "", startup_table(startup), ""]

    parts += ["## Sequence-length scaling, D=64, B*H=8 (non-causal)", "",
              "Latency in milliseconds (GPU flash = median per-call latency; CPU = median call).", "",
              latency_table(df, "scaling_d64", 64), ""]
    parts += ["### Speedup and GPU efficiency", "", speedup_table(df, "scaling_d64", 64), ""]

    parts += ["## Sequence-length scaling, D=128, B*H=8 (non-causal)", "",
              latency_table(df, "scaling_d128", 128), "", speedup_table(df, "scaling_d128", 128), ""]

    parts += ["## Causal masking, D=64, B*H=8", "", causal_table(df), ""]

    parts += ["## Batch/head scaling (S=1024, D=64)", ""]
    sub = df[df.suite == "batch_heads"]
    headers = ["B*H", "CPU 1T (ms)", "CPU all cores (ms)", "GPU flash (ms)", "GPU naive (ms)", "GPU/CPU all cores"]
    rows = []
    for bh in sorted((sub.batch * sub.heads).unique()):
        s = sub[(sub.batch * sub.heads) == bh]

        def v(impl):
            x = s[s.implementation == impl]["latency_ms"]
            return float(x.iloc[0]) if len(x) else np.nan

        rows.append([
            bh,
            f"{v('cpu_flash_1t'):.3f}",
            f"{v('cpu_parallel'):.3f}",
            f"{v('gpu_flash'):.3f}",
            f"{v('gpu_naive'):.3f}" if not np.isnan(v('gpu_naive')) else "—",
            f"{v('cpu_parallel') / v('gpu_flash'):.1f}×" if v('gpu_flash') else "—",
        ])
    parts += [md_table(headers, rows), ""]

    parts += ["## CPU thread scaling (S=%d, D=64, B*H=8)" % threads.seq.min(), "",
              threads_table(threads), ""]

    parts += ["## Block / tile-size sweep (S=1024, D=64)", "",
              blocks_table(blocks), "",
              "GPU rows: `bc` = keys per staged K/V tile, `br` = queries per threadgroup.", ""]

    parts += ["## Host/device transfer bandwidth", "", transfer_table(transfer), ""]

    parts += ["## Numerical accuracy (all benchmarked shapes)", "", accuracy_table(df), "",
              "Errors are computed against the original single-threaded CPU naive implementation.", ""]

    parts += ["## GPU memory footprint (B*H=8, D=64)", "", memory_table(), ""]

    (RES / "tables.md").write_text("\n".join(parts))
    print(f"wrote {(RES / 'tables.md').relative_to(ROOT)}")


def main():
    timings, blocks, threads, transfer, startup = load()
    chart_latency(timings)
    chart_throughput(timings)
    chart_causal(timings)
    chart_threads(threads, timings)
    chart_blocks(blocks)
    chart_batch_heads(timings)
    chart_transfer(transfer, timings)
    chart_memory()
    chart_accuracy(timings)
    write_tables(timings, threads, blocks, transfer, startup)


if __name__ == "__main__":
    main()
