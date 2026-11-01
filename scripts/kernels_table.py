#!/usr/bin/env python3
"""Summarize bench/results/kernels.json (Google Benchmark) as a Markdown table."""
import json
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "bench/results/kernels.json"
d = json.load(open(src))
med = {b["run_name"]: b["real_time"] for b in d["benchmarks"] if b.get("aggregate_name") == "median"}
rows = ["| Kernel | rows | scalar (us) | NEON (us) | speedup |", "|---|---|---|---|---|"]
for n in ["cmp_i64", "cmp_f64", "cmp_u32", "sum_i64", "sum_f64", "max_f64", "count", "mask_to_sel"]:
    for sz in ["4096", "16777216"]:
        s, v = med[f"BM_{n}/scalar/{sz}"], med[f"BM_{n}/neon/{sz}"]
        rows.append(f"| {n} | {sz} | {s:.3f} | {v:.3f} | {s / v:.2f}x |")
print("\n".join(rows))
print("\nMedian of 5 repetitions (Google Benchmark, --benchmark_min_time=0.2s).")
