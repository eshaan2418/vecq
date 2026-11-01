#!/usr/bin/env python3
"""End-to-end benchmark: vecq vs sqlite3 on the same synthetic CSV.

For every query, results from vecq and sqlite3 are compared (floats within
1e-6 relative) before any timing is reported. Query time excludes loading for
both engines; load/import time is reported separately.

usage: bench/e2e.py --rows 10000000 --runs 7 --out bench/results
"""

import argparse
import csv
import io
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time

QUERIES = {
    "Q1 filter+count": "SELECT COUNT(*) FROM t WHERE fare > 20",
    "Q2 filter+sum/avg": "SELECT SUM(fare), AVG(tip) FROM t WHERE distance > 2.5 AND passenger_count >= 2",
    "Q3 group by (5 groups)": "SELECT payment, COUNT(*), AVG(fare) FROM t GROUP BY payment",
    "Q4 filter+group+top-k": (
        "SELECT zone, SUM(tip) FROM t WHERE pickup_hour >= 17 AND pickup_hour < 20 "
        "GROUP BY zone ORDER BY SUM(tip) DESC LIMIT 5"
    ),
    "Q5 string filter+2-key group": (
        "SELECT vendor, pickup_hour, MAX(fare), MIN(distance) FROM t WHERE payment = 'card' "
        "GROUP BY vendor, pickup_hour"
    ),
}

SCHEMA = (
    "CREATE TABLE t(trip_id INTEGER, vendor TEXT, pickup_hour INTEGER, passenger_count INTEGER, "
    "distance REAL, fare REAL, tip REAL, payment TEXT, zone TEXT);"
)


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def parse_rows(text):
    rows = list(csv.reader(io.StringIO(text)))
    return rows[1:] if rows else []


def norm(cell):
    if cell == "":
        return None
    try:
        return float(cell)
    except ValueError:
        return cell


def same(a, b):
    if len(a) != len(b):
        return False
    key = lambda r: [str(x) for x in r]
    for ra, rb in zip(sorted(a, key=key), sorted(b, key=key)):
        if len(ra) != len(rb):
            return False
        for x, y in zip(map(norm, ra), map(norm, rb)):
            if isinstance(x, float) and isinstance(y, float):
                if abs(x - y) > 1e-6 * max(1.0, abs(x), abs(y)):
                    return False
            elif x != y:
                return False
    return True


def vecq_times(binary, data, q, runs, extra):
    p = sh([binary, "--csv", "--timing", "--repeat", str(runs), *extra, data, q])
    times = [float(m) for m in re.findall(r"query: ([0-9.]+) ms", p.stderr)]
    load = float(re.search(r"load: ([0-9.]+) ms", p.stderr).group(1))
    return times, load, parse_rows(p.stdout)


def sqlite_run(data, runs):
    script = [SCHEMA, f".import --csv --skip 1 {data} t",
              "UPDATE t SET passenger_count = NULL WHERE passenger_count = '';",
              ".headers on", ".mode csv", ".timer on"]
    for q in QUERIES.values():
        script += [q + ";"] * runs
    t0 = time.perf_counter()
    p = sh(["sqlite3", ":memory:"], input="\n".join(script) + "\n")
    total = time.perf_counter() - t0
    times = [float(x) * 1000 for x in re.findall(r"Run Time: real ([0-9.]+)", p.stdout)]
    # First timer line is the UPDATE; then runs per query.
    update_ms, times = times[0], times[1:]
    body = re.sub(r"Run Time: real [^\n]*\n", "\x00", p.stdout)
    chunks = [c for c in body.split("\x00")]
    outputs = chunks[1:]  # chunk 0 precedes the UPDATE's timer line
    per_q = {}
    for i, name in enumerate(QUERIES):
        per_q[name] = (times[i * runs:(i + 1) * runs], parse_rows(outputs[i * runs]))
    import_ms = total * 1000 - sum(times)
    return per_q, import_ms, update_ms


def summarize(xs):
    xs = sorted(xs)
    return {"median_ms": statistics.median(xs), "min_ms": xs[0], "max_ms": xs[-1], "runs": len(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10_000_000)
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--build", default="build")
    ap.add_argument("--data", default="bench/data")
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()

    os.makedirs(args.data, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    data = os.path.join(args.data, f"trips_{args.rows}.csv")
    if not os.path.exists(data):
        with open(data, "w") as f:
            subprocess.run([os.path.join(args.build, "vecq-gen"), str(args.rows), "42"], stdout=f, check=True)
    binary = os.path.join(args.build, "vecq")

    env = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": sh(["sysctl", "-n", "machdep.cpu.brand_string"]).stdout.strip()
        if sys.platform == "darwin" else platform.processor(),
        "ncpu": os.cpu_count(),
        "os": platform.platform(),
        "sqlite": sh(["sqlite3", "--version"]).stdout.split()[0],
        "compiler": sh(["c++", "--version"]).stdout.splitlines()[0],
        "rows": args.rows,
        "csv_bytes": os.path.getsize(data),
        "runs": args.runs,
        "threads": args.threads,
    }
    print(json.dumps(env, indent=2))

    sq, import_ms, update_ms = sqlite_run(data, args.runs)
    configs = {
        f"vecq neon {args.threads}T": ["--threads", str(args.threads)],
        "vecq neon 1T": ["--threads", "1"],
        "vecq scalar 1T": ["--threads", "1", "--no-simd"],
    }
    results = {"env": env, "sqlite_import_ms": import_ms, "queries": {}}
    loads = {}
    for name, q in QUERIES.items():
        entry = {"sql": q, "sqlite": summarize(sq[name][0])}
        for cfg, extra in configs.items():
            times, load, rows = vecq_times(binary, data, q, args.runs, extra)
            if not same(rows, sq[name][1]):
                print(f"MISMATCH {name} {cfg}\nvecq: {rows}\nsqlite: {sq[name][1]}", file=sys.stderr)
                sys.exit(1)
            entry[cfg] = summarize(times)
            loads.setdefault(cfg, []).append(load)
        entry["results_match_sqlite"] = True
        results["queries"][name] = entry
        print(name, {k: round(v["median_ms"], 2) for k, v in entry.items() if isinstance(v, dict)})
    results["vecq_load_ms"] = {k: summarize(v) for k, v in loads.items()}

    with open(os.path.join(args.out, "e2e.json"), "w") as f:
        json.dump(results, f, indent=2)

    cols = ["sqlite"] + list(configs)
    lines = ["| Query | " + " | ".join(cols) + f" | speedup vs sqlite ({list(configs)[0]}) |",
             "|---" * (len(cols) + 2) + "|"]
    for name, e in results["queries"].items():
        cells = [f"{e[c]['median_ms']:.2f}" for c in cols]
        sp = e["sqlite"]["median_ms"] / e[list(configs)[0]]["median_ms"]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {sp:.1f}x |")
    lines.append("")
    lines.append(f"Median query time in ms over {args.runs} runs. Load: sqlite `.import` {import_ms / 1000:.1f} s; "
                 + "; ".join(f"{k} {v['median_ms'] / 1000:.2f} s" for k, v in results["vecq_load_ms"].items()) + ".")
    b = env["csv_bytes"]
    lines.append("")
    lines.append(f"Dataset: {env['rows']} rows, {b} bytes ({b / 1e6:.0f} MB). Threads: {env['threads']}. "
                 f"Machine: {env['machine']}, {env['os']}, {env['compiler']}, SQLite {env['sqlite']}.")
    with open(os.path.join(args.out, "e2e.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
