#!/usr/bin/env python3
"""SIH26145 — measure detection latency per window, with a stage breakdown.

Runs the real serving code path in-process (classifier -> autoencoder -> fusion),
which is what the <10 ms/window gate is about. It deliberately does NOT go over
HTTP: loopback connection setup and uvicorn's thread hand-off add tens to
hundreds of milliseconds that belong to the transport, not the detector, and
mixing them produced a p95 of 600 ms while the detector itself was under 3 ms.

Run this on an otherwise idle host — a concurrent tcpreplay/tcpdump will
preempt the measuring thread and inflate the tail.

Usage: python evaluation/bench_latency.py [--n 500] [--out evaluation/latency.json]
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(int(len(sorted_vals) * q), len(sorted_vals) - 1)
    return round(sorted_vals[i], 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="evaluation/latency.json")
    args = ap.parse_args()

    from serving import app as A
    from drive_demo import profile, sequence
    from fusion.score import fuse

    asyncio.run(A.load_models())
    if A._state["xgb"] is None:
        raise SystemExit("no classifier artifact — train first")

    kinds = ["benign", "udp_flood", "portscan", "slowloris",
             "covert_timing", "stego_exfil", "malformed"]
    rows = [(profile(k), sequence(k)) for k in kinds]

    # warm up so lazy init is not attributed to the first measured window
    for f, s in rows:
        A.do_score(A.ScoreReq(features=f, sequence=s))

    total, t_clf, t_ae, t_fuse = [], [], [], []
    for i in range(args.n):
        feats, seq = rows[i % len(rows)]
        t0 = time.perf_counter()
        c = A.classify(feats)
        t1 = time.perf_counter()
        err = A.ae_error(seq)
        t2 = time.perf_counter()
        fuse(c["p_attack"], err, c["top_attack"], c["top_features"])
        t3 = time.perf_counter()
        t_clf.append((t1 - t0) * 1000)
        t_ae.append((t2 - t1) * 1000)
        t_fuse.append((t3 - t2) * 1000)
        total.append((t3 - t0) * 1000)

    total.sort()
    out = {
        "n": args.n,
        "device": A._state["dev"],
        "mean_ms": round(sum(total) / len(total), 3),
        "p50_ms": pct(total, 0.50), "p95_ms": pct(total, 0.95),
        "p99_ms": pct(total, 0.99), "max_ms": round(total[-1], 3),
        "breakdown": {
            "classifier": round(sum(t_clf) / len(t_clf), 3),
            "autoencoder": round(sum(t_ae) / len(t_ae), 3),
            "fusion": round(sum(t_fuse) / len(t_fuse), 3),
        },
        "note": "in-process detector path; excludes HTTP transport",
    }
    Path(ROOT / args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\ngate <10ms p95: {'PASS' if out['p95_ms'] < 10 else 'FAIL'}")


if __name__ == "__main__":
    main()
