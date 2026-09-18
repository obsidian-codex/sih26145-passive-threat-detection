#!/usr/bin/env python3
"""SIH26145 — Cut replay slices from the day pcaps according to attack windows.

For each attack label we cut the DENSEST `peak_*` sub-window computed by
replay/build_windows.py, not the label's min..max span. CICIDS2017 attack
labels have straggler flows hours away from the real activity (PortScan spans
16:05-18:23 but 126k of its 158k flows land inside four minutes at 17:52), so
cutting from the raw span produced slices that were almost entirely benign.

Benign slices are drawn from EVERY day (not just Monday) at times that do not
overlap any attack window, so the benign class reflects the same mix of hosts
and services the attacks are embedded in rather than one day's traffic profile.

Slices are cut with editcap -A/-B, whose timestamps are interpreted in the
local timezone; we pass UTC and require TZ=UTC (the box is Etc/UTC, and we set
it explicitly in the subprocess env to stay correct on other machines).

Usage: python replay/make_slices.py [--windows data/windows.json]
"""
import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BENIGN_SLICE_MIN = 5.0      # length of one benign slice
BENIGN_PER_DAY = 2          # benign slices cut from each day pcap
ATTACK_CTX_MIN = 0.5        # context padding on both sides of an attack peak
GUARD_MIN = 10.0            # keep benign slices this far from any attack window


def _capinfos(pcap: Path, flag: str) -> str:
    # -M forces exact machine-readable values; without it capinfos abbreviates
    # ("29 k" for 29889), which silently rounds every packet count.
    r = subprocess.run(["capinfos", "-M", flag, str(pcap)],
                       capture_output=True, text=True)
    return r.stdout + r.stderr


def _num(pcap: Path, flag: str, key: str) -> float:
    """Pull one numeric capinfos field.

    Values arrive as e.g. "Capture duration:    299.938823 seconds", so we take
    the first numeric token after the colon rather than trying to strip units
    (rstrip("s") leaves "299.938823 second" and fails to parse).
    """
    for line in _capinfos(pcap, flag).splitlines():
        if key.lower() in line.lower():
            m = re.search(r"[-+]?\d*\.?\d+", line.split(":", 1)[-1])
            if m:
                return float(m.group())
    return 0.0


def n_packets(pcap: Path) -> int:
    return int(_num(pcap, "-c", "Number of packets"))


def duration_s(pcap: Path) -> float:
    return _num(pcap, "-u", "Capture duration")


def slug(s: str) -> str:
    out = "".join(c if c.isalnum() else "-" for c in s.lower())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def utc(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap-dir", default="datasets/raw")
    ap.add_argument("--windows", default="data/windows.json")
    ap.add_argument("--out-dir", default="replay/slices")
    ap.add_argument("--plan-out", default="data/plan.json")
    ap.add_argument("--force", action="store_true", help="re-cut slices that exist")
    args = ap.parse_args()

    db = json.loads(Path(args.windows).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan: list[dict] = []
    for pcap_name, entry in sorted(db.items()):
        pcap = Path(args.pcap_dir) / pcap_name
        if not pcap.exists():
            print(f"!! missing {pcap}, skipped")
            continue

        wins = entry.get("windows", [])
        for w in wins:
            pad = ATTACK_CTX_MIN * 60
            plan.append({"day": pcap_name, "src_pcap": str(pcap), "label": w["label"],
                         "t_start": int(w["peak_start"] - pad),
                         "t_end": int(w["peak_end"] + pad),
                         "expect_flows": w["peak_flows"]})

        # benign slices: inside the capture, GUARD_MIN away from every attack
        span = entry.get("benign_span")
        if not span:
            continue
        b0, b1 = span
        blocked = [(w["start"] - GUARD_MIN * 60, w["end"] + GUARD_MIN * 60) for w in wins]
        placed = 0
        # walk candidate starts across the day, keep the first BENIGN_PER_DAY that fit
        step = BENIGN_SLICE_MIN * 60
        t = b0 + 600.0
        while t + step < b1 and placed < BENIGN_PER_DAY:
            if not any(t < hi and (t + step) > lo for lo, hi in blocked):
                plan.append({"day": pcap_name, "src_pcap": str(pcap), "label": "BENIGN",
                             "t_start": int(t), "t_end": int(t + step),
                             "expect_flows": None})
                placed += 1
                t += (b1 - b0) / (BENIGN_PER_DAY + 1)      # spread across the day
            else:
                t += step
        if placed < BENIGN_PER_DAY:
            print(f"   note: only placed {placed}/{BENIGN_PER_DAY} benign slices "
                  f"for {pcap_name}")

    plan.sort(key=lambda p: (p["day"], p["t_start"]))

    env = {**os.environ, "TZ": "UTC"}
    manifest, failed = [], []
    for i, p in enumerate(plan):
        sid = f"{p['day'].split('-')[0].lower()}__{slug(p['label'])}__{i}"
        dst = out_dir / f"{sid}.pcap"
        if args.force or not dst.exists() or n_packets(dst) == 0:
            subprocess.run(["editcap", "-A", utc(p["t_start"]), "-B", utc(p["t_end"]),
                            p["src_pcap"], str(dst)], check=True, env=env)
        n, dur = n_packets(dst), duration_s(dst)
        manifest.append({**p, "slice_id": sid, "slice_path": str(dst),
                         "n_packets": n, "duration_s": round(dur, 1)})
        flag = "" if n else "   <-- EMPTY"
        print(f"  {sid:<44} {n:>9,} pkts {dur:>7.1f}s  [{p['label']}]{flag}")
        if not n:
            failed.append(sid)

    Path(args.plan_out).write_text(json.dumps(manifest, indent=2))
    tot = sum(m["duration_s"] for m in manifest)
    print(f"\nplan: {len(manifest)} slices "
          f"({sum(1 for m in manifest if m['label'] != 'BENIGN')} attack / "
          f"{sum(1 for m in manifest if m['label'] == 'BENIGN')} benign), "
          f"{sum(m['n_packets'] for m in manifest):,} packets, "
          f"{tot/60:.1f} min replay @1x -> {args.plan_out}")
    if failed:
        print(f"!! {len(failed)} EMPTY slices: {failed}")


if __name__ == "__main__":
    main()
