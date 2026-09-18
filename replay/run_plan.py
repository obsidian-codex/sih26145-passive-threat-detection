#!/usr/bin/env python3
"""SIH26145 — execute the replay plan end-to-end.

Reads data/plan.json, ensures the diode is up, replays every slice through it,
captures monitor-side traffic, and appends one row per slice to

    data/captures/manifest.jsonl
        slice_id, day, label, expected/captured packet counts, delivery ratio,
        replay wall-time, multiplier used

The manifest is the label ground truth for everything downstream, so it records
the delivery ratio: a slice that only partially crossed the diode would bias the
features and must be visible, not silently averaged in.

Replays default to 1x wall-clock timing because the extractor's inter-arrival
features are timing-sensitive; --multiplier is available for smoke tests only.

Usage:  python replay/run_plan.py [--plan data/plan.json] [--out-dir data/captures]
                                    [--only SUBSTR] [--multiplier 1.0] [--redo]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "replay"))
from make_slices import n_packets            # shared, -M-correct capinfos parse

MIN_DELIVERY = 0.95      # warn below this fraction of packets arriving


def diode_up() -> bool:
    r = subprocess.run(["sudo", "ip", "netns", "list"], capture_output=True, text=True)
    return "ns-source" in r.stdout and "ns-monitor" in r.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="data/plan.json")
    ap.add_argument("--out-dir", default="data/captures")
    ap.add_argument("--only", default=None, help="substring filter on label or slice_id")
    ap.add_argument("--multiplier", type=float, default=1.0)
    ap.add_argument("--redo", action="store_true", help="re-replay already-captured slices")
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    done: set[str] = set()
    if manifest_path.exists() and not args.redo:
        done = {json.loads(l)["slice_id"]
                for l in manifest_path.read_text().splitlines() if l.strip()}

    if not diode_up():
        print("[plan] diode not up -> running diode/setup_diode.sh")
        subprocess.run(["bash", str(ROOT / "diode/setup_diode.sh")], check=True)

    todo = [m for m in plan
            if not (args.only and args.only.lower()
                    not in (m["label"] + m["slice_id"]).lower())]
    eta = sum(m["duration_s"] for m in todo if m["slice_id"] not in done) / args.multiplier
    print(f"[plan] {len(todo)} slices selected, {len(done)} already captured, "
          f"ETA ~{eta/60:.0f} min at {args.multiplier}x")

    poor = []
    for i, m in enumerate(todo, 1):
        if m["slice_id"] in done:
            print(f"[{i}/{len(todo)}] {m['slice_id']} already captured, skip")
            continue
        cap = out_dir / f"{m['slice_id']}.pcap"
        print(f"\n[{i}/{len(todo)}] {m['slice_id']}  "
              f"{m['n_packets']:,} pkts / {m['duration_s']}s  [{m['label']}]")
        t0 = time.time()
        subprocess.run(["bash", str(ROOT / "replay/replay_slice.sh"),
                        m["slice_path"], str(cap), str(args.multiplier)], check=True)
        got = n_packets(cap)
        ratio = got / m["n_packets"] if m["n_packets"] else 0.0
        row = {**{k: m[k] for k in ("slice_id", "day", "label", "n_packets")},
               "captured_packets": got,
               "delivery_ratio": round(ratio, 4),
               "multiplier": args.multiplier,
               "replay_wall_s": round(time.time() - t0, 1),
               "capture": str(cap.relative_to(ROOT))}
        with manifest_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        mark = "ok" if ratio >= MIN_DELIVERY else "LOW DELIVERY"
        print(f"[{i}/{len(todo)}] {got:,}/{m['n_packets']:,} = {ratio:.1%} {mark}")
        if ratio < MIN_DELIVERY:
            poor.append((m["slice_id"], ratio))

    rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    print(f"\nDONE: {len(rows)} slices in {manifest_path} "
          f"(replay wall-time {sum(r['replay_wall_s'] for r in rows)/60:.1f} min)")
    if poor:
        print(f"!! {len(poor)} slices below {MIN_DELIVERY:.0%} delivery: "
              + ", ".join(f"{s} {r:.1%}" for s, r in poor))


if __name__ == "__main__":
    main()
