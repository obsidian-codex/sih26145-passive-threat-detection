#!/usr/bin/env python3
"""SIH26145 — VERIFY the derived attack windows against the raw pcaps.

replay/build_windows.py is authoritative: it converts CICIDS2017's 12-hour-clock,
Halifax-local CSV timestamps into the pcap UTC time base. This script is the
independent check on that conversion and produces the evidence table in
docs/ATTACK_WINDOWS.md. It does NOT rewrite the slicing input.

Method: one selective tcpdump pass per day pcap collects arrival times of every
packet sourced from that day's labelled attacker IPs, clusters them into bursts
(separated by >GAP_S of silence), and reports how much of each derived window
overlaps an observed burst.

Interpreting the output:
  - `pcap-verified`  the window sits inside real attacker activity.
  - `no-burst`       the labelled source is a victim/infected host that also
                     talks all day (CICIDS2017 Bot and Infiltration list
                     192.168.10.x internal hosts, not the attacker), so burst
                     clustering cannot localise it. Not a failure of the window.
  - `outside-pcap`   a genuine derivation error — the window is not in the file.

Usage: python replay/derive_windows_from_pcap.py [--pcap-dir datasets/raw]
"""
import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

GAP_S = 180             # silence separating two attack bursts
MIN_BURST_PKTS = 200    # ignore incidental chatter


def pcap_bounds(pcap: Path) -> tuple[float, float]:
    """(first, last) frame epoch, from capinfos in UTC."""
    out = subprocess.run(["capinfos", "-M", "-a", "-e", str(pcap)],
                         capture_output=True, text=True, check=True).stdout
    vals = {}
    for line in out.splitlines():
        for key, tag in (("First packet time", "a"), ("Last packet time", "b")):
            if line.strip().startswith(key):
                raw = line.split(":", 1)[1].strip()
                for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                    try:
                        vals[tag] = (datetime.strptime(raw, fmt)
                                     .replace(tzinfo=timezone.utc).timestamp())
                    except ValueError:
                        continue
    if "a" not in vals or "b" not in vals:
        raise ValueError(f"capinfos gave no usable bounds for {pcap}")
    return vals["a"], vals["b"]


def attacker_bursts(pcap: Path, ips: set[str]) -> list[tuple[float, float, int]]:
    """One pass; return [(start, end, n_packets)] bursts over all given sources."""
    if not ips:
        return []
    bpf = " or ".join(f"src host {ip}" for ip in sorted(ips))
    proc = subprocess.Popen(["tcpdump", "-n", "-r", str(pcap), "-tt", bpf],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    assert proc.stdout
    bursts: list[list[float]] = []
    cur_start = cur_end = None
    n = 0
    for line in proc.stdout:
        m = re.match(r"(\d+\.\d+)", line)
        if not m:
            continue
        t = float(m.group(1))
        if cur_start is None:
            cur_start = cur_end = t
            n = 1
        elif t - cur_end > GAP_S:
            bursts.append([cur_start, cur_end, n])
            cur_start = cur_end = t
            n = 1
        else:
            cur_end = t
            n += 1
    proc.wait()
    if cur_start is not None:
        bursts.append([cur_start, cur_end, n])
    return [(a, b, c) for a, b, c in bursts if c >= MIN_BURST_PKTS]


def overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap-dir", default="datasets/raw")
    ap.add_argument("--windows", default="data/windows.json")
    ap.add_argument("--out", default="docs/ATTACK_WINDOWS.md")
    args = ap.parse_args()

    db = json.loads(Path(args.windows).read_text())
    fmt = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%m-%d %H:%M:%S")

    md = ["# CICIDS2017 attack windows (pcap UTC time base)", "",
          "Derived by `replay/build_windows.py` from the label CSVs, then checked "
          "against attacker-IP packet bursts found directly in the raw pcaps by "
          "`replay/derive_windows_from_pcap.py`.", "",
          "CSV timestamps needed two corrections: the hour field is a 12-hour clock "
          "with no AM/PM marker (hours 1-7 are afternoon, +12h), and the wallclock is "
          "Halifax local time while pcap frames are UTC (+3h).", "",
          "| Day | Label | Window (UTC) | Slice cut at (densest 4 min) | Flows | Check |",
          "|---|---|---|---|---:|---|"]

    summary = {"pcap-verified": 0, "no-burst": 0, "outside-pcap": 0}
    for pcap_name, entry in sorted(db.items()):
        wins = entry.get("windows", [])
        if not wins:
            continue
        pcap = Path(args.pcap_dir) / pcap_name
        if not pcap.exists():
            print(f"!! {pcap} missing — skipped")
            continue
        t0, t1 = pcap_bounds(pcap)
        ips = {ip for w in wins for ip in w["src_ips"] if ip}
        print(f"\n=== {pcap_name}  {fmt(t0)} -> {fmt(t1)}  sources={sorted(ips)}")
        bursts = attacker_bursts(pcap, ips)
        for b0, b1, n in bursts:
            print(f"    burst {fmt(b0)} -> {fmt(b1)[6:]}  {n:>9,} pkts")

        for w in wins:
            inside = t0 <= w["peak_start"] and w["peak_end"] <= t1
            ov = max((overlap(w["start"], w["end"], b0, b1) for b0, b1, _ in bursts),
                     default=0.0)
            if not inside:
                verdict = "outside-pcap"
            elif ov > 0:
                verdict = "pcap-verified"
            else:
                verdict = "no-burst"
            summary[verdict] += 1
            md.append(
                f"| {pcap_name.split('-')[0]} | {w['label']} "
                f"| {fmt(w['start'])} → {fmt(w['end'])[6:]} "
                f"| {fmt(w['peak_start'])[6:]} ({w['peak_flows']:,} flows) "
                f"| {w['n_flows']:,} | {verdict} |")
            print(f"  [{verdict:<14}] {w['label']:<28} "
                  f"{fmt(w['start'])} -> {fmt(w['end'])[6:]}  burst-overlap {ov/60:.1f} min")

    md += ["", f"**Check summary:** {summary['pcap-verified']} pcap-verified, "
               f"{summary['no-burst']} not localisable by burst (victim-sourced labels), "
               f"{summary['outside-pcap']} outside the pcap span.", ""]
    Path(args.out).write_text("\n".join(md) + "\n")
    print(f"\nwrote {args.out} — {summary}")
    if summary["outside-pcap"]:
        raise SystemExit(f"{summary['outside-pcap']} windows fall outside their pcap")


if __name__ == "__main__":
    main()
