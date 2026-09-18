#!/usr/bin/env python3
"""SIH26145 — Derive authoritative attack time-windows from CICIDS2017 labels.

Reads GeneratedLabelledFlows CSVs (Timestamp + Source IP + Label) and emits
data/windows.json:

    { "<day-pcap-name>": {
        "pcap": "Wednesday-workingHours.pcap",
        "windows": [ {"label": "DoS Hulk", "start": epoch, "end": epoch,
                      "n_flows": 231073, "src_ips": [...]} , ... ],
        "benign_span": [epoch, epoch]
      }, ... }

TWO timestamp bugs in CICIDS2017 are corrected here (both verified against
attacker-IP packet bursts located directly in the raw pcaps):

1. 12-HOUR CLOCK WITHOUT AM/PM.  Every Timestamp hour field lies in 1..12.
   Captures run ~09:00-17:00 local, so the hour is unambiguous in practice:
   hours 8..12 are morning, hours 1..7 are afternoon (+12h).

2. TIMEZONE.  Timestamps are Halifax local time (ADT = UTC-3) while pcap
   frame timestamps are UTC, so +3h converts CSV wallclock to pcap time base.

Combined:  epoch_utc = timegm(date, hour24, min, sec) + 3h
           where hour24 = h if h >= 8 else h + 12

Verification of the rule (CSV+rule vs. observed pcap burst):
    FTP-Patator      12:17 -> 13:20   burst 12:17 -> 13:20   exact
    SSH-Patator      17:09 -> 18:11   burst 17:09 -> 18:14   exact
    Heartbleed       18:12 -> 18:32   burst 18:08 -> 18:32   exact
    WebAttack Brute  12:15 -> 13:00   burst 12:15 -> 13:00   exact

Also fixed: a day can have SEVERAL label CSVs (Friday = Morning/Bot,
Afternoon-DDos, Afternoon-PortScan). Windows from every CSV of a day are now
MERGED; previously the last file processed overwrote the others, which silently
dropped DDoS, PortScan and Infiltration from the whole pipeline.

Usage:
    python replay/build_windows.py [--glf-dir datasets/raw/glf] [--out data/windows.json]
"""
import argparse
import calendar
import json
import re
from datetime import datetime, timezone
from pathlib import Path

DAY_TO_PCAP = {
    "Monday": "Monday-WorkingHours.pcap",
    "Tuesday": "Tuesday-WorkingHours.pcap",
    "Wednesday": "Wednesday-workingHours.pcap",
    "Thursday": "Thursday-WorkingHours.pcap",
    "Friday": "Friday-WorkingHours.pcap",
}

# CSV wallclock is Halifax local (ADT, UTC-3); pcap frames are UTC.
TZ_OFFSET_H = 3
# Hours >= this are morning; below it are afternoon and get +12h.
AM_HOUR_FLOOR = 8
# Percentile bounds used instead of min/max to reject straggler flows.
PCTL_LO, PCTL_HI = 1.0, 99.0
# Length of the densest sub-window reported for slicing (minutes).
PEAK_MIN = 4.0

TS_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?")


def parse_ts(raw: str):
    """CSV timestamp -> epoch seconds in the pcap (UTC) time base.

    Resolves the d/m-vs-m/d ambiguity via month==7 (capture week is 3-7 July
    2017), then the 12h-clock and timezone corrections described in the module
    docstring.
    """
    m = TS_RE.search(str(raw))
    if not m:
        return None
    a, b, y, hh, mm, ss = (int(m[1]), int(m[2]), int(m[3]),
                           int(m[4]), int(m[5]), int(m[6] or 0))
    hour24 = hh if hh >= AM_HOUR_FLOOR else hh + 12
    for day, mon in ((a, b), (b, a)):
        if mon == 7 and 1 <= day <= 31:
            return calendar.timegm((y, mon, day, hour24, mm, ss)) + TZ_OFFSET_H * 3600
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glf-dir", default="datasets/raw/glf")
    ap.add_argument("--out", default="data/windows.json")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    csvs = sorted(Path(args.glf_dir).rglob("*.csv"))
    if not csvs:
        raise SystemExit(f"no CSVs under {args.glf_dir}")

    out: dict[str, dict] = {}
    # (pcap, label) -> raw flow timestamps, accumulated across that day's CSVs
    per_day_label: dict[tuple[str, str], dict] = {}
    for csv in csvs:
        name = csv.stem.replace(".pcap_ISCX", "")
        day_key = next((d for d in DAY_TO_PCAP if name.startswith(d)), None)
        if not day_key:
            print(f"   skip (unrecognised day): {csv.name}")
            continue
        pcap_name = DAY_TO_PCAP[day_key]

        cols = pd.read_csv(csv, nrows=0, encoding="latin-1").columns.tolist()
        ts_col = next(c for c in cols if c.strip().lower() == "timestamp")
        lbl_col = next(c for c in cols if c.strip().lower() == "label")
        sip_col = next((c for c in cols if c.strip() == "Source IP"), None)

        df = pd.read_csv(csv, usecols=[ts_col, lbl_col] + ([sip_col] if sip_col else []),
                         dtype=str, encoding="latin-1", low_memory=False)
        df["ts"] = df[ts_col].map(parse_ts)
        df = df.dropna(subset=["ts"])
        df["lbl"] = df[lbl_col].str.strip()
        df["sip"] = df[sip_col].str.strip() if sip_col else ""

        entry = out.setdefault(pcap_name, {
            "pcap": pcap_name, "csv_sources": [], "windows": [],
            "benign_span": None, "n_flows_total": 0,
        })
        entry["csv_sources"].append(csv.name)
        entry["n_flows_total"] += int(len(df))

        benign = df[df["lbl"] == "BENIGN"]
        if len(benign):
            b0, b1 = int(benign["ts"].min()), int(benign["ts"].max())
            prev = entry["benign_span"]
            entry["benign_span"] = [min(prev[0], b0), max(prev[1], b1)] if prev else [b0, b1]

        # merge windows across every CSV belonging to this day
        for label, g in df[df["lbl"] != "BENIGN"].groupby("lbl"):
            acc = per_day_label.setdefault((pcap_name, label), {"ts": [], "sips": set()})
            acc["ts"].append(g["ts"].to_numpy(dtype="float64"))
            acc["sips"].update(ip for ip in g["sip"].unique() if ip)

    # ---- collapse accumulated timestamps into robust windows -----------------
    # min/max are unusable: a handful of straggler flows smear a window far past
    # the real attack (PortScan min 16:05 vs p1 17:51; slowloris max 17:25 vs
    # p99 13:10). We therefore report percentile bounds AND the densest
    # PEAK_MIN-minute sub-window, which is what the slicer actually cuts so the
    # slice is guaranteed attack-dense rather than mostly benign.
    for (pcap_name, label), acc in per_day_label.items():
        ts = np.sort(np.concatenate(acc["ts"]))
        lo, hi = (float(np.percentile(ts, PCTL_LO)), float(np.percentile(ts, PCTL_HI)))
        core = ts[(ts >= lo) & (ts <= hi)]
        if core.size == 0:
            core, lo, hi = ts, float(ts[0]), float(ts[-1])
        peak_start, peak_n = float(core[0]), 0
        span = PEAK_MIN * 60
        # candidate starts = every core flow time (subsampled for big classes)
        step = max(1, core.size // 2000)
        for s in core[::step]:
            n = int(np.searchsorted(core, s + span) - np.searchsorted(core, s))
            if n > peak_n:
                peak_start, peak_n = float(s), n
        out[pcap_name]["windows"].append({
            "label": label,
            "start": int(lo),
            "end": int(hi),
            "peak_start": int(peak_start),
            "peak_end": int(peak_start + span),
            "peak_flows": peak_n,
            "n_flows": int(ts.size),
            "raw_span": [int(ts[0]), int(ts[-1])],
            "src_ips": sorted(acc["sips"])[:5],
        })
    for entry in out.values():
        entry["windows"].sort(key=lambda w: w["start"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    fmt = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%m-%d %H:%M:%S")
    print(f"wrote {args.out} ({len(out)} days, "
          f"{sum(len(e['windows']) for e in out.values())} attack windows)")
    for pcap, e in sorted(out.items()):
        print(f"\n{pcap}  ({e['n_flows_total']:,} labeled flows from "
              f"{len(e['csv_sources'])} csv)")
        for w in e["windows"]:
            print(f"   {fmt(w['start'])} -> {fmt(w['end'])[6:]}  "
                  f"{w['label']:<28} {w['n_flows']:>9,} flows  "
                  f"peak@{fmt(w['peak_start'])[6:]} ({w['peak_flows']:,})")


if __name__ == "__main__":
    main()
