#!/usr/bin/env python3
"""SIH26145 — drive the scoring API with synthetic windows to exercise the
console without waiting for a live replay.

Not part of the detection pipeline: this is a UI/serving smoke test. It builds
feature rows in the SAME schema the extractor emits (models/artifacts/
feature_columns.json) for a benign profile and several attack profiles, posts
them to /score, and reports the verdict distribution so we can see the fusion
bands and the WebSocket push actually working.

Usage: python scripts/drive_demo.py [--api http://127.0.0.1:8200]
                                    [--rounds 3] [--delay 0.35]
"""
import argparse
import json
import math
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

ART = Path("models/artifacts")


def base_row() -> dict:
    """A quiet, well-behaved forward-only window."""
    return {
        "bucket": 0, "n_packets": 40, "bytes_total": 40 * 320,
        "duration_s": 5.0, "rate_pkts_per_s": 8.0, "rate_bytes_per_s": 2560.0,
        "pkt_size_mean": 320.0, "pkt_size_std": 90.0,
        "pkt_size_min": 60.0, "pkt_size_max": 900.0,
        "iat_mean_us": 125_000.0, "iat_std_us": 40_000.0,
        "iat_min_us": 900.0, "iat_max_us": 400_000.0,
        "payload_entropy_mean": 4.2, "payload_entropy_max": 6.0,
        "ttl_mean": 62.0, "ttl_min": 61.0, "ttl_max": 64.0,
        "n_dst_ports": 3, "n_dst_ips": 2, "port_spread": 0.08,
        "dst_port": 443, "src_port": 51000, "proto": 6.0, "proto_mask": 2.0,
        "flag_syn_frac": 0.06, "flag_ack_frac": 0.88, "flag_psh_frac": 0.22,
        "flag_fin_frac": 0.04, "flag_rst_frac": 0.0, "flag_urg_frac": 0.0,
        "retrans_like_frac": 0.01,
        "src_o12": 192.168, "src_o4": 10.5, "dst_o12": 10.200, "dst_o4": 0.2,
    }


def profile(name: str) -> dict:
    """Feature-space signature of each attack class we can articulate."""
    r = base_row()
    if name == "benign":
        return r
    if name == "udp_flood":            # volumetric: rate + uniform tiny packets
        r.update(n_packets=12_000, bytes_total=12_000 * 84, rate_pkts_per_s=2400.0,
                 rate_bytes_per_s=201_600.0, pkt_size_mean=84.0, pkt_size_std=1.0,
                 pkt_size_min=84.0, pkt_size_max=84.0, iat_mean_us=410.0,
                 iat_std_us=30.0, iat_min_us=380.0, iat_max_us=600.0, proto=17.0,
                 proto_mask=4.0, flag_syn_frac=0.0, flag_ack_frac=0.0,
                 flag_psh_frac=0.0, payload_entropy_mean=7.9, n_dst_ports=1)
        return r
    if name == "portscan":             # many ports, SYN-only, no payload
        r.update(n_packets=3200, bytes_total=3200 * 60, rate_pkts_per_s=640.0,
                 rate_bytes_per_s=38_400.0, pkt_size_mean=60.0, pkt_size_std=0.5,
                 pkt_size_min=60.0, pkt_size_max=60.0, n_dst_ports=1800,
                 port_spread=0.94, iat_mean_us=1500.0, iat_std_us=400.0,
                 flag_syn_frac=1.0, flag_ack_frac=0.0, flag_psh_frac=0.0,
                 payload_entropy_mean=0.0, payload_entropy_max=0.0)
        return r
    if name == "slowloris":            # few packets, long gaps, half-open
        r.update(n_packets=22, rate_pkts_per_s=1.2, bytes_total=22 * 120,
                 rate_bytes_per_s=144.0, iat_mean_us=2_400_000.0,
                 iat_std_us=900_000.0, iat_max_us=9_000_000.0,
                 pkt_size_mean=120.0, pkt_size_std=15.0,
                 flag_syn_frac=0.5, flag_ack_frac=0.5, flag_fin_frac=0.0,
                 dst_port=80, n_dst_ports=1)
        return r
    if name == "covert_timing":        # bimodal IATs encoding bits
        r.update(n_packets=300, rate_pkts_per_s=20.0, iat_mean_us=50_000.0,
                 iat_std_us=49_000.0, iat_min_us=1000.0, iat_max_us=100_000.0,
                 pkt_size_mean=64.0, pkt_size_std=0.0, proto=17.0, proto_mask=4.0,
                 payload_entropy_mean=0.6)
        return r
    if name == "stego_exfil":          # high-entropy payloads, steady egress
        r.update(n_packets=900, bytes_total=900 * 1400, rate_bytes_per_s=252_000.0,
                 rate_pkts_per_s=180.0, pkt_size_mean=1400.0, pkt_size_std=8.0,
                 pkt_size_max=1470.0, payload_entropy_mean=7.98,
                 payload_entropy_max=8.0, proto=17.0, proto_mask=4.0,
                 dst_port=53, n_dst_ports=1)
        return r
    if name == "malformed":            # bad TTLs/flags, impossible combinations
        r.update(n_packets=60, ttl_mean=1.0, ttl_min=1.0, ttl_max=3.0,
                 flag_syn_frac=1.0, flag_rst_frac=1.0, flag_fin_frac=1.0,
                 flag_urg_frac=1.0, pkt_size_mean=44.0, pkt_size_min=20.0,
                 retrans_like_frac=0.7, payload_entropy_mean=0.1)
        return r
    raise ValueError(name)


def sequence(kind: str, seq_len: int = 32) -> list[list[float]]:
    """[log-size, log-iat, proto, payload-entropy] rows, matching the extractor."""
    out = []
    for i in range(seq_len):
        if kind == "benign":
            size, iat, proto = random.gauss(320, 110), random.expovariate(1 / 0.12), 1.0
            ent = random.gauss(5.0, 1.2)
        elif kind == "udp_flood":
            size, iat, proto = 84.0, 0.0004, 2.0
            ent = random.gauss(0.5, 0.3)
        elif kind == "portscan":
            size, iat, proto = random.gauss(60, 15), random.expovariate(1 / 0.003), 1.0
            ent = random.gauss(1.2, 0.5)
        elif kind == "covert_timing":
            size, iat, proto = 64.0, (0.002 if i % 2 else 0.05), 2.0
            ent = random.gauss(4.5, 0.8)
        elif kind == "stego_exfil":
            size, iat, proto = 1400.0, 0.0055, 2.0
            ent = random.gauss(7.5, 0.4)
        else:
            size, iat, proto = random.gauss(200, 200), random.expovariate(1 / 0.02), 1.0
            ent = random.gauss(3.5, 1.5)
        ent = max(0.0, min(ent, 8.0))
        out.append([math.log1p(max(size, 1.0)), math.log1p(max(iat, 1e-6) * 1e6),
                     proto, ent])
    return out


def post(api: str, row: dict, seq: list[list[float]] | None) -> dict:
    body = json.dumps({"features": row, "sequence": seq}).encode()
    req = urllib.request.Request(api + "/score", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8200")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--delay", type=float, default=0.35)
    args = ap.parse_args()

    script = (["benign"] * 4 + ["udp_flood"] * 2 + ["benign"] * 3 + ["portscan"] * 2
              + ["benign"] * 2 + ["slowloris"] * 2 + ["covert_timing"] * 2
              + ["benign"] * 2 + ["stego_exfil"] * 2 + ["malformed"] + ["benign"] * 3)

    tally: dict[str, int] = {}
    lat: list[float] = []
    print(f"driving {args.api} — {args.rounds} rounds x {len(script)} windows")
    for rnd in range(args.rounds):
        for kind in script:
            try:
                out = post(args.api, profile(kind), sequence(kind))
            except urllib.error.URLError as e:
                raise SystemExit(f"cannot reach {args.api}: {e}")
            tally[out["verdict"]] = tally.get(out["verdict"], 0) + 1
            lat.append(out["latency_ms"])
            print(f"  r{rnd} {kind:<14} idx={out['threat_score']:<6} "
                  f"{out['verdict']:<9} fam={out.get('attack_family','-'):<16} "
                  f"{out['latency_ms']:>6.2f}ms")
            time.sleep(args.delay)

    lat.sort()
    print(f"\nverdicts: {tally}")
    print(f"latency  mean={sum(lat)/len(lat):.2f}ms  "
          f"p95={lat[int(len(lat)*0.95)]:.2f}ms  max={lat[-1]:.2f}ms")


if __name__ == "__main__":
    main()
