#!/usr/bin/env python3
"""SIH26145 — assemble evaluation/REPORT.md from every artifact in the run.

Pulls together the four things a reviewer needs to trust the numbers:
  1. diode integrity proof (is the link genuinely one-way?)
  2. data provenance (which attack windows, how much of each slice crossed?)
  3. model metrics (per class, plus the binary operating point)
  4. measured inference latency

Usage: python evaluation/make_report.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "models/artifacts"


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:                                        # noqa: BLE001
        return None


def _jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def section_diode(L: list[str]) -> None:
    proof = ROOT / "data/diode/proof/result.txt"
    L.append("## 1. Diode integrity\n")
    if not proof.exists():
        L.append("_no proof on record — run `make diode`_\n")
        return
    L.append("```")
    L.append(proof.read_text().strip())
    L.append("```")
    L.append("\nThe monitor namespace has an unconditional `iptables OUTPUT DROP`, so no "
             "reverse-direction feature can exist. `bwd_*` columns are absent by "
             "construction rather than filtered out after the fact.\n")


def section_data(L: list[str]) -> None:
    L.append("## 2. Data provenance\n")
    wins = _read(ROOT / "data/windows.json")
    if wins:
        n = sum(len(e["windows"]) for e in wins.values())
        L.append(f"Attack windows derived from the CICIDS2017 label CSVs: **{n}** across "
                 f"{len(wins)} day captures.\n")
        L.append("CSV timestamps required two corrections, both verified against "
                 "attacker-IP packet bursts located directly in the raw pcaps: the hour "
                 "field is a 12-hour clock with no AM/PM marker (hours 1-7 are "
                 "afternoon), and the wallclock is Halifax local time while pcap frames "
                 "are UTC (+3h). Windows are bounded by the 1st/99th percentile of each "
                 "label's flow times, and slices are cut at the densest 4-minute "
                 "sub-window so a slice is attack-dominated rather than mostly benign.\n")

    plan = _read(ROOT / "data/plan.json") or []
    caps = _jsonl(ROOT / "data/captures/manifest.jsonl")
    if caps:
        by = {c["slice_id"]: c for c in caps}
        L.append(f"Replayed **{len(caps)}/{len(plan)}** slices through the emulated "
                 "diode at 1x original timing.\n")
        L.append("| slice | label | sent | captured | delivery |")
        L.append("|---|---|---:|---:|---:|")
        for c in caps:
            L.append(f"| `{c['slice_id']}` | {c['label']} | {c['n_packets']:,} "
                     f"| {c['captured_packets']:,} | {c['delivery_ratio']:.3%} |")
        worst = min(caps, key=lambda c: c["delivery_ratio"])
        mean = sum(c["delivery_ratio"] for c in caps) / len(caps)
        L.append(f"\nMean delivery {mean:.3%}; worst slice `{worst['slice_id']}` at "
                 f"{worst['delivery_ratio']:.3%}.\n")
        del by

    feats = sorted((ROOT / "data/features").glob("extract_*.json"))
    if feats:
        tot_f = tot_s = tot_q = bad = 0
        for f in feats:
            d = _read(f) or {}
            tot_f += d.get("flow_windows", 0)
            tot_s += d.get("src_windows", 0)
            tot_q += (d.get("sequences") or {}).get("windows", 0)
            bad += d.get("malformed_pkts", 0)
        L.append(f"Forward-only extraction over {len(feats)} slices: "
                 f"**{tot_f:,}** flow-window rows, **{tot_s:,}** source-bucket rows, "
                 f"**{tot_q:,}** sequence windows, {bad:,} unparseable frames.\n")


def section_classifier(L: list[str]) -> None:
    m = _read(ART / "classical_metrics.json")
    L.append("## 3. Known-attack classifier (XGBoost)\n")
    if not m:
        L.append("_not trained yet_\n")
        return
    sp = m.get("split", {})
    L.append(f"- {m['n_rows']:,} rows — train {m['n_train']:,} / test {m['n_test']:,}")
    L.append(f"- split: **{sp.get('kind', 'n/a')}** "
             f"(tail {sp.get('test_frac')} of each slice's own timeline, so every class "
             "is present on both sides and the model is always scored on later traffic)")
    L.append(f"- clock features excluded; raw IP octets "
             f"{'INCLUDED' if sp.get('uses_ip_features') else 'excluded'} "
             "(CICIDS2017 sources almost every attack from one host, so address "
             "features would make this an IP blocklist)")
    L.append(f"- **macro-F1 {m['macro_f1']}**, weighted-F1 {m.get('weighted_f1')}, "
             f"accuracy {m['accuracy']} over {m.get('n_test_classes')} test classes\n")

    L.append("| class | precision | recall | F1 | support | PR-AUC |")
    L.append("|---|---:|---:|---:|---:|---:|")
    pr = m.get("pr_auc_per_class", {})
    for cls, v in sorted(m["per_class"].items(),
                         key=lambda kv: -kv[1].get("support", 0)):
        L.append(f"| {cls} | {v['precision']} | {v['recall']} | {v['f1']} "
                 f"| {v.get('support', '—')} | {pr.get(cls, '—')} |")

    b = m.get("binary_attack_detection") or {}
    if b:
        L.append(f"\n**Binary attack/benign operating point** — {b['definition']}:\n")
        L.append(f"- ROC-AUC **{b['roc_auc']}**, PR-AUC **{b['pr_auc']}**")
        L.append(f"- at threshold {b['threshold_at_recall95']} → recall "
                 f"{b['recall']:.3f}, precision {b['precision']:.3f}, "
                 f"**FPR {b['fpr']:.4f}**")
        L.append(f"- TP {b['tp']:,} · FP {b['fp']:,} · TN {b['tn']:,} · FN {b['fn']:,}\n")
    if m.get("shap_top_features"):
        L.append("Top attributions: " +
                 ", ".join(f"`{f}`" for f in m["shap_top_features"][:8]) + "\n")


def section_ae(L: list[str]) -> None:
    a = _read(ART / "ae_metrics.json")
    cfg = _read(ART / "lstm_ae_config.json")
    L.append("## 4. Novel-threat detector (LSTM-Autoencoder, benign-only training)\n")
    if not a or not cfg:
        L.append("_not trained yet_\n")
        return
    L.append(f"- trained on {a.get('n_benign_train', 0):,} benign windows, "
             f"calibrated on {a.get('n_benign_val', 0):,} held-out benign windows")
    L.append(f"- split: {a.get('split')}")
    L.append(f"- threshold (95th pct of benign error): `{cfg['threshold']:.5f}` → "
             f"benign flag-rate {a.get('benign_flag_rate_at_thr')}")
    L.append(f"- separation ROC-AUC **{a.get('auc', '—')}** "
             f"(two-sided {a.get('auc_two_sided', '—')}), "
             f"attack detection at threshold {a.get('detection_rate_at_thr', '—')}\n")
    if a.get("warning"):
        L.append(f"> **Caveat.** {a['warning']}\n")
    fam = a.get("per_family") or {}
    if fam:
        L.append("| attack family | windows | mean error | flagged at threshold |")
        L.append("|---|---:|---:|---:|")
        for k, v in sorted(fam.items(), key=lambda kv: -kv[1]["flagged_at_thr"]):
            L.append(f"| {k} | {v['windows']:,} | {v['err_mean']} "
                     f"| {v['flagged_at_thr']:.2%} |")
        L.append("")


def section_latency(L: list[str]) -> None:
    L.append("## 5. Inference latency\n")
    bench = _read(ROOT / "evaluation/latency.json")
    if bench:
        L.append(f"Measured over {bench['n']:,} windows on an idle host "
                 f"({bench['device']}), full fuse path (classifier + autoencoder + "
                 "fusion):\n")
        L.append(f"- mean **{bench['mean_ms']} ms**, p50 {bench['p50_ms']} ms, "
                 f"p95 **{bench['p95_ms']} ms**, p99 {bench['p99_ms']} ms, "
                 f"max {bench['max_ms']} ms")
        L.append(f"- gate (<10 ms/window): "
                 f"**{'PASS' if bench['p95_ms'] < 10 else 'FAIL'}**\n")
        if bench.get("breakdown"):
            L.append("| stage | mean ms |")
            L.append("|---|---:|")
            for k, v in bench["breakdown"].items():
                L.append(f"| {k} | {v} |")
            L.append("")
        return
    h = _read(ROOT / "data/demo_health.json")
    if h:
        L.append(f"- avg {h.get('avg_latency_ms')} ms/window, p95 "
                 f"{h.get('p95_latency_ms')} ms (device {h.get('device')})\n")
    else:
        L.append("_no benchmark on record — run `make bench`_\n")


def section_fusion(L: list[str]) -> None:
    L.append("## 4b. Alert fusion (classifier + autoencoder)\n")
    f = _read(ART / "fusion_config.json")
    if not f:
        L.append("_not calibrated yet_\n")
        return
    mode = f.get("mode", "one-sided")
    L.append(f"- `score = {f.get('w_clf', 0.65)}·p_classifier + "
             f"{f.get('w_ae', 0.35)}·AE_flag`")
    L.append(f"- AE flag: `{mode}` sigmoid of reconstruction error "
             f"(threshold `{f.get('thr')}`, scale `{f.get('scale')}`)")
    L.append(f"- ±: benign flag-rate **{f.get('benign_flag_rate')}**, attack "
             f"flag at mean error **{f.get('attack_flag_at_mean_err')}**")
    fam = f.get("per_family_flag") or {}
    if fam:
        L.append("| family | AE flag |")
        L.append("|---|---:|")
        for k, v in sorted(fam.items(), key=lambda kv: -kv[1]):
            L.append(f"| {k} | {v} |")
        L.append("")
    L.append("> The AE is the novelty channel; the classifier carries the known "
             "families. Weights can be retuned by editing `fusion_config.json` "
             "(see `models/calibrate_fusion.py`).\n")


def main() -> None:
    L = ["# SIH26145 — Evaluation Report",
         "",
         "AI-based detection of cyber threats in unidirectional IP traffic.",
         f"_generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}_",
         ""]
    section_diode(L)
    section_data(L)
    section_classifier(L)
    section_ae(L)
    section_fusion(L)
    section_latency(L)
    L += ["## Artifacts\n",
          "| file | purpose |", "|---|---|",
          "| `models/artifacts/classical_confusion.png` | row-normalised confusion matrix |",
          "| `models/artifacts/classical_shap.png` | feature attribution |",
          "| `models/artifacts/ae_errors.png` | benign vs attack reconstruction error |",
          "| `docs/ATTACK_WINDOWS.md` | derived attack windows, pcap time base |",
          "| `data/captures/manifest.jsonl` | per-slice label + delivery ground truth |",
          ""]
    out = ROOT / "evaluation/REPORT.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
