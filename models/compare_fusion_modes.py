#!/usr/bin/env python3
"""Decide the fusion mode empirically.

Loads the autoencoder's benign-val and per-family attack reconstruction errors,
then scores BOTH the one-sided and two-sided flag on every attack window and the
benign windows, reporting the attack detection rate and the benign false-alarm
rate each would contribute through fusion. This is the number that matters,
because the AE only ever contributes w_ae * flag to the fused score.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main() -> None:
    ART = ROOT / "models/artifacts"
    cfg = json.loads((ART / "lstm_ae_config.json").read_text())

    # benign val errors (the calibrated split)
    from ae_utils import load_benign_val_errors
    val = load_benign_val_errors()
    med, iqr = float(np.median(val)), float(np.subtract(*np.percentile(val, [75, 25])))

    # attack errors from the saved per-family means * window counts -> a weighted
    # replays of the actual distribution is impossible cheaply, so we use each
    # family's mean as a point mass and weight by window count.
    ae = json.loads((ART / "ae_metrics.json").read_text())
    fam = ae["per_family"]
    fam_means = [(k, v["err_mean"], v["windows"]) for k, v in fam.items()]

    print(f"benign val: n={len(val)}, median={med:.4f}, IQR={iqr:.4f}")
    print(f"{'family':<28}{'err':>7}{'win':>8} | {'1-sided':>8}{'2-sided':>8}")
    rows = {"one": [], "two": []}
    for name, e, w in sorted(fam_means, key=lambda x: -x[1] * x[2]):
        for mode in ("one", "two"):
            if mode == "one":
                thr = float(np.quantile(val, 0.90))
                flag = sigmoid((e - thr) / max(float(np.std(val)) * 0.7, 1e-6))
            else:
                dist = abs(e - med) / iqr
                flag = sigmoid(dist - float(np.quantile(np.abs(val - med) / iqr, 0.90)))
            rows[mode].append((name, e, w, flag))
        print(f"{name[:28]:<28}{e:>7.3f}{w:>8,} | "
              f"{[x[3] for x in rows['one'] if x[0]==name][0]:>8.3f}"
              f"{[x[3] for x in rows['two'] if x[0]==name][0]:>8.3f}")

    for mode in ("one", "two"):
        flagged = sum(w * max(f, 0.15) for _, e, w, f in rows[mode])
        total = sum(w for _, _, w, _ in rows[mode])
        print(f"\n{mode}-sided: weighted mean AE flag over all attacks "
              f"= {flagged/total:.3f}")

    # benign false-alarm contribution through the SAME flag
    print("\nbenign flag-rate (share > 0.5):")
    for mode, thr in (("one", np.quantile(val, 0.90)), ("two", None)):
        if mode == "one":
            flags = sigmoid((val - thr) / max(float(np.std(val)) * 0.7, 1e-6))
        else:
            flags = sigmoid(np.abs(val - med) / iqr -
                            float(np.quantile(np.abs(val - med) / iqr, 0.90)))
        print(f"   {mode:>4}: {(flags > 0.5).mean():.3f}  (benign errs over flag threshold)")


if __name__ == "__main__":
    main()
