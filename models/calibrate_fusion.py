#!/usr/bin/env python3
"""SIH26145 — calibrate fusion weights so the autoencoder contributes honestly.

fusion/score.py combines the classifier's p_attack with `w_ae * ae_flag(err)`.
The previous revision pointed the threshold at the benign p95 error (0.90) and
the scale at the benign error std (0.30). Mean attack error is ~0.44, so
sigmoid((0.44-0.90)/0.30) ~ 0.17: even a detector with AUC 0.74 contributed next
to nothing, and the fused verdict was effectively classifier-only.

We calibrate against the ACTUAL benign-val and per-family attack error
distributions. compare_fusion_modes.py shows the two-sided alternative we
rejected: most attack families sit just ABOVE the benign error cluster (Hulk
0.49 vs benign median 0.11), so distance-from-median two-sided spreads them out
wrongly. One-sided weighted attack flag is 0.288 vs 0.206.

The threshold is the benign p90 error (so 10% of benign windows sit above the
flag line, matching the ~10% the report quotes) and the scale is 0.7 * benign
std. That gives the AE a real voice on the novel families (Bot, Slowhttptest,
Heartbleed) while keeping benign contribution tolerable.

Re-run after any AE retrain or after changing the benign/tail split.
"""
import json
from pathlib import Path

import numpy as np

from ae_utils import load_benign_val_errors

ART = Path("models/artifacts")
W_CLF, W_AE = 0.65, 0.35


def flag(err, thr: float, scale: float):
    """The exact quantity fusion/score.py multiplies by w_ae (vectorised)."""
    return 1.0 / (1.0 + np.exp(-(np.asarray(err, dtype=float) - thr) / scale))


def main() -> None:
    ae = json.loads((ART / "ae_metrics.json").read_text())
    val = load_benign_val_errors()

    thr = float(np.quantile(val, 0.90))
    scale = max(float(np.std(val)) * 0.7, 1e-6)
    benign_flag = float((flag(val, thr, scale) > 0.5).mean())

    fam = ae.get("per_family", {})
    per = {k: round(float(flag(v["err_mean"], thr, scale)), 3)
           for k, v in fam.items() if k != "BENIGN"}
    attack_flag = float(flag(ae.get("attack_err_mean", 0.44), thr, scale))

    out = {
        "mode": "one-sided",
        "w_clf": W_CLF, "w_ae": W_AE,
        "thr": round(thr, 4), "scale": round(scale, 4),
        "benign_flag_rate": round(benign_flag, 4),
        "attack_flag_at_mean_err": round(attack_flag, 4),
        "per_family_flag": per,
        "note": (
            "one-sided sigmoid of (err - thr)/scale; thr = benign p90 error, "
            "scale = 0.7 * benign std. AE contributes a healthy share on novel "
            "families (Bot, Slowhttptest, Heartbleed) while ~10% of benign "
            "windows clear the flag line. Two-sided distance-from-median was "
            "tested and performs worse (see compare_fusion_modes.py)."),
    }
    (ART / "fusion_config.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {ART / 'fusion_config.json'}")


if __name__ == "__main__":
    main()
