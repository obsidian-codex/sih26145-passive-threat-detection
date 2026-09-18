#!/usr/bin/env python3
"""SIH26145 — Alert fusion: combine known-type classifier confidence with
novel-type autoencoder anomaly evidence into ONE threat score.

    score = W_CLF * p_attack_clf + W_AE * ae_flag

where ae_flag is either the classic one-sided sigmoid of (error - threshold) or
a two-sided distance from the benign error mode, chosen by which key the
calibration wrote. The two-sided form matters: rigid floods and portscans
reconstruct BETTER than benign traffic, so a one-sided high-error score can
miss exactly the families whose regularity is the anomaly.

Verdict bands: score >= 0.5 HIGH / >= 0.25 MEDIUM else OK (CRITICAL folds into
HIGH so the demo's clearly-flagged windows still show the family). Weights and
the threshold live in models/artifacts/fusion_config.json (tunable without
retrain — see models/calibrate_fusion.py).
"""
import json
import math
from pathlib import Path

ART = Path("models/artifacts")
# reasonable defaults if nothing has been calibrated yet
DEFAULTS = {"w_clf": 0.65, "w_ae": 0.35, "thr": 0.9, "scale": 0.30,
            "mode": "one-sided"}

_cfg: dict | None = None


def load_config(refresh: bool = False) -> dict:
    """Fusion weights, read from disk ONCE and cached.

    This used to stat+read two JSON files on every fuse() call, i.e. on every
    /score request. Under concurrent disk load (a replay writing captures) those
    reads dominated request time — measured ~600ms per window against ~3ms of
    actual model math. Pass refresh=True after retuning weights.
    """
    global _cfg
    if _cfg is not None and not refresh:
        return _cfg
    cfg = dict(DEFAULTS)
    f = ART / "fusion_config.json"
    if f.exists():
        cfg.update(json.loads(f.read_text()))
    elif (ART / "lstm_ae_config.json").exists():
        ae = json.loads((ART / "lstm_ae_config.json").read_text())
        cfg["thr"] = ae["threshold"]
        cfg["scale"] = max(ae.get("val_error_std", 1.0), 1e-6)
    # normalise a typo/freeform mode name rather than silently mis-scoring
    mode = str(cfg.get("mode", "one-sided")).lower()
    cfg["mode"] = "two-sided" if mode in ("two-sided", "2-sided", "both") else "one-sided"
    _cfg = cfg
    return _cfg


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def ae_flag(err: float, cfg: dict) -> float:
    if cfg["mode"] == "two-sided":
        # distance from the benign error mode, normalised by its IQR
        med, iqr = cfg["benign_err_median"], cfg["benign_err_iqr"]
        dist = abs(float(err) - med) / max(float(iqr), 1e-9)
        return sigmoid(dist - cfg["dist_thr"])
    # classic one-sided: how far the error clears the benign threshold
    return sigmoid((float(err) - cfg["thr"]) / max(float(cfg["scale"]), 1e-9))


def fuse(p_attack_clf: float, ae_err: float | None = None,
         label_pred: str = "", top_features: list[str] | None = None) -> dict:
    cfg = load_config()
    ae_term = 0.0
    if ae_err is not None:
        ae_term = cfg["w_ae"] * ae_flag(ae_err, cfg)
    clf_term = cfg["w_clf"] * float(p_attack_clf)
    score = min(1.0, clf_term + ae_term)
    verdict = ("HIGH" if score >= 0.50 else
               "MEDIUM" if score >= 0.25 else "OK")
    reasons = []
    if clf_term > 0.05 and label_pred and label_pred != "BENIGN":
        reasons.append(f"classifier matched known pattern: {label_pred} "
                       f"(p={p_attack_clf:.2f})")
        if top_features:
            reasons.append("top signals: " + ", ".join(top_features[:4]))
    if ae_term > 0.05:
        reasons.append(f"sequence anomaly: recon error {ae_err:.3f} "
                       f"({cfg['mode']}, flag {ae_term / cfg['w_ae']:.2f})")
    return {"threat_score": round(score, 3), "verdict": verdict,
            "components": {"classifier": round(clf_term, 3),
                           "autoencoder": round(ae_term, 3)},
            "reasons": reasons}


if __name__ == "__main__":
    import sys
    pa = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
    err = float(sys.argv[2]) if len(sys.argv) > 2 else None
    print(json.dumps(fuse(pa, err, "DoS Hulk"), indent=2))

