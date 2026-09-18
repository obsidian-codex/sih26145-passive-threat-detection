#!/usr/bin/env python3
"""SIH26145 — Classical detector: XGBoost on forward-only tabular features.

Consumes BOTH extractor views (flow-window + source-time-bucket) as one union
schema. XGBoost's sparsity-aware splits handle the missing half natively, so a
flow row simply leaves the source-view columns NaN and vice versa.

THREE methodology rules are enforced here, each fixing a defect that made the
previous run's numbers meaningless:

1. TEMPORAL HOLDOUT, NOT SLICE HOLDOUT.
   There is exactly one replayed slice per attack class, so a group split by
   slice_id sent every attack class wholly into train and left a BENIGN-only
   test set (macro-F1 0.25 was measuring nothing). Instead each slice is split
   along its own time axis: the first (1-TEST_FRAC) of a slice's duration
   trains, the tail tests. Every class therefore appears on both sides, no flow
   spans the boundary, and the model is always scored on later traffic than it
   trained on.

2. NO CLOCK FEATURES.
   `bucket` is an absolute 5-second index, so each slice occupies a unique
   contiguous range of it. It was previously left in the feature matrix, which
   let the trees identify a slice (hence its label) from the clock alone.

3. NO ADDRESS FEATURES BY DEFAULT.
   CICIDS2017 runs nearly every attack from 172.16.0.1, so raw source/dest
   octets turn the "behavioural detector" into an IP blocklist that collapses
   outside this dataset. Pass --use-ip-features to include them for comparison.

Outputs (models/artifacts/): xgb_model.joblib, feature_columns.json,
label_classes.json, classical_metrics.json, classical_confusion.png,
classical_shap.png

Usage: python models/train_xgb.py [--features-dir data/features] [--max-rows 900000]
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

TEST_FRAC = 0.30          # tail fraction of each slice's timeline held out
BUCKET_S = 5.0            # must match extraction/extract_features.py bucket_s

# Never model inputs: labels, provenance, and anything carrying a wall clock.
META = {"label", "is_attack", "slice_id", "view", "bucket", "t_first", "t_last",
        "t_key", "src", "dst"}
IP_FEATURES = {"src_o12", "src_o4", "dst_o12", "dst_o4"}


def load_union(features_dir: Path, use_ip: bool, benign_cap_per_file: int = 60_000):
    """Concatenate both views under a unioned column set, with a time key.

    Benign rows are subsampled PER FILE as they are read rather than after a
    full concat: the host has ~6 GB of RAM and the captures total >4M feature
    rows, so materialising everything first and trimming afterwards is the one
    step most likely to OOM. Attack rows are always kept in full.
    """
    import pandas as pd

    flow_files = sorted(features_dir.glob("flows_*.parquet"))
    src_files = sorted(features_dir.glob("srcwin_*.parquet"))
    if not flow_files and not src_files:
        raise SystemExit(f"no parquet features under {features_dir} — run extractor first")

    rng = np.random.default_rng(42)
    parts, dropped = [], 0
    for files, view in ((flow_files, "flow"), (src_files, "src")):
        for p in files:
            df = pd.read_parquet(p)
            if not len(df):
                continue
            if view == "flow":
                if "t_first" not in df.columns:
                    raise SystemExit(
                        "flow features lack t_first — re-run extraction/batch_extract.py "
                        "with the current extractor so the temporal split is possible")
                df["t_key"] = df["t_first"].astype("float64")
            else:
                df["t_key"] = df["bucket"].astype("float64") * BUCKET_S
            df["view"] = view
            if int(df["is_attack"].iloc[0]) == 0 and len(df) > benign_cap_per_file:
                # keep a time-stratified sample so the temporal split still works
                keep = np.sort(rng.choice(len(df), benign_cap_per_file, replace=False))
                dropped += len(df) - benign_cap_per_file
                df = df.iloc[keep]
            parts.append(df)

    big = pd.concat(parts, ignore_index=True)
    del parts
    if dropped:
        print(f"   subsampled {dropped:,} benign rows at load time (RAM budget)")

    # address octets are derived but kept out of the model unless asked for
    for col in ("src", "dst"):
        if col in big:
            octs = big[col].astype(str).str.split(".", expand=True)
            big[f"{col}_o12"] = (pd.to_numeric(octs[0], errors="coerce").fillna(-1) * 256
                                 + pd.to_numeric(octs[1], errors="coerce").fillna(-1))
            big[f"{col}_o4"] = pd.to_numeric(
                octs[3] if octs.shape[1] > 3 else None, errors="coerce").fillna(-1)

    drop = set(META) | (set() if use_ip else IP_FEATURES)
    feature_cols = sorted(c for c in big.columns if c not in drop)
    return big, feature_cols


def temporal_split(big, test_frac: float = TEST_FRAC):
    """Boolean test mask: the last `test_frac` of EACH slice's own timeline."""
    is_test = np.zeros(len(big), dtype=bool)
    for sid, idx in big.groupby("slice_id").groups.items():
        t = big.loc[idx, "t_key"]
        lo, hi = t.min(), t.max()
        if not np.isfinite(lo) or hi <= lo:
            continue
        cut = lo + (hi - lo) * (1.0 - test_frac)
        is_test[big.index.get_indexer(idx[t >= cut])] = True
    return is_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="data/features")
    ap.add_argument("--artifacts-dir", default="models/artifacts")
    ap.add_argument("--max-rows", type=int, default=900_000)
    ap.add_argument("--test-frac", type=float, default=TEST_FRAC)
    ap.add_argument("--use-ip-features", action="store_true",
                    help="include raw src/dst octets (dataset-specific; off by default)")
    ap.add_argument("--benign-cap-per-file", type=int, default=60_000,
                    help="benign rows kept from each parquet at load time (RAM budget)")
    args = ap.parse_args()

    from sklearn.metrics import (average_precision_score, classification_report,
                                 confusion_matrix, precision_recall_curve,
                                 roc_auc_score)
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # class names carry an en-dash (e.g. "Web Attack \u2013 Brute Force") that
    # DejaVu Sans cannot render; the glyph was dropped and matplotlib emitted a
    # warning on every save. En-dashes are replaced with hyphens in plot labels.
    import pandas as pd
    import xgboost as xgb

    big, feature_cols = load_union(Path(args.features_dir), args.use_ip_features,
                                   args.benign_cap_per_file)
    print(f"loaded {len(big):,} rows / {big.slice_id.nunique()} slices / "
          f"{big.label.nunique()} classes / {len(feature_cols)} features")

    # RAM discipline: cap benign, keep every attack row
    benign, attack = big[big.is_attack == 0], big[big.is_attack == 1]
    cap = max(20_000, args.max_rows - len(attack))
    if len(benign) > cap:
        benign = benign.sample(cap, random_state=42)
    big = pd.concat([benign, attack], ignore_index=True)

    le = LabelEncoder()
    y = le.fit_transform(big["label"])
    te_mask = temporal_split(big, args.test_frac)
    tr_mask = ~te_mask

    X = big[feature_cols].astype("float32")
    Xtr, Xte, ytr, yte = X[tr_mask], X[te_mask], y[tr_mask], y[te_mask]

    tr_classes = set(np.unique(ytr))
    te_classes = set(np.unique(yte))
    print(f"train {len(Xtr):,} rows / {len(tr_classes)} classes    "
          f"test {len(Xte):,} rows / {len(te_classes)} classes")
    missing = [le.classes_[i] for i in sorted(set(range(len(le.classes_))) - te_classes)]
    if missing:
        print(f"   note: no test rows for {missing} (too few windows to hold out)")

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.12, subsample=0.85,
        colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.5,
        tree_method="hist", eval_metric="mlogloss", n_jobs=8, random_state=42,
        num_class=len(le.classes_))
    model.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr),
              verbose=False)

    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)

    art = Path(args.artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)

    present = sorted(te_classes | set(np.unique(pred)))
    report = classification_report(
        yte, pred, labels=present, target_names=[str(le.classes_[i]) for i in present],
        output_dict=True, zero_division=0)

    prauc = {}
    for i, cls in enumerate(le.classes_):
        bin_y = (yte == i).astype(int)
        if 0 < bin_y.sum() < len(bin_y):
            prauc[str(cls)] = round(float(average_precision_score(bin_y, proba[:, i])), 4)

    # ── binary operating point ────────────────────────────────────────────────
    # p_attack MUST match serving/app.py: 1 - P(benign). The previous version
    # divided a class-probability sum by n_classes, so the threshold chosen here
    # did not correspond to anything the live service computed.
    bi = int(list(le.classes_).index("BENIGN"))
    p_attack = 1.0 - proba[:, bi]
    y_bin = (yte != bi).astype(int)
    binary = {}
    if 0 < y_bin.sum() < len(y_bin):
        prec, rec, thr = precision_recall_curve(y_bin, p_attack)
        ok = np.where(rec[:-1] >= 0.95)[0]
        chosen = float(thr[ok[-1]]) if len(ok) else 0.5
        flagged = p_attack >= chosen
        tp = int((flagged & (y_bin == 1)).sum()); fn = int((~flagged & (y_bin == 1)).sum())
        fp = int((flagged & (y_bin == 0)).sum()); tn = int((~flagged & (y_bin == 0)).sum())
        binary = {
            "definition": "p_attack = 1 - P(BENIGN), identical to serving/app.py",
            "threshold_at_recall95": round(chosen, 6),
            "recall": round(tp / max(tp + fn, 1), 4),
            "precision": round(tp / max(tp + fp, 1), 4),
            "fpr": round(fp / max(fp + tn, 1), 5),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "roc_auc": round(float(roc_auc_score(y_bin, p_attack)), 4),
            "pr_auc": round(float(average_precision_score(y_bin, p_attack)), 4),
        }

    # ── confusion matrix, row-normalised (class counts differ by 1000x) ───────
    cm = confusion_matrix(yte, pred, labels=present)
    names = [str(le.classes_[i]).replace("\x96", "-").replace("\u2013", "-")
             .replace("\u2014", "-") for i in present]
    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(1.05 * len(names) + 3, 0.85 * len(names) + 2.5))
    im = ax.imshow(cmn, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"row-normalised confusion — temporal holdout ({len(Xte):,} rows)")
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            if cm[r, c]:
                ax.text(c, r, f"{cmn[r, c]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cmn[r, c] < 0.6 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(art / "classical_confusion.png", dpi=140)
    plt.close(fig)

    # ── attribution ──────────────────────────────────────────────────────────
    # SHAP for multiclass returns (n, features, classes) on modern versions and a
    # list of per-class arrays on older ones; the previous code assumed one shape
    # and silently degraded to an error string in the metrics file.
    shap_top: list[str] = []
    try:
        import shap
        samp = Xte.sample(min(1500, len(Xte)), random_state=1)
        sv = shap.TreeExplainer(model).shap_values(samp)
        arr = np.asarray(np.stack(sv, axis=-1) if isinstance(sv, list) else sv)
        # (rows, features) for binary, (rows, features, classes) for multiclass
        if arr.ndim == 3:
            imp = np.abs(arr).mean(axis=(0, 2))
        else:
            imp = np.abs(arr).mean(axis=0)
        imp = np.asarray(imp).ravel()
        if imp.size != len(samp.columns):
            raise ValueError(f"shap importance shape {imp.shape} vs "
                             f"{len(samp.columns)} features")
        order = np.argsort(imp)[::-1][:15]
        shap_top = [str(samp.columns[i]) for i in order[:8]]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(order)), imp[order][::-1], color="#e61919")
        ax.set_yticks(range(len(order)), [samp.columns[i] for i in order][::-1], fontsize=8)
        ax.set_xlabel("mean |SHAP| over classes")
        ax.set_title("forward-only feature attribution (top 15)")
        fig.tight_layout(); fig.savefig(art / "classical_shap.png", dpi=140)
        plt.close(fig)
    except Exception as e:                                   # noqa: BLE001
        print(f"   shap unavailable ({e}); falling back to gain importance")
    if not shap_top:
        gain = model.feature_importances_
        shap_top = [str(feature_cols[i]) for i in np.argsort(gain)[::-1][:8]]

    metrics = {
        "split": {"kind": "temporal-holdout-per-slice", "test_frac": args.test_frac,
                  "uses_ip_features": bool(args.use_ip_features)},
        "n_rows": int(len(X)), "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
        "n_classes": int(len(le.classes_)),
        "n_test_classes": int(len(te_classes)),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "accuracy": round(float(report["accuracy"]), 4),
        "binary_attack_detection": binary,
        "pr_auc_per_class": prauc,
        "shap_top_features": shap_top,
        "per_class": {k: {"precision": round(v["precision"], 3),
                          "recall": round(v["recall"], 3),
                          "f1": round(v["f1-score"], 3),
                          "support": int(v["support"])}
                      for k, v in report.items()
                      if k not in ("accuracy", "macro avg", "weighted avg")},
    }
    (art / "classical_metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(model, art / "xgb_model.joblib")
    (art / "feature_columns.json").write_text(json.dumps(feature_cols))
    (art / "label_classes.json").write_text(json.dumps([str(c) for c in le.classes_]))

    print(f"\nmacro-F1 {metrics['macro_f1']}   accuracy {metrics['accuracy']}")
    if binary:
        print(f"binary attack detection: ROC-AUC {binary['roc_auc']}  "
              f"PR-AUC {binary['pr_auc']}  "
              f"FPR@95%recall {binary['fpr']}")
    print("top features:", ", ".join(shap_top[:6]))
    print(f"artifacts -> {art}/")


if __name__ == "__main__":
    main()
