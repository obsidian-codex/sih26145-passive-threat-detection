#!/usr/bin/env python3
"""SIH26145 — Novel-threat detector: LSTM Autoencoder on benign-only sequences.

Trained exclusively on sequence windows from BENIGN slices. At inference,
reconstruction error above a benign-derived threshold flags traffic whose
temporal texture was never seen in normal one-way streams — the novel/zero-day
half of the detector, complementing the classifier's known families.

Two methodology fixes over the previous revision:

1. NO OVERLAPPING-WINDOW LEAKAGE.
   Sequence windows are cut with stride 25 over length 50, so consecutive
   windows share half their packets. The benign train/validation split used to
   be a random permutation over those windows, which put near-duplicates on both
   sides and produced an optimistically small validation error — and therefore a
   threshold that was too tight. The split is now the TAIL of each benign
   slice's own timeline, so no validation window shares packets with training.

2. THRESHOLD FROM BENIGN ONLY, EVALUATED HONESTLY.
   Attack data is never used to pick the operating point. It is only used
   afterwards to report separation. If attack windows reconstruct BETTER than
   benign ones (AUC < 0.5) that is reported loudly rather than hidden: highly
   uniform flood traffic is genuinely easy to reconstruct, and pretending
   otherwise would misrepresent what this model can do.

Outputs (models/artifacts/):
  lstm_ae.pt, lstm_ae_config.json, ae_metrics.json, ae_errors.png

Usage: python models/lstm_ae.py [--features-dir data/features]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

VAL_TAIL_FRAC = 0.25          # tail of each benign slice held out for thresholding
QUANTILES = [0.90, 0.95, 0.97, 0.99, 0.995, 0.999]


class LSTMAE(nn.Module):
    def __init__(self, feat_dim: int = 4, hidden: int = 64, latent: int = 16):
        super().__init__()
        self.enc = nn.LSTM(feat_dim, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.dec = nn.LSTM(latent, hidden, batch_first=True)
        self.out = nn.Linear(hidden, feat_dim)

    def forward(self, x):
        _, (h, _) = self.enc(x)                       # h: (1, B, H)
        z = self.to_latent(h[-1])                     # (B, latent) bottleneck
        z_t = z.unsqueeze(1).repeat(1, x.size(1), 1)  # (B, T, latent)
        y, _ = self.dec(z_t, (h, torch.zeros_like(h)))
        return self.out(y)


def load_seqs(features_dir: Path):
    """All sequence windows with label / slice / start-time metadata."""
    Xs, labels, slices, starts = [], [], [], []
    for f in sorted(features_dir.glob("seqs_*.npz")):
        z = np.load(f, allow_pickle=True)
        n = len(z["X"])
        if not n:
            continue
        Xs.append(z["X"])
        labels.append(np.asarray(z["labels"], dtype=object))
        slices.append(z["slice_id"] if "slice_id" in z
                      else np.full(n, f.stem.replace("seqs_", "")))
        starts.append(z["start_t"] if "start_t" in z else np.arange(n, dtype=float))
    if not Xs:
        raise SystemExit(f"no seqs_*.npz under {features_dir} — run the extractor first")
    return (np.concatenate(Xs), np.concatenate(labels),
            np.concatenate(slices).astype(str), np.concatenate(starts))


def tail_split(slices: np.ndarray, starts: np.ndarray, frac: float):
    """Mask selecting the last `frac` of each slice's timeline."""
    is_val = np.zeros(len(slices), dtype=bool)
    for sid in np.unique(slices):
        m = slices == sid
        t = starts[m]
        if t.size < 4 or t.max() <= t.min():
            continue
        cut = t.min() + (t.max() - t.min()) * (1.0 - frac)
        idx = np.where(m)[0]
        is_val[idx[t >= cut]] = True
    return is_val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="data/features")
    ap.add_argument("--artifacts-dir", default="models/artifacts")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-train", type=int, default=120_000,
                    help="cap benign training windows (VRAM discipline)")
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    art = Path(args.artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)

    X, labels, slices, starts = load_seqs(Path(args.features_dir))
    benign = labels == "BENIGN"
    print(f"{len(X):,} windows / {benign.sum():,} benign / "
          f"{(~benign).sum():,} attack / dim {X.shape[1]}x{X.shape[2]}")
    if benign.sum() < 500:
        raise SystemExit("not enough benign sequence windows to train")

    # ── benign train / val split along each benign slice's own timeline ────────
    val_of_benign = tail_split(slices[benign], starts[benign], VAL_TAIL_FRAC)
    Xb = X[benign]
    Xb_tr, Xb_val = Xb[~val_of_benign], Xb[val_of_benign]
    if len(Xb_tr) > args.max_train:
        keep = np.random.default_rng(42).choice(len(Xb_tr), args.max_train, replace=False)
        Xb_tr = Xb_tr[keep]
    print(f"benign train {len(Xb_tr):,}  benign val(tail) {len(Xb_val):,}")

    mu = Xb_tr.reshape(-1, X.shape[-1]).mean(0)
    sd = Xb_tr.reshape(-1, X.shape[-1]).std(0) + 1e-6
    norm = lambda a: (a - mu) / sd                                    # noqa: E731

    Xtr = torch.tensor(norm(Xb_tr), dtype=torch.float32, device=dev)
    Xval = torch.tensor(norm(Xb_val), dtype=torch.float32, device=dev)

    model = LSTMAE(feat_dim=X.shape[-1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.MSELoss(reduction="none")

    best_val, best_state, patience, bad = float("inf"), None, 10, 0
    for ep in range(args.epochs):
        model.train()
        p = torch.randperm(Xtr.size(0), device=dev)
        tot = 0.0
        for s in range(0, len(p), args.batch):
            xb = Xtr[p[s:s + args.batch]]
            opt.zero_grad()
            loss = lossf(model(xb), xb).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss) * len(xb)
        model.eval()
        with torch.no_grad():
            vloss = float(lossf(model(Xval), Xval).mean())
        sched.step()
        if vloss < best_val - 1e-6:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 10 == 0:
            print(f"ep {ep:03d}  train {tot/max(len(Xtr),1):.5f}  val {vloss:.5f}")
        if bad >= patience:
            print(f"early stop @ ep{ep} (best val {best_val:.5f})")
            break
    if best_state:
        model.load_state_dict(best_state)      # keep the best epoch, not the last
    model.eval()

    def errors(a: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for s in range(0, len(a), 4096):
                xa = torch.tensor(norm(a[s:s + 4096]), dtype=torch.float32, device=dev)
                out.append(lossf(model(xa), xa).mean(dim=(1, 2)).cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    val_err = errors(Xb_val)
    thr_grid = np.quantile(val_err, QUANTILES)          # benign-only calibration
    thr = float(thr_grid[1])                            # 95th percentile default

    metrics = {
        "n_benign_train": int(len(Xb_tr)), "n_benign_val": int(len(Xb_val)),
        "split": f"tail {VAL_TAIL_FRAC:.0%} of each benign slice (no window overlap)",
        "val_error_mean": float(val_err.mean()), "val_error_std": float(val_err.std()),
        "threshold_pct95": thr,
        "benign_flag_rate_at_thr": round(float((val_err > thr).mean()), 4),
    }

    # ── separation, reported after the threshold is already fixed ─────────────
    per_family, fam_flag = {}, {}
    if (~benign).any():
        Xa, la = X[~benign], labels[~benign]
        ea = errors(Xa)
        y = np.r_[np.zeros(len(val_err)), np.ones(len(ea))]
        s = np.r_[val_err, ea]
        auc = float(roc_auc_score(y, s))
        # two-sided alternative: distance from the benign error mode, which also
        # catches traffic that is ABNORMALLY EASY to reconstruct (rigid floods)
        med = float(np.median(val_err))
        iqr = float(np.subtract(*np.percentile(val_err, [75, 25]))) or 1e-6
        auc2 = float(roc_auc_score(y, np.abs(s - med) / iqr))
        metrics["auc"] = round(auc, 4)
        metrics["auc_two_sided"] = round(auc2, 4)
        metrics["attack_err_mean"] = float(ea.mean())
        metrics["detection_rate_at_thr"] = round(float((ea > thr).mean()), 4)
        for fam in sorted(set(la.tolist())):
            m = la == fam
            per_family[str(fam)] = {
                "windows": int(m.sum()),
                "err_mean": round(float(ea[m].mean()), 5),
                "flagged_at_thr": round(float((ea[m] > thr).mean()), 4),
            }
            fam_flag[str(fam)] = per_family[str(fam)]["flagged_at_thr"]
        metrics["per_family"] = per_family
        if auc < 0.5:
            metrics["warning"] = (
                "attack windows reconstruct BETTER than benign (AUC<0.5): uniform "
                "flood/scan timing is easier to model than bursty benign traffic. "
                "Use auc_two_sided / the classifier for these families.")
            print("  !! " + metrics["warning"])

        fig, ax = plt.subplots(figsize=(9, 5))
        lo, hi = float(min(s.min(), 0)), float(np.percentile(s, 99.5))
        bins = np.linspace(lo, hi, 90)
        ax.hist(val_err, bins=bins, alpha=0.6, density=True, label="benign (held-out tail)")
        ax.hist(ea, bins=bins, alpha=0.6, density=True, label="attack")
        ax.axvline(thr, color="#e61919", lw=1.6, label=f"threshold p95 = {thr:.4f}")
        ax.set_xlabel("reconstruction error (MSE)")
        ax.set_ylabel("density")
        ax.set_title(f"LSTM-AE benign-vs-attack separation — ROC-AUC {auc:.3f} "
                     f"(two-sided {auc2:.3f})")
        ax.legend()
        fig.tight_layout(); fig.savefig(art / "ae_errors.png", dpi=140)
        plt.close(fig)

    torch.save(model.state_dict(), art / "lstm_ae.pt")
    (art / "lstm_ae_config.json").write_text(json.dumps({
        "feat_dim": int(X.shape[-1]), "seq_len": int(X.shape[1]),
        "hidden": 64, "latent": 16,
        "mu": mu.tolist(), "sd": sd.tolist(),
        "threshold": thr,
        "threshold_candidates": {str(q): float(t) for q, t in zip(QUANTILES, thr_grid)},
        "val_error_mean": metrics["val_error_mean"],
        "val_error_std": metrics["val_error_std"],
    }, indent=2))
    (art / "ae_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nthreshold(p95 benign) {thr:.5f}   "
          f"benign flag-rate {metrics['benign_flag_rate_at_thr']:.3f}")
    if "auc" in metrics:
        print(f"AUC {metrics['auc']}  two-sided {metrics['auc_two_sided']}  "
              f"attack detection @thr {metrics['detection_rate_at_thr']}")
        for fam, v in sorted(fam_flag.items(), key=lambda kv: -kv[1])[:8]:
            print(f"   {fam:<28} flagged {v:.2%}")
    print(f"artifacts -> {art}/ (device={dev})")


if __name__ == "__main__":
    main()
