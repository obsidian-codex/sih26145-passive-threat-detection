#!/usr/bin/env python3
"""Shared helpers for the AE calibration scripts.

Models/calibrate_fusion.py and models/compare_fusion_modes.py both need the
reconstruction errors of the benign windows the autoencoder was validated on.
That split is reproduced here rather than duplicated.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ART = ROOT / "models/artifacts"


def load_benign_val_errors() -> np.ndarray:
    """Reconstruction error of the benign windows the trainer held out.

    Uses the saved artifact plus a re-run of the saved model over the benign
    sequence windows — recomputed (not cached) so the threshold is calibrated on
    the same data the trainer reported on.
    """
    import torch
    import torch.nn as nn
    from models.lstm_ae import LSTMAE

    cfg = json.loads((ART / "lstm_ae_config.json").read_text())
    model = LSTMAE(feat_dim=cfg["feat_dim"])
    model.load_state_dict(torch.load(ART / "lstm_ae.pt", map_location="cpu",
                                     weights_only=True))
    model.eval()

    xs, starts, slices = [], [], []
    for f in sorted((ROOT / "data/features").glob("seqs_*.npz")):
        z = np.load(f, allow_pickle=True)
        if not (np.asarray(z["labels"], dtype=object) == "BENIGN").any():
            continue
        m = np.asarray(z["labels"], dtype=object) == "BENIGN"
        xs.append(z["X"][m])
        starts.append(z["start_t"][m])
        slices.append(np.full(m.sum(), f.stem))
    X = np.concatenate(xs); st = np.concatenate(starts); sl = np.concatenate(slices)

    is_val = np.zeros(len(sl), bool)
    for sid in np.unique(sl):
        m = sl == sid
        t = st[m]
        if t.size < 4:
            continue
        cut = t.min() + (t.max() - t.min()) * 0.25
        idx = np.where(m)[0]
        is_val[idx[t >= cut]] = True

    Xval = torch.tensor((X[is_val] - np.asarray(cfg["mu"])) / np.asarray(cfg["sd"]),
                        dtype=torch.float32)
    lossf = nn.MSELoss(reduction="none")
    out = []
    with torch.no_grad():
        for s in range(0, len(Xval), 4096):
            out.append(lossf(model(Xval[s:s + 4096]), Xval[s:s + 4096])
                       .mean(dim=(1, 2)).numpy())
    return np.concatenate(out)
