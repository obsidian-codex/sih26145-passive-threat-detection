# SIH26145 — AI-Based Detection of Cyber Threats in Unidirectional IP Traffic

Software-emulated data diode (netns + iptables + one-way UDP relay) → forward-only feature extraction →
hybrid ML detection (XGBoost classifier for known attacks + LSTM-Autoencoder for novel threats) → fused
threat score → FastAPI + live dashboard.

## Read these first

| File | Purpose |
|---|---|
| `docs/DECISIONS.md` | Design decisions and implementation reasoning |
| `docs/HANDOFF.md` | Cold-start guide for a new dev/AI taking over |

## Quick map

```
diode/        netns+iptables setup, one-way UDP relay      (needs sudo)
attacks/      scapy attack generators (custom threats)
replay/       slice -> replay -> capture -> label manifest
extraction/   forward-only features: tabular parquet + seq npz
models/       xgboost/rf baseline; lstm autoencoder
fusion/       classifier confidence x recon error -> threat score
serving/      fastapi inference service
dashboard/    live visualization + alert feed
evaluation/   metrics, latency bench
datasets/raw/ downloads (gitignored) — see scripts/download_datasets.sh
data/         intermediate artifacts (gitignored)
```

## Windows demo

Run this from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\gotodashboard.ps1
```

Then open `http://127.0.0.1:8401/?v=3#telemetry`. Go to `ALERT FEED` and click
`TEST ATTACK` to send a synthetic test window through the scoring API.

Generated model artifacts, datasets, PCAPs, virtual environments, and local
model archives are intentionally excluded from Git.

## Important limitation

This is a software-emulated, passive prototype, not physical diode hardware.
The dashboard and anomaly demo are working; full known-attack family scoring
requires a compatible serialized XGBoost artifact.

## Setup

```bash
sudo bash scripts/bootstrap_sudo.sh   # once: apt tooling + scoped sudoers
bash scripts/setup_venv.sh            # python env (.venv/)
bash scripts/download_datasets.sh     # resumable dataset fetch (~50GB)
```
