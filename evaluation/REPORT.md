# SIH26145 — Evaluation Report

AI-based detection of cyber threats in unidirectional IP traffic.
_generated 2026-09-08 13:41 UTC_

## 1. Diode integrity

```
DIODE VERIFICATION 2026-08-26T10:16:18+00:00
forward frames received on monitor : 5 / 5 (135 bytes)
bytes leaked monitor -> source     : 0 (PASS)
iptables OUTPUT drops in ns-monitor: 2
```

The monitor namespace has an unconditional `iptables OUTPUT DROP`, so no reverse-direction feature can exist. `bwd_*` columns are absent by construction rather than filtered out after the fact.

## 2. Data provenance

Attack windows derived from the CICIDS2017 label CSVs: **14** across 5 day captures.

CSV timestamps required two corrections, both verified against attacker-IP packet bursts located directly in the raw pcaps: the hour field is a 12-hour clock with no AM/PM marker (hours 1-7 are afternoon), and the wallclock is Halifax local time while pcap frames are UTC (+3h). Windows are bounded by the 1st/99th percentile of each label's flow times, and slices are cut at the densest 4-minute sub-window so a slice is attack-dominated rather than mostly benign.

Replayed **24/24** slices through the emulated diode at 1x original timing.

| slice | label | sent | captured | delivery |
|---|---|---:|---:|---:|
| `friday__benign__0` | BENIGN | 144,195 | 144,195 | 100.000% |
| `friday__bot__1` | Bot | 29,889 | 29,861 | 99.910% |
| `friday__benign__2` | BENIGN | 25,350 | 25,350 | 100.000% |
| `friday__portscan__3` | PortScan | 338,923 | 338,923 | 100.000% |
| `friday__ddos__4` | DDoS | 390,516 | 385,463 | 98.710% |
| `monday__benign__5` | BENIGN | 55,733 | 55,724 | 99.980% |
| `monday__benign__6` | BENIGN | 95,984 | 95,982 | 100.000% |
| `thursday__web-attack-brute-force__7` | Web Attack  Brute Force | 63,646 | 63,645 | 100.000% |
| `thursday__web-attack-xss__8` | Web Attack  XSS | 48,619 | 48,619 | 100.000% |
| `thursday__web-attack-sql-injection__9` | Web Attack  Sql Injection | 64,093 | 64,091 | 100.000% |
| `thursday__benign__10` | BENIGN | 47,590 | 47,590 | 100.000% |
| `thursday__benign__11` | BENIGN | 77,289 | 77,281 | 99.990% |
| `thursday__infiltration__12` | Infiltration | 172,056 | 172,054 | 100.000% |
| `tuesday__benign__13` | BENIGN | 51,582 | 51,575 | 99.990% |
| `tuesday__ftp-patator__14` | FTP-Patator | 96,413 | 96,413 | 100.000% |
| `tuesday__benign__15` | BENIGN | 67,365 | 67,359 | 99.990% |
| `tuesday__ssh-patator__16` | SSH-Patator | 84,967 | 84,964 | 100.000% |
| `wednesday__benign__17` | BENIGN | 1,208,569 | 1,208,193 | 99.970% |
| `wednesday__dos-slowloris__18` | DoS slowloris | 86,352 | 86,345 | 99.990% |
| `wednesday__dos-slowhttptest__19` | DoS Slowhttptest | 68,719 | 68,719 | 100.000% |
| `wednesday__dos-hulk__20` | DoS Hulk | 589,578 | 587,550 | 99.660% |
| `wednesday__dos-goldeneye__21` | DoS GoldenEye | 189,303 | 188,716 | 99.690% |
| `wednesday__benign__22` | BENIGN | 59,296 | 59,296 | 100.000% |
| `wednesday__heartbleed__23` | Heartbleed | 116,576 | 116,427 | 99.870% |

Mean delivery 99.906%; worst slice `friday__ddos__4` at 98.710%.

Forward-only extraction over 24 slices: **829,418** flow-window rows, **86,042** source-bucket rows, **157,182** sequence windows, 0 unparseable frames.

## 3. Known-attack classifier (XGBoost)

- 900,000 rows — train 617,143 / test 282,857
- split: **temporal-holdout-per-slice** (tail 0.3 of each slice's own timeline, so every class is present on both sides and the model is always scored on later traffic)
- clock features excluded; raw IP octets excluded (CICIDS2017 sources almost every attack from one host, so address features would make this an IP blocklist)
- **macro-F1 0.4602**, weighted-F1 0.6961, accuracy 0.6742 over 15 test classes

| class | precision | recall | F1 | support | PR-AUC |
|---|---:|---:|---:|---:|---:|
| PortScan | 0.974 | 0.928 | 0.95 | 98980 | 0.9816 |
| BENIGN | 0.69 | 0.383 | 0.493 | 39477 | 0.6523 |
| DoS Hulk | 0.955 | 0.843 | 0.896 | 32300 | 0.9264 |
| DDoS | 0.933 | 0.815 | 0.87 | 26013 | 0.9083 |
| DoS GoldenEye | 0.688 | 0.419 | 0.521 | 14971 | 0.6127 |
| DoS Slowhttptest | 0.355 | 0.368 | 0.361 | 11287 | 0.2725 |
| SSH-Patator | 0.337 | 0.515 | 0.407 | 9648 | 0.5351 |
| Infiltration | 0.328 | 0.464 | 0.384 | 8596 | 0.4442 |
| Web Attack  Brute Force | 0.198 | 0.282 | 0.232 | 7320 | 0.1793 |
| DoS slowloris | 0.409 | 0.533 | 0.463 | 6963 | 0.4977 |
| Heartbleed | 0.448 | 0.613 | 0.517 | 6843 | 0.5525 |
| FTP-Patator | 0.184 | 0.372 | 0.246 | 6418 | 0.2373 |
| Web Attack  Sql Injection | 0.131 | 0.198 | 0.158 | 6227 | 0.1658 |
| Web Attack  XSS | 0.234 | 0.312 | 0.268 | 4559 | 0.2322 |
| Bot | 0.089 | 0.284 | 0.136 | 3255 | 0.1733 |

**Binary attack/benign operating point** — p_attack = 1 - P(BENIGN), identical to serving/app.py:

- ROC-AUC **0.9122**, PR-AUC **0.9853**
- at threshold 0.839801 → recall 0.950, precision 0.917, **FPR 0.5302**
- TP 231,211 · FP 20,929 · TN 18,548 · FN 12,169

Top attributions: `dst_port`, `src_port`, `flag_syn_frac`, `ttl_max`, `flag_rst_frac`, `ttl_mean`, `pkt_size_min`, `ttl_min`

## 4. Novel-threat detector (LSTM-Autoencoder, benign-only training)

- trained on 52,434 benign windows, calibrated on 17,105 held-out benign windows
- split: tail 25% of each benign slice (no window overlap)
- threshold (95th pct of benign error): `0.90132` → benign flag-rate 0.05
- separation ROC-AUC **0.7436** (two-sided 0.8011), attack detection at threshold 0.0843

| attack family | windows | mean error | flagged at threshold |
|---|---:|---:|---:|
| Bot | 988 | 0.7028 | 32.69% |
| DoS Slowhttptest | 2,396 | 0.65998 | 26.54% |
| Web Attack  Brute Force | 2,085 | 0.63041 | 23.26% |
| Web Attack  Sql Injection | 2,142 | 0.58985 | 21.85% |
| Heartbleed | 4,329 | 0.64293 | 21.28% |
| Web Attack  XSS | 1,630 | 0.59143 | 20.74% |
| FTP-Patator | 3,259 | 0.59409 | 20.37% |
| SSH-Patator | 2,959 | 0.62798 | 19.26% |
| DoS slowloris | 3,064 | 0.52726 | 18.37% |
| DoS GoldenEye | 7,113 | 0.37013 | 7.23% |
| Infiltration | 6,355 | 0.39332 | 6.80% |
| PortScan | 13,182 | 0.15378 | 3.77% |
| DDoS | 15,006 | 0.41478 | 3.61% |
| DoS Hulk | 23,135 | 0.49416 | 1.89% |

## 4b. Alert fusion (classifier + autoencoder)

- `score = 0.65·p_classifier + 0.35·AE_flag`
- AE flag: `one-sided` sigmoid of reconstruction error (threshold `0.6776`, scale `0.2243`)
- ±: benign flag-rate **0.1**, attack flag at mean error **0.26**
| family | AE flag |
|---|---:|
| Bot | 0.528 |
| DoS Slowhttptest | 0.48 |
| Heartbleed | 0.461 |
| Web Attack  Brute Force | 0.448 |
| SSH-Patator | 0.445 |
| FTP-Patator | 0.408 |
| Web Attack  XSS | 0.405 |
| Web Attack  Sql Injection | 0.403 |
| DoS slowloris | 0.338 |
| DoS Hulk | 0.306 |
| DDoS | 0.237 |
| Infiltration | 0.22 |
| DoS GoldenEye | 0.203 |
| PortScan | 0.088 |

> The AE is the novelty channel; the classifier carries the known families. Weights can be retuned by editing `fusion_config.json` (see `models/calibrate_fusion.py`).

## 5. Inference latency

Measured over 500 windows on an idle host (cuda), full fuse path (classifier + autoencoder + fusion):

- mean **4.532 ms**, p50 4.5 ms, p95 **5.967 ms**, p99 8.147 ms, max 10.318 ms
- gate (<10 ms/window): **PASS**

| stage | mean ms |
|---|---:|
| classifier | 3.948 |
| autoencoder | 0.569 |
| fusion | 0.015 |

## Artifacts

| file | purpose |
|---|---|
| `models/artifacts/classical_confusion.png` | row-normalised confusion matrix |
| `models/artifacts/classical_shap.png` | feature attribution |
| `models/artifacts/ae_errors.png` | benign vs attack reconstruction error |
| `docs/ATTACK_WINDOWS.md` | derived attack windows, pcap time base |
| `data/captures/manifest.jsonl` | per-slice label + delivery ground truth |

