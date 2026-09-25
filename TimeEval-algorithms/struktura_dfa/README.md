# Struktura DFA

|||
| :--- | :--- |
| Citekey | Peng1994MosaicOrganization |
| Source Code | https://github.com/koscak-labs/struktura |
| Learning type | unsupervised |
| Input dimensionality | multivariate |
|||

## Description

Detrended Fluctuation Analysis (DFA) anomaly detector. For each channel, the DFA scaling
exponent alpha is computed on sliding windows. The first `baseline_fraction` of the windows
gives a robust baseline (median and MAD). A window whose robust z-score
`|alpha - median| / (1.4826 * MAD)` exceeds `threshold` is flagged.

Output follows the ESA-ADB convention used by the `std` algorithm: no header, one 0/1 column
per target channel, channels sorted by name. (The evaluation min-max scales each column and
casts it to `uint8`, so continuous scores would not survive it.)

Two modes:

- `causal = false` (default): the whole flagged window is marked. This localises anomalies in
  hindsight, but it marks up to `window_size` samples before the data that triggered the flag,
  so it is look-ahead and not an online detector.
- `causal = true`: when a window's last sample arrives and the window is flagged, the next
  `stride` samples are marked. Nothing before the decision is marked.

Self-calibrating: no training phase, no labelled data. Deterministic. The input CSV is streamed,
so memory is about 8 bytes per value.

## Parameters

- `window_size`: `int`, optional (default=1024)
  Samples per DFA window. Must be >= 64.

- `stride`: `int`, optional (default=256)
  Step between consecutive windows.

- `baseline_fraction`: `float`, optional (default=0.2)
  Fraction of the windows (from the start) used for the baseline.

- `threshold`: `float`, optional (default=3.5)
  Robust z-score above which a window is flagged.

- `causal`: `bool`, optional (default=false)
  See above.

- `target_channels`: `List[str]`, optional (default=None)
  Channels to score; all data columns if not given.

- `random_state`: `int`, optional (default=42)
  Not used (deterministic). Present for framework compatibility.

## Results on Mission 1, channels 41-46 (defaults, no tuning)

Official `ESAScores`, `ChannelAwareFScore` and `ADTQC` code, beta 0.5, 84_months test split
(7,364,161 rows), labels of category Anomaly (29 events). The control moves this algorithm's own
alarm runs (same number, same lengths, per channel) to random positions, 5 seeds; the max over
seeds is shown.

| | event-wise recall | event-wise F0.5 | affiliation F0.5 | channel F0.5 | ADTQC |
|---|---|---|---|---|---|
| `causal = false` | 0.862 | 0.179 | 0.816 | 0.861 | 0.099 |
| random runs, same shape (max of 5) | 0.241 | 0.015 | 0.495 | 0.140 | 0.400 |
| `causal = true` | 0.172 | 0.033 | 0.820 | 0.172 | 0.715 |
| random runs, same shape (max of 5) | 0.172 | 0.008 | 0.507 | 0.117 | 0.663 |
| `std`, tol = 3 | 0.310 | 0.0003 | 0.495 | 0.293 | 0.826 |

Reading: the default mode places its alarms where the anomalies are far better than chance,
but most of its recall comes from marking data before the change (its ADTQC shows detections
starting early). Without look-ahead the recall is at chance level; the alarms that are raised
still overlap anomalies better than chance (affiliation and event-wise F0.5). Use the default
mode for offline analysis and the causal mode when only past data may be used.

These numbers come from one channel subset and one split; the full 58-channel set was not run.

## Reference

C.-K. Peng, S. V. Buldyrev, S. Havlin, M. Simons, H. E. Stanley, and
A. L. Goldberger, "Mosaic organization of DNA nucleotides," Physical Review
E, vol. 49, no. 2, pp. 1685-1689, 1994.
