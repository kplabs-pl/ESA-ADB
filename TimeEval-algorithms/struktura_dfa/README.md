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
(at least 10) gives a robust baseline (median and MAD); a channel with too few windows for one is
left unflagged, with a warning. A window whose robust z-score
`|alpha - median| / (1.4826 * MAD)` exceeds `threshold` is flagged.

Output follows the ESA-ADB convention used by the `std` algorithm: no header, one 0/1 column
per target channel, channels sorted by name. (The evaluation min-max scales each column and
casts it to `uint8`, so continuous scores would not survive it.)

Two modes:

- `causal = false` (default): the whole flagged window is marked. This localises anomalies in
  hindsight, but it marks up to `window_size` samples before the data that triggered the flag,
  so it is look-ahead and not an online detector.
- `causal = true`: when a window's last sample arrives and the window is flagged, the next
  `stride` samples are marked. Nothing before the decision is marked, and no decision is made
  until the baseline windows have passed.

Self-calibrating: no training phase, no labelled data. Deterministic. The input CSV is streamed,
so memory is about 8 bytes per value.

## Parameters

- `window_size`: `int`, optional (default=1024)
  Samples per DFA window. Values below 72 are raised to 72.

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

**Setup:**
- The official `ESAScores`, `ChannelAwareFScore` and `ADTQC` code, beta 0.5.
- The 84_months test split (7,364,161 rows), with labels of category Anomaly (29 events).

**Control:** the algorithm's own output, shifted in time by one random offset (10-90% of the
length, the same offset for all channels), 20 seeds. The shift keeps every alarm, its length and
its timing across channels, and changes only where the alarms fall relative to the labels. The
table shows the maximum over the 20 shifts.

| | event-wise recall | event-wise F0.5 | affiliation F0.5 | channel F0.5 | ADTQC |
|---|---|---|---|---|---|
| `causal = false` | 0.862 | 0.179 | 0.816 | 0.861 | 0.099 |
| same output shifted (max of 20) | 0.138 | 0.026 | 0.506 | 0.125 | |
| `causal = true` | 0.103 | 0.028 | 0.749 | 0.103 | 0.859 |
| same output shifted (max of 20) | 0.103 | 0.026 | 0.485 | 0.100 | |
| `std`, tol = 3 | 0.310 | 0.0003 | 0.495 | 0.293 | 0.826 |

**Default mode (`causal = false`):**
- It beats every shifted copy of its own output on all four detection scores.
- Its low ADTQC means detections start before the anomalies do. That is the look-ahead described
  above, so this mode suits offline localisation only.

**Causal mode:**
- It detects 3 of the 29 events, which ties the best shifted copy on event-wise recall.
- The alarms it does raise lie closer to the anomalies than any shift: affiliation F0.5 0.749 vs
  at most 0.485.
- So it is a precise but low-recall online detector on this data.

**Tuning did not help.** We chose window size and threshold on the training split (12 settings,
by channel F0.5; the pick was window 128, threshold 2.5). On the test split that setting beat its
own shifted copies on no metric (smallest p 0.238 on the Anomaly labels). The defaults are kept.

**Limits:**
- One channel subset and one split. The full 58-channel set was not run.
- 20 shifts give a smallest possible p of 1/21. Several metrics are reported, with no correction
  for multiple comparisons.

## Reference

C.-K. Peng, S. V. Buldyrev, S. Havlin, M. Simons, H. E. Stanley, and
A. L. Goldberger, "Mosaic organization of DNA nucleotides," Physical Review
E, vol. 49, no. 2, pp. 1685-1689, 1994.
