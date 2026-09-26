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
`|alpha - median| / (1.4826 * MAD)` exceeds `threshold` is flagged. Windows whose DFA fit has
r^2 <= 0.3 (e.g. constant stretches) are skipped: they are never flagged and do not count toward
the baseline.

Output follows the ESA-ADB convention used by the `std` algorithm: no header, one 0/1 column
per target channel, channels sorted by name. (The evaluation min-max scales each column and
casts it to `uint8`, so continuous scores would not survive it.)

Two modes:

- `causal = false` (default): the whole flagged window is marked. This localises anomalies in
  hindsight, but it marks up to `window_size` samples before the data that triggered the flag,
  so it is look-ahead and not an online detector.
- `causal = true`: when a window's last sample arrives and the window is flagged, the next
  `stride` samples are marked. Nothing before the decision is marked, and no decision is made
  until the baseline windows have passed. The baseline length is a fraction of all usable windows
  in the file, so the file length must be known in advance.

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

## Reference

C.-K. Peng, S. V. Buldyrev, S. Havlin, M. Simons, H. E. Stanley, and
A. L. Goldberger, "Mosaic organization of DNA nucleotides," Physical Review
E, vol. 49, no. 2, pp. 1685-1689, 1994.
