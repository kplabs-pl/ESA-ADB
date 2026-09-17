# Struktura DFA

|||
| :--- | :--- |
| Citekey | Peng1994MosaicOrganization |
| Source Code | https://github.com/koscak-labs/struktura |
| Learning type | unsupervised |
| Input dimensionality | multivariate |
|||

## Description

Detrended Fluctuation Analysis (DFA) anomaly detector. Computes the DFA
scaling exponent alpha on sliding windows and flags timestamps where alpha
deviates from a robust baseline (median + MAD, established from the first
`baseline_fraction` of the series).

For multivariate input, each channel is scored independently and the
per-timestamp anomaly score is the maximum across channels.

Self-calibrating: no training phase, no labelled data required. Deterministic:
the same input always produces the same output.

**First Rust-based algorithm in ESA-ADB.** Self-contained implementation using
only `libm`, `serde`, and `csv` crates. No external ML dependencies.

## Parameters

- `window_size`: `int`, optional (default=1024)
  Number of samples per DFA window. Must be >= 64. Larger windows capture
  longer-range correlations but reduce temporal resolution.

- `stride`: `int`, optional (default=256)
  Step size between consecutive windows. Smaller stride = finer resolution
  but more computation.

- `baseline_fraction`: `float`, optional (default=0.2)
  Fraction of the series (from the start) used to establish the baseline.
  Uses median + MAD for robustness against early anomalies.

- `random_state`: `int`, optional (default=42)
  Not used (algorithm is fully deterministic). Present for TimeEval
  framework compatibility.

## Reference

C.-K. Peng, S. V. Buldyrev, S. Havlin, M. Simons, H. E. Stanley, and
A. L. Goldberger, "Mosaic organization of DNA nucleotides," Physical Review
E, vol. 49, no. 2, pp. 1685-1689, 1994.
