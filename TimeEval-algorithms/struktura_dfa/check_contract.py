"""Check that struktura_dfa's output survives the ESA-ADB evaluation path unchanged.

Replays what the pipeline does with an algorithm's output: timeeval/adapters/docker.py reads it with
np.genfromtxt, timeeval/core/experiments.py expands a 1-D result to one column, applies scale_scores
(per-column MinMax on float32, NaN/Inf masked), takes the max over channels, casts it to uint8 and
assigns it to the test timestamps. Also checks the parameters and edge cases documented in README.md.

Run: python check_contract.py path/to/struktura-timeeval
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scale_scores(y_scores):
    """timeeval/core/experiments.py Experiment.scale_scores, same logic."""
    y_scores = np.asarray(y_scores, dtype=np.float32)
    if y_scores.ndim == 1:
        y_scores = np.expand_dims(y_scores, -1)
    for i in range(y_scores.shape[-1]):
        mask = np.isinf(y_scores[..., i]) | np.isneginf(y_scores[..., i]) | np.isnan(y_scores[..., i])
        scores = y_scores[..., i][~mask]
        if scores.size != 0:
            if len(scores.shape) == 1:
                scores = scores.reshape(-1, 1)
            y_scores[..., i][~mask] = MinMaxScaler().fit_transform(scores).ravel()
    return y_scores


def evaluate_path(out, timestamps):
    """The pipeline's handling of the output file, up to the assignment that failed before."""
    y = np.genfromtxt(out, delimiter=",")
    if y.ndim == 1:
        y = np.expand_dims(y, -1)
    scaled = scale_scores(y)
    ts = pd.DataFrame({"timestamp": timestamps})
    ts["Score"] = scaled.max(axis=1).astype(np.uint8)  # raised a length mismatch on the merged version
    assert np.array_equal(scaled, y), "scale_scores changed the 0/1 output"
    return scaled


def run(exe, inp, out, params):
    cfg = {"dataInput": str(inp), "dataOutput": str(out), "executionType": "execute", "customParameters": params}
    return subprocess.run([exe, json.dumps(cfg)], capture_output=True, text=True)


def main():
    exe = sys.argv[1]
    rng = np.random.default_rng(0)
    n = 20000
    t = pd.date_range("2000-01-01", periods=n, freq="30s")
    walk_after = rng.standard_normal(n)
    walk_after[12000:] = np.cumsum(rng.standard_normal(n - 12000)) * 0.05  # structure change, similar spread
    df = pd.DataFrame({"timestamp": t, "channel_2": rng.standard_normal(n), "channel_10": walk_after,
                       "channel_1": rng.standard_normal(n)})
    for c in ["channel_2", "channel_10", "channel_1"]:
        df[f"is_anomaly_{c}"] = 0
    df.loc[12000:, "is_anomaly_channel_10"] = 1

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        inp, out = d / "test.csv", d / "scores.csv"
        df.to_csv(inp, index=False)

        # Output format, channel order, target_channels, both modes.
        for params, expect in [({}, ["channel_1", "channel_10", "channel_2"]),
                               ({"causal": True}, ["channel_1", "channel_10", "channel_2"]),
                               ({"target_channels": ["channel_2", "channel_10", "missing"]}, ["channel_10", "channel_2"])]:
            r = run(exe, inp, out, params)
            assert r.returncode == 0, r.stderr
            y = evaluate_path(out, t)
            assert y.shape == (n, len(expect)), (y.shape, expect)
            assert set(np.unique(y)) <= {0.0, 1.0}, np.unique(y)
            col = expect.index("channel_10")
            others = [i for i in range(len(expect)) if i != col]
            print(f"params={params}: channel_10 flagged {int(y[:12000, col].sum())} before / "
                  f"{int(y[12000:, col].sum())} after the change, other channels {int(y[:, others].sum())}")
            assert y[12000:, col].sum() > 0, "structure change not flagged"
            if params.get("causal"):
                assert y[:12000, col].sum() == 0, "causal mode flagged rows before the change"

        # Causal mode makes no decision inside the baseline, even where the signal is odd there.
        early = df.copy()
        early.loc[:3000, "channel_10"] = np.cumsum(rng.standard_normal(3001)) * 0.05
        early.to_csv(inp, index=False)
        assert run(exe, inp, out, {"causal": True}).returncode == 0
        y = evaluate_path(out, t)
        windows = (n - 1024) // 256 + 1  # noise windows all pass the fit-quality filter
        bl_count = max(int(windows * 0.2), 10)
        first_decision = bl_count * 256 + 1024  # end of the first window after the baseline
        assert y[:first_decision].sum() == 0, "causal mode decided before its baseline existed"
        print(f"causal: no flags before row {first_decision} (baseline = first {bl_count} windows)")

        # A short pure-noise file: too few windows for a baseline, so nothing is flagged.
        short = df.iloc[:3000].copy()
        short["channel_10"] = rng.standard_normal(3000)
        short.to_csv(inp, index=False)
        r = run(exe, inp, out, {})
        assert r.returncode == 0, r.stderr
        y = evaluate_path(out, t[:3000])
        assert y.sum() == 0, f"short noise flagged {int(y.sum())} rows"
        assert "usable windows" in r.stderr, r.stderr
        print("short file: no flags, warning printed")

        # Header with a byte-order mark and quoted names parses to the same channels.
        df.to_csv(inp, index=False)
        text = inp.read_text()
        header, rest = text.split("\n", 1)
        inp.write_text("﻿" + ",".join(f'"{h}"' for h in header.split(",")) + "\n" + rest, encoding="utf-8")
        assert run(exe, inp, out, {}).returncode == 0
        assert evaluate_path(out, t).shape == (n, 3)
        print("BOM and quoted header: 3 channels")

        # window_size below the smallest workable DFA window is raised to it and still detects.
        df.to_csv(inp, index=False)
        r = run(exe, inp, out, {"window_size": 64, "stride": 16})
        assert r.returncode == 0 and "window=72" in r.stderr, r.stderr
        print("window_size 64 raised to 72")

        # An unreadable line fails the run instead of silently truncating the input.
        raw = inp.read_bytes()
        cut = raw.index(b"\n", len(raw) // 2) + 1
        inp.write_bytes(raw[:cut] + b"\xff\xfe,1,2,3,0,0,0\n" + raw[cut:])
        r = run(exe, inp, out, {})
        assert r.returncode == 1, (r.returncode, r.stderr)
        print("unreadable line: exit 1")
    print("ok contract")


if __name__ == "__main__":
    main()
