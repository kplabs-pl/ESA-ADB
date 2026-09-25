"""Contract test: the output must survive the ESA-ADB evaluation path unchanged.

Mirrors timeeval/adapters/docker.py (np.genfromtxt of the output) and timeeval/core/experiments.py
(scale_scores = per-column MinMax, then max over channels, astype(np.uint8), assigned to the test
timestamps). Run: python contract_test.py path/to/struktura-timeeval
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

exe = sys.argv[1]
rng = np.random.default_rng(0)
n = 20000
t = pd.date_range("2000-01-01", periods=n, freq="30s")
white = rng.standard_normal(n)
walk_after = white.copy()
walk_after[12000:] = np.cumsum(rng.standard_normal(n - 12000)) * 0.05  # structure change, similar spread
df = pd.DataFrame({
    "timestamp": t,
    "channel_2": rng.standard_normal(n),
    "channel_10": walk_after,
    "channel_1": rng.standard_normal(n),
})
for c in ["channel_2", "channel_10", "channel_1"]:
    df[f"is_anomaly_{c}"] = 0
df.loc[12000:, "is_anomaly_channel_10"] = 1

with tempfile.TemporaryDirectory() as d:
    inp, out = Path(d) / "test.csv", Path(d) / "scores.csv"
    df.to_csv(inp, index=False)
    for params, expect in [({}, ["channel_1", "channel_10", "channel_2"]),
                           ({"causal": True}, ["channel_1", "channel_10", "channel_2"]),
                           ({"target_channels": ["channel_2", "channel_10", "missing"]}, ["channel_10", "channel_2"])]:
        cfg = {"dataInput": str(inp), "dataOutput": str(out), "executionType": "execute", "customParameters": params}
        subprocess.run([exe, json.dumps(cfg)], check=True, capture_output=True)
        y = np.genfromtxt(out, delimiter=",")
        y = y.reshape(n, -1) if y.ndim == 1 else y
        assert y.shape == (n, len(expect)), (y.shape, expect)
        assert set(np.unique(y)) <= {0.0, 1.0}, np.unique(y)
        scaled = np.column_stack([MinMaxScaler().fit_transform(y[:, [i]]).ravel() if y[:, i].max() > y[:, i].min() else y[:, i]
                                  for i in range(y.shape[1])])
        ts = pd.DataFrame({"timestamp": t})
        ts["Score"] = scaled.max(axis=1).astype(np.uint8)  # the exact official cast; raised before this fix
        col = expect.index("channel_10")
        others = [i for i in range(len(expect)) if i != col]
        print(f"params={params}: shape {y.shape}, channel_10 flagged {int(y[:12000, col].sum())} before / {int(y[12000:, col].sum())} after the change, "
              f"other channels flagged {int(y[:, others].sum())}")
        assert y[12000:, col].sum() > 0, "structure change not flagged"
        if params.get("causal"):
            assert y[:12000, col].sum() == 0, "causal mode flagged rows before the change"
print("ok contract")
