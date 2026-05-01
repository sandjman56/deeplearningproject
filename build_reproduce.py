"""Builds reproduce_results.ipynb — loads saved checkpoints, regenerates figures + metrics
without retraining. Required by assignment §5.3."""
import json
from pathlib import Path

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "source": t.splitlines(keepends=True), "outputs": [], "execution_count": None})


md("""# Reproduce Results — C-MAPSS LSTM vs. TCN

Runs **inference only** from saved checkpoints in `./checkpoints/`. No training.

Run `cmapss_lstm_vs_tcn.ipynb` first to generate the checkpoints, then run this notebook top-to-bottom.""")

code("""import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("./data/cmapss")
CKPT_DIR = Path("./checkpoints")
FIG_DIR  = Path("./figures"); FIG_DIR.mkdir(exist_ok=True)

cfg = json.load(open(CKPT_DIR / "results.json"))["config"]
FEATURES = cfg["features"]; W = cfg["W"]; RUL_CAP = cfg["rul_cap"]; N_FEATURES = len(FEATURES)
print("config:", cfg)
""")

code("""# ---- model defs (must match training notebook) ----
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers,
                            dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation; self.pad = pad
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation))
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation))
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
    def _causal(self, y): return y[..., :-self.pad] if self.pad > 0 else y
    def forward(self, x):
        y = self._causal(self.conv1(x)); y = self.drop(F.relu(y))
        y = self._causal(self.conv2(y)); y = self.drop(F.relu(y))
        res = x if self.down is None else self.down(x)
        return F.relu(y + res)

class TCN(nn.Module):
    def __init__(self, input_dim, channels=(64,64,64,64), kernel=3, dropout=0.2):
        super().__init__()
        blocks = []; in_ch = input_dim
        for i, ch in enumerate(channels):
            blocks.append(TemporalBlock(in_ch, ch, kernel, dilation=2**i, dropout=dropout))
            in_ch = ch
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(channels[-1], 1)
    def forward(self, x):
        y = x.transpose(1, 2); y = self.tcn(y)
        return self.head(y[:, :, -1]).squeeze(-1)
""")

code("""# ---- load + preprocess test set (same recipe) ----
COLS = ["unit", "cycle"] + [f"op_{i}" for i in (1,2,3)] + [f"s_{i}" for i in range(1, 22)]
def load_split(name):
    return pd.read_csv(DATA_DIR / f"{name}_FD001.txt", sep=r"\\s+", header=None, names=COLS, engine="python")

train_df = load_split("train")
test_df  = load_split("test")
yte = pd.read_csv(DATA_DIR / "RUL_FD001.txt", header=None, names=["RUL"])["RUL"].values.astype(np.float32)
yte_clip = np.minimum(yte, RUL_CAP).astype(np.float32)

mean = train_df[FEATURES].mean(); std = train_df[FEATURES].std().replace(0, 1.0)
test_df[FEATURES] = (test_df[FEATURES] - mean) / std

def make_test_windows(df, W):
    Xs = []
    for _, g in df.groupby("unit", sort=False):
        feats = g[FEATURES].values
        if len(feats) < W:
            pad = np.repeat(feats[:1], W - len(feats), axis=0)
            feats = np.vstack([pad, feats])
        Xs.append(feats[-W:])
    return np.asarray(Xs, dtype=np.float32)

Xte = make_test_windows(test_df, W)
print("Xte:", Xte.shape)
""")

code("""# ---- load checkpoints + predict ----
def rmse(p, t): return float(np.sqrt(np.mean((p - t) ** 2)))
def phm_score(p, t):
    d = p - t
    return float(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0).sum())

@torch.no_grad()
def predict(model, X):
    model.eval()
    return model(torch.from_numpy(X).to(DEVICE)).cpu().numpy()

lstm = LSTMRegressor(N_FEATURES).to(DEVICE)
lstm.load_state_dict(torch.load(CKPT_DIR / f"lstm_W{W}_best.pt", map_location=DEVICE))
tcn = TCN(N_FEATURES).to(DEVICE)
tcn.load_state_dict(torch.load(CKPT_DIR / f"tcn_W{W}_best.pt", map_location=DEVICE))

p_lstm = predict(lstm, Xte); p_tcn = predict(tcn, Xte)

results = {
    "LSTM": {"rmse_capped": rmse(p_lstm, yte_clip), "rmse_uncapped": rmse(p_lstm, yte), "phm_score": phm_score(p_lstm, yte)},
    "TCN":  {"rmse_capped": rmse(p_tcn,  yte_clip), "rmse_uncapped": rmse(p_tcn,  yte), "phm_score": phm_score(p_tcn,  yte)},
}
print(json.dumps(results, indent=2))
""")

code("""# ---- regenerate pred-vs-true scatter ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
for a, name, p in [(ax[0], "LSTM", p_lstm), (ax[1], "TCN", p_tcn)]:
    a.scatter(yte, p, s=18, alpha=0.7)
    lo, hi = 0, max(yte.max(), p.max(), RUL_CAP) + 5
    a.plot([lo, hi], [lo, hi], "k--", lw=1)
    a.axhline(RUL_CAP, ls=":", color="gray", lw=1, label=f"cap={RUL_CAP}")
    a.set_xlabel("true RUL")
    a.set_title(f"{name} (test RMSE={results[name]['rmse_capped']:.2f})")
    a.legend(loc="lower right")
ax[0].set_ylabel("predicted RUL")
fig.tight_layout()
fig.savefig(FIG_DIR / "pred_vs_true.png", dpi=140)
plt.show()
""")


nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "reproduce_results.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
