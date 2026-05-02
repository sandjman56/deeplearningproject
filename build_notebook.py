"""Builds cmapss_lstm_vs_tcn.ipynb from cell definitions below."""
import json
from pathlib import Path

cells = []

def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    })

def code(text):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": text.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    })


md("""# C-MAPSS Turbofan RUL: LSTM (baseline) vs. TCN (variant)

**Course:** 24-788 Intro to Deep Learning, Spring 2026 — Mini-Project

**Task.** Predict Remaining Useful Life (RUL) of turbofan engines on the NASA C-MAPSS FD001 subset from sensor time series.

**Models.**
- **Baseline:** LSTM (covered in Appendix's Sequences/Time Series list).
- **Variant:** Temporal Convolutional Network (TCN), Bai et al. 2018 ([arXiv:1803.01271](https://arxiv.org/abs/1803.01271)). Stacked dilated causal 1D convolutions. Not in the course Appendix.

**Hypothesis.** TCN's dilated receptive field captures long-range degradation trends in parallel without BPTT bottlenecks; the LSTM's gating should remain competitive at short windows where local recurrence helps. We test this with a window-length sweep.

**Metrics.** RMSE on RUL (cycles) and the PHM asymmetric score (penalizes late predictions more than early ones).
""")

md("## 0. Setup")

code("""import os, sys, time, math, random, zipfile, urllib.request, io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

DATA_DIR = Path("./data/cmapss")
CKPT_DIR = Path("./checkpoints"); CKPT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR  = Path("./figures");     FIG_DIR.mkdir(parents=True, exist_ok=True)
""")

md("""## 1. Download FD001

NASA Prognostics Data Repository. ~25 MB zip; FD001 train/test/RUL files are ~3 MB total.""")

code("""ZIP_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
ZIP_PATH = Path("./cmapss.zip")
DATA_DIR.mkdir(parents=True, exist_ok=True)

if not (DATA_DIR / "train_FD001.txt").exists():
    if not ZIP_PATH.exists():
        print("downloading...")
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(DATA_DIR)
    # The zip nests files; flatten if needed.
    for p in DATA_DIR.rglob("*FD00*.txt"):
        target = DATA_DIR / p.name
        if p != target:
            p.rename(target)
print("files:", sorted(p.name for p in DATA_DIR.glob("*FD001*.txt")))
""")

md("""## 2. Load + preprocess

- Compute RUL per row as `max_cycle_for_unit - cycle`, clipped at 125 (standard practice — early-life RUL signal is uninformative).
- Drop near-constant sensors (FD001 has 7 of them).
- Z-score normalize using **train statistics only** (no leakage).
- Build sliding windows of length W per engine. For test trajectories, take the **last** window only (we predict RUL at the truncation point).
""")

code("""COLS = ["unit", "cycle"] + [f"op_{i}" for i in (1, 2, 3)] + [f"s_{i}" for i in range(1, 22)]

def load_split(name):
    df = pd.read_csv(DATA_DIR / f"{name}_FD001.txt",
                     sep=r"\\s+", header=None, names=COLS, engine="python")
    return df

train_df = load_split("train")
test_df  = load_split("test")
rul_test = pd.read_csv(DATA_DIR / "RUL_FD001.txt", header=None, names=["RUL"])["RUL"].values

print("train rows:", len(train_df), "| units:", train_df.unit.nunique())
print("test  rows:", len(test_df),  "| units:", test_df.unit.nunique())
train_df.head()
""")

code("""RUL_CAP = 125

def add_rul(df, cap=RUL_CAP):
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["RUL"] = (max_cycle - df["cycle"]).clip(upper=cap)
    return df

train_df = add_rul(train_df)
# For test, true RUL is provided per unit (at the final observed cycle); we don't add per-row RUL.

# Drop low-variance sensors (FD001 standard: 1, 5, 6, 10, 16, 18, 19) + constant op settings.
sensor_cols = [c for c in train_df.columns if c.startswith("s_")]
op_cols     = [c for c in train_df.columns if c.startswith("op_")]
stds = train_df[sensor_cols + op_cols].std()
keep = [c for c in sensor_cols + op_cols if stds[c] > 1e-4]
print(f"keeping {len(keep)}/{len(sensor_cols)+len(op_cols)} features:", keep)

# Z-score using train stats only.
mean = train_df[keep].mean()
std  = train_df[keep].std().replace(0, 1.0)
train_df[keep] = (train_df[keep] - mean) / std
test_df[keep]  = (test_df[keep]  - mean) / std

FEATURES = keep
N_FEATURES = len(FEATURES)
""")

code("""def make_train_windows(df, W):
    Xs, ys = [], []
    for _, g in df.groupby("unit", sort=False):
        feats = g[FEATURES].values
        ruls  = g["RUL"].values
        n = len(g)
        if n < W:
            # left-pad with first row (rare in FD001)
            pad = np.repeat(feats[:1], W - n, axis=0)
            feats = np.vstack([pad, feats])
            n = W
        for t in range(W - 1, n):
            Xs.append(feats[t - W + 1 : t + 1])
            ys.append(ruls[t])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

def make_test_windows(df, W):
    # last W rows per unit -> one window per engine
    Xs = []
    for _, g in df.groupby("unit", sort=False):
        feats = g[FEATURES].values
        if len(feats) < W:
            pad = np.repeat(feats[:1], W - len(feats), axis=0)
            feats = np.vstack([pad, feats])
        Xs.append(feats[-W:])
    return np.asarray(Xs, dtype=np.float32)
""")

md("""### Train/val split

Hold out 20 of 100 engines for validation (by unit ID, no leakage).""")

code("""rng = np.random.default_rng(SEED)
all_units = train_df.unit.unique()
val_units = set(rng.choice(all_units, size=20, replace=False).tolist())
tr_units  = [u for u in all_units if u not in val_units]
print(f"train units: {len(tr_units)} | val units: {len(val_units)}")

tr_df  = train_df[train_df.unit.isin(tr_units)]
val_df = train_df[train_df.unit.isin(val_units)]
""")

md("## 3. Models")

code("""class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):                 # x: (B, T, F)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)
""")

code("""class TemporalBlock(nn.Module):
    \"\"\"Causal dilated conv block with residual (Bai et al. 2018).\"\"\"
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.pad = pad
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation))
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation))
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def _causal(self, y):
        return y[..., :-self.pad] if self.pad > 0 else y

    def forward(self, x):
        y = self._causal(self.conv1(x)); y = self.drop(F.relu(y))
        y = self._causal(self.conv2(y)); y = self.drop(F.relu(y))
        res = x if self.down is None else self.down(x)
        return F.relu(y + res)


class TCN(nn.Module):
    def __init__(self, input_dim, channels=(64, 64, 64, 64), kernel=3, dropout=0.2):
        super().__init__()
        blocks = []
        in_ch = input_dim
        for i, ch in enumerate(channels):
            blocks.append(TemporalBlock(in_ch, ch, kernel, dilation=2**i, dropout=dropout))
            in_ch = ch
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(channels[-1], 1)

    def forward(self, x):                 # x: (B, T, F)
        y = x.transpose(1, 2)             # (B, F, T)
        y = self.tcn(y)
        return self.head(y[:, :, -1]).squeeze(-1)
""")

md("## 4. Training")

code("""def phm_score(pred, true):
    \"\"\"NASA PHM 2008 asymmetric score. Late predictions penalized more.\"\"\"
    d = pred - true
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(s.sum())

def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def train_model(model, Xtr, ytr, Xval, yval, epochs=30, batch=512, lr=1e-3, name="model"):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=batch, shuffle=True,
    )
    Xval_t = torch.from_numpy(Xval).to(DEVICE)
    yval_t = torch.from_numpy(yval).to(DEVICE)

    history = {"train_loss": [], "val_rmse": []}
    best_val = math.inf
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0; n = 0
        for xb, yb in tr_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * xb.size(0); n += xb.size(0)
        ep_loss /= n

        model.eval()
        with torch.no_grad():
            vpred = model(Xval_t).cpu().numpy()
        v_rmse = rmse(vpred, yval)
        history["train_loss"].append(ep_loss)
        history["val_rmse"].append(v_rmse)

        if v_rmse < best_val:
            best_val = v_rmse
            torch.save(model.state_dict(), CKPT_DIR / f"{name}_best.pt")
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"  ep {ep:3d} | train_mse {ep_loss:7.3f} | val_rmse {v_rmse:6.3f}")
    print(f"  done in {time.time()-t0:.1f}s | best val_rmse {best_val:.3f}")
    model.load_state_dict(torch.load(CKPT_DIR / f"{name}_best.pt", map_location=DEVICE))
    return model, history
""")

code("""W = 30  # default window length

Xtr, ytr   = make_train_windows(tr_df, W)
Xval, yval = make_train_windows(val_df, W)
Xte        = make_test_windows(test_df, W)
yte        = rul_test.astype(np.float32)
yte_clip   = np.minimum(yte, RUL_CAP).astype(np.float32)  # report against capped target too
print("Xtr", Xtr.shape, "Xval", Xval.shape, "Xte", Xte.shape)

print("\\n--- LSTM ---")
lstm = LSTMRegressor(N_FEATURES)
lstm, hist_lstm = train_model(lstm, Xtr, ytr, Xval, yval, name=f"lstm_W{W}")

print("\\n--- TCN ---")
tcn = TCN(N_FEATURES)
tcn, hist_tcn = train_model(tcn, Xtr, ytr, Xval, yval, name=f"tcn_W{W}")
""")

md("## 5. Evaluation on test set")

code("""@torch.no_grad()
def predict(model, X):
    model.eval()
    out = model(torch.from_numpy(X).to(DEVICE)).cpu().numpy()
    return out

pred_lstm = predict(lstm, Xte)
pred_tcn  = predict(tcn,  Xte)

results = {}
for name, p in [("LSTM", pred_lstm), ("TCN", pred_tcn)]:
    results[name] = {
        "rmse_capped":   rmse(p, yte_clip),
        "rmse_uncapped": rmse(p, yte),
        "phm_score":     phm_score(p, yte),
        "n_params":      sum(x.numel() for x in (lstm if name=="LSTM" else tcn).parameters()),
    }

import json
print(json.dumps(results, indent=2))
""")

md("## 6. Plots")

code("""fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hist_lstm["train_loss"], label="LSTM train MSE")
ax[0].plot(hist_tcn["train_loss"],  label="TCN train MSE")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE"); ax[0].legend(); ax[0].set_title("Training loss")
ax[1].plot(hist_lstm["val_rmse"], label="LSTM val RMSE")
ax[1].plot(hist_tcn["val_rmse"],  label="TCN val RMSE")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("RMSE (cycles)"); ax[1].legend(); ax[1].set_title("Validation RMSE")
fig.tight_layout()
fig.savefig(FIG_DIR / "training_curves.png", dpi=140)
plt.show()
""")

code("""fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
for a, name, p in [(ax[0], "LSTM", pred_lstm), (ax[1], "TCN", pred_tcn)]:
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

md("""## 7. Window-length sweep (analysis hook)

Tests the core hypothesis: TCN's dilated receptive field should help more as W grows; LSTM should be relatively flat or degrade due to BPTT difficulty.""")

code("""sweep = {}
for W_ in [15, 30, 50]:
    Xtr_, ytr_   = make_train_windows(tr_df,  W_)
    Xval_, yval_ = make_train_windows(val_df, W_)
    Xte_         = make_test_windows(test_df, W_)

    print(f"\\n=== W={W_} | train {Xtr_.shape} ===")
    m1 = LSTMRegressor(N_FEATURES); m1, _ = train_model(m1, Xtr_, ytr_, Xval_, yval_, epochs=20, name=f"lstm_W{W_}")
    m2 = TCN(N_FEATURES);           m2, _ = train_model(m2, Xtr_, ytr_, Xval_, yval_, epochs=20, name=f"tcn_W{W_}")
    p1 = predict(m1, Xte_); p2 = predict(m2, Xte_)
    sweep[W_] = {
        "LSTM_rmse": rmse(p1, yte_clip), "LSTM_phm": phm_score(p1, yte),
        "TCN_rmse":  rmse(p2, yte_clip), "TCN_phm":  phm_score(p2, yte),
    }
    print(sweep[W_])

sweep_df = pd.DataFrame(sweep).T
print("\\n", sweep_df)
""")

code("""fig, ax = plt.subplots(1, 2, figsize=(11, 4))
Ws = sorted(sweep.keys())
ax[0].plot(Ws, [sweep[w]["LSTM_rmse"] for w in Ws], "o-", label="LSTM")
ax[0].plot(Ws, [sweep[w]["TCN_rmse"]  for w in Ws], "s-", label="TCN")
ax[0].set_xlabel("window length W"); ax[0].set_ylabel("test RMSE (capped)")
ax[0].set_title("RMSE vs. window length"); ax[0].legend()

ax[1].plot(Ws, [sweep[w]["LSTM_phm"] for w in Ws], "o-", label="LSTM")
ax[1].plot(Ws, [sweep[w]["TCN_phm"]  for w in Ws], "s-", label="TCN")
ax[1].set_xlabel("window length W"); ax[1].set_ylabel("PHM score (lower = better)")
ax[1].set_title("PHM score vs. window length"); ax[1].legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "window_sweep.png", dpi=140)
plt.show()
""")

md("""## 8. Save final results

`reproduce_results.ipynb` reads these checkpoints + this JSON to regenerate the figures without retraining (assignment requirement §5.3).""")

code("""import json
with open(CKPT_DIR / "results.json", "w") as f:
    json.dump({
        "headline": results,
        "sweep": sweep,
        "config": {"W": W, "rul_cap": RUL_CAP, "features": FEATURES, "seed": SEED},
    }, f, indent=2)
print("saved:", CKPT_DIR / "results.json")
print("checkpoints:", sorted(p.name for p in CKPT_DIR.glob("*.pt")))
print("figures:",     sorted(p.name for p in FIG_DIR.glob("*.png")))
""")


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "cmapss_lstm_vs_tcn.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
