# 24-788 Mini Project: C-MAPSS RUL, LSTM vs. TCN

Predict Remaining Useful Life (RUL) of turbofan engines on the NASA C-MAPSS **FD001** subset.

- **Baseline (covered in lecture):** LSTM
- **Variant (not in lecture):** Temporal Convolutional Network, Bai et al. 2018, [arXiv:1803.01271](https://arxiv.org/abs/1803.01271)

## Files

| File | Purpose |
|---|---|
| `cmapss_lstm_vs_tcn.ipynb` | Main notebook: download data, train both models, evaluate, plots, save checkpoints. |
| `reproduce_results.ipynb` | Loads saved checkpoints and regenerates test metrics + figures **without retraining**. |
| `checkpoints/` | Best val RMSE weights (`*.pt`) and final `results.json`. |
| `figures/` | PNGs for the report. |
| `data/cmapss/` | Auto downloaded FD001 files (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`). |

## Environment

```bash
pip install torch numpy pandas matplotlib
```

That's it. No PyTorch Geometric, no h5py, no RDKit.

## How to run

### Option A: Google Colab (recommended, free GPU)

1. Open https://colab.research.google.com → **File → Upload notebook** → pick `cmapss_lstm_vs_tcn.ipynb`.
2. **Runtime → Change runtime type → T4 GPU** (Save).
3. **Runtime → Run all**. Total wall time: **~3–5 minutes** including the window sweep.
4. (Optional) Save outputs to Drive: add a first cell `from google.colab import drive; drive.mount('/content/drive')` and change `CKPT_DIR`/`FIG_DIR` to a path under `/content/drive/MyDrive/...`. Otherwise download the `checkpoints/` and `figures/` folders manually before the runtime disconnects.
5. To reproduce from saved checkpoints: upload `reproduce_results.ipynb` next to the same `checkpoints/` and `data/` directories and Run All.

### Option B: Kaggle

1. https://kaggle.com → **Create → New Notebook → File → Import Notebook** → pick `cmapss_lstm_vs_tcn.ipynb`.
2. Right sidebar → **Accelerator: GPU T4 x2** (or P100). Internet must be **on** for the data download.
3. Run All. Outputs land in `/kaggle/working/` and persist with the notebook version.

### Option C: Local (your 16 GB Intel Mac, no GPU)

Will work. FD001 is tiny (100 engines, ~17k training timesteps). Expected wall time on CPU:

| Step | Time |
|---|---|
| Download + preprocess | <30 s |
| Train LSTM (W=30, 30 epochs) | ~3 min |
| Train TCN (W=30, 30 epochs) | ~2 min |
| Window sweep (W ∈ {15, 30, 50}, 20 epochs each, both models) | ~10 min |
| **Total** | **~15 min** |

```bash
cd /Users/sanderschulman/Developer/deeplearningproject
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy pandas matplotlib jupyter
jupyter notebook cmapss_lstm_vs_tcn.ipynb
```

If you want to skip the window sweep on CPU (it's the slow part), just don't run cells 22–24. The headline LSTM vs. TCN comparison at W=30 is enough for the baseline plus one variant scope.

## What the notebook does

1. **Download** the C-MAPSS zip from the NASA Prognostics Data Repository and extracts FD001.
2. **Preprocess:** clip RUL at 125 cycles (standard), drop near constant sensors, z score normalize using train stats only, build sliding windows.
3. **Train** LSTM and TCN with identical optimizer/loss/window/budget for a fair comparison. Save best val RMSE checkpoints.
4. **Evaluate** on the 100 held out test trajectories with **RMSE** and the asymmetric **PHM score** (penalizes late predictions more, late = unsafe).
5. **Plots:** training curves, predicted vs. true scatter, window length sweep.
6. **Save** `results.json` and checkpoints for the reproduce notebook.

## Hypothesis tested

> TCN's dilated receptive field captures long range degradation trends in parallel without BPTT bottlenecks, so it should match or beat LSTM at W=30 and pull further ahead as W grows. LSTM should remain competitive at short windows where local recurrence is enough.

The window length sweep (cell 22) is the experiment that addresses this directly.

## Citations

### Papers and dataset

- Saxena, Goebel, Simon, Eklund, *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM 2008. (C-MAPSS dataset and PHM scoring function.)
- Zheng, Ristovski, Farahat, Gupta, *Long Short-Term Memory Network for Remaining Useful Life Estimation*, ICPHM 2017. (LSTM baseline architecture and the convention of capping RUL at 125 for C-MAPSS.)
- Bai, Kolter, Koltun, *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*, [arXiv:1803.01271](https://arxiv.org/abs/1803.01271), 2018. (TCN variant.)
- Hochreiter, Schmidhuber, *Long Short-Term Memory*, Neural Computation 9(8):1735–1780, 1997. (Original LSTM paper.)
- van den Oord, Dieleman, Zen, Simonyan, Vinyals, Graves, Kalchbrenner, Senior, Kavukcuoglu, *WaveNet: A Generative Model for Raw Audio*, [arXiv:1609.03499](https://arxiv.org/abs/1609.03499), 2016. (Dilated causal convolutions, similar to the TCN block.)
- Kingma, Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015. (Optimizer used for training both models.)
- Li, Ding, Sun, *Remaining Useful Life Estimation in Prognostics Using Deep Convolution Neural Networks*, Reliability Engineering & System Safety 172:1–11, 2018. (Additional reference for C-MAPSS preprocessing conventions.)

### Software libraries

- Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, NeurIPS 2019. ([pytorch.org](https://pytorch.org)). Model definitions, training, GPU/CPU runtime.
- Harris et al., *Array Programming with NumPy*, Nature 585:357–362, 2020. ([numpy.org](https://numpy.org)). Array operations and data tensors.
- McKinney, *Data Structures for Statistical Computing in Python*, SciPy 2010. ([pandas.pydata.org](https://pandas.pydata.org)). Reading and reshaping the C-MAPSS text files.
- Hunter, *Matplotlib: A 2D Graphics Environment*, Computing in Science & Engineering 9(3):90–95, 2007. ([matplotlib.org](https://matplotlib.org)). Training curve, scatter, and window sweep figures.
- Project Jupyter, *Jupyter Notebooks*, [jupyter.org](https://jupyter.org). Notebook runtime for `cmapss_lstm_vs_tcn.ipynb` and `reproduce_results.ipynb`.

All paper references are duplicated in `Project_Template/references.bib` and cited from the LaTeX report.

### Implementation notes

The `TemporalBlock` and `TCN` modules in `cmapss_lstm_vs_tcn.ipynb` follow the original Bai et al. design (causal dilated 1D convolutions with `weight_norm`, residual connection, dropout) but are written from scratch from the paper's description rather than copied from any existing repository. The LSTM regressor head is a standard 2 layer `nn.LSTM` with a small MLP head, no third party code reused. C-MAPSS preprocessing (RUL clipping at 125, dropping near constant sensors, train stat z scoring) follows the conventions established by Zheng et al. 2017 and Li et al. 2018.

## Report

The conference style writeup is in `Project_Template/report_template.tex` and compiles via:

```bash
cd Project_Template
pdflatex report_template.tex
bibtex   report_template
pdflatex report_template.tex
pdflatex report_template.tex
```

## AI tool use

Per the assignment §7 collaboration statement: AI assistance (Claude) was used for boilerplate code generation, data loading and preprocessing scaffolding, notebook structuring, and LaTeX formatting. All hypotheses, experimental design choices, results analysis, and the written report's content are my own.
