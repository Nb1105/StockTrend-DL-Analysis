# Next-Day DJIA Direction Prediction: MLP vs 1D-CNN vs LSTM vs Transformer

Predicts whether the Dow Jones Industrial Average closes up or down on the next trading day, using
market data (OHLCV-derived features), FinBERT sentiment and TF-IDF text features of daily Reddit headlines
(Kaggle "Daily News for Stock Market Prediction"). Four Keras architectures are compared under three
input settings (an ablation):

| Setting | Inputs |
|---|---|
| `market_only` | technical features from OHLCV |
| `market+sentiment` | + FinBERT daily sentiment statistics |
| `market+sentiment+text` | + 20 TF-IDF/SVD components of the headlines |

## Project structure
```
DL_Project/
├── data/                    dataset CSVs + finbert_daily.csv (precomputed FinBERT sentiment)
├── docs/                    project synopsis
├── models_saved/            trained .keras models (seed 0 of each model/setting)
├── results/                 latest run (overwritten each run)
│   ├── report.html          open this: KPIs, charts, full metrics table
│   ├── metrics.csv, trading_sim.json, run_info.json
│   ├── accuracy_by_model.png, confusion_matrices.png
│   ├── tuning/              validation-only search results (per model)
│   └── history/<timestamp>/ archived copy of every run
├── src/
│   ├── config.py            paths, hyper-parameters, per-model input configuration
│   ├── data.py              features, windowing, chronological split, scaling
│   ├── models/              mlp.py, cnn.py, lstm.py, transformer.py
│   ├── train.py             training routine (early stopping, LR schedule)
│   ├── evaluate.py          metrics + significance test vs majority baseline
│   ├── tune.py              validation-only configuration search
│   ├── report.py            figures + HTML report
│   ├── main.py              full experiment (entry point)
│   └── finbert_features.py  FinBERT sentiment extraction (only if finbert_daily.csv is missing)
└── requirements.txt
```

## Setup (Windows, PowerShell, from this folder)
TensorFlow supports Python 3.10 - 3.12 (not 3.13+).
```powershell
py -3.12 -m venv ..\.venv312
..\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process Bypass`.

## Run
```powershell
python src\main.py                              # full experiment, 5 seeds, all models (~15-20 min on CPU)
python src\main.py --models LSTM Transformer    # subset
python src\main.py --seeds 3 --no-save          # faster, no .keras files
python src\tune.py                              # optional: re-select per-model inputs on the validation split (~30 min)
start results\report.html                       # view the results
```
Every `main.py` run refreshes the files in `results/` and stores a copy in `results/history/`.

## Method
- Chronological 70/15/15 split; standardisation uses training data only (clipped to ±5).
- All models are scored on the **same** test days, whatever their window length.
- Per-model window, market feature set and loss weighting (by |next-day return|) are selected on the
  **validation** split (`src/tune.py`); the test set is never used for selection.
- Each model is trained with 5 seeds and the probabilities averaged; an all-model ensemble is also reported.
- Each model is compared with the majority-class baseline using a one-sided binomial test (`p_vs_majority`).

## Caveat
Next-day index direction is close to a coin flip. Differences of a few accuracy points on ~290 test days are
within sampling noise; use `p_vs_majority` and AUC, not accuracy alone, to judge whether a model has real signal.
