# Stock Direction Prediction: MLP vs 1D-CNN vs LSTM vs Transformer

Predicts next-day DJIA direction from market data (OHLCV) and FinBERT sentiment of Reddit headlines
(Kaggle "Daily News for Stock Market Prediction"). Compares four Keras models, with and without sentiment.

## Folder layout
```
data/                 dataset CSVs + finbert_daily.csv (precomputed FinBERT sentiment)
src/data.py           features, windowing, temporal split, metrics
src/finbert_features.py   FinBERT sentiment extraction (optional if finbert_daily.csv exists)
src/models/           mlp.py, cnn.py, lstm.py, transformer.py
src/main.py           runs all four models and writes results
src/train.py          legacy PyTorch version (not needed)
results/              metrics, plots, trading simulation
requirements.txt
```

## Run on Windows (PowerShell, from this folder)

1. Install Python 3.10, 3.11 or 3.12 (TensorFlow does not support 3.13+ yet). Tick "Add Python to PATH".

2. Create environment and install packages:
```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process Bypass`, then activate again.

3. Check the data. If `data\finbert_daily.csv` exists (it should be in the zip), skip to step 5.
   Otherwise download the dataset and compute sentiment:
```powershell
mkdir data
curl.exe -L -o data\stocknews.zip https://www.kaggle.com/api/v1/datasets/download/aaron7sun/stocknews
Expand-Archive data\stocknews.zip -DestinationPath data -Force
```

4. (Only if finbert_daily.csv is missing) Extract FinBERT sentiment. Downloads about 440 MB into
   `models\finbert`; runs on CPU, so allow a long time (about 2,000 days x 25 headlines):
```powershell
python src\finbert_features.py
```

5. Train and compare all four models:
```powershell
python src\main.py
```
Options:
```powershell
python src\main.py --models LSTM Transformer   # only some models
python src\main.py --seeds 3                   # fewer seeds, faster
python src\main.py --no-save                   # do not save .keras model files
```

## Outputs
- `results\metrics_keras.csv` : accuracy, precision, recall, F1 (mean over seeds) per model and setting
- `results\confusion_matrices_keras.png`, `results\ablation_accuracy_keras.png`
- `results\trading_sim_keras.json` : simple long/flat simulation vs buy-and-hold
- `models_saved\*.keras` : trained model files

## Notes
- Split is chronological 70/15/15; normalisation uses training data only.
- Preliminary PyTorch results: all models near chance (about 0.47 to 0.52 accuracy; majority baseline 0.507);
  sentiment gave a small gain; LSTM + sentiment was best at 0.518.
