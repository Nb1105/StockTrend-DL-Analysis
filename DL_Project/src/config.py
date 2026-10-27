"""Central configuration: paths, experiment settings and training hyper-parameters.

Every script imports from here, so a change in one place applies to the whole pipeline."""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")    # hide TensorFlow C++ info/warning spam
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # deterministic CPU kernels, no oneDNN banner

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"            # always holds the outputs of the most recent run
HISTORY_DIR = RESULTS_DIR / "history"     # one time-stamped copy per run, never overwritten
MODEL_DIR = ROOT / "models_saved"

# ---------------------------------------------------------------- data
SPLIT = (.70, .15, .15)                   # chronological train / validation / test
MAX_WINDOW = 20                           # every model is scored on the same days, whatever its window
CLIP = 5.0                                # standardised inputs are clipped to +-CLIP (2008 crisis outliers)
TEXT_COMPONENTS = 20                      # TF-IDF -> SVD components of the daily headlines

# ---------------------------------------------------------------- training
EPOCHS, PATIENCE, BATCH_SIZE, LR, WEIGHT_DECAY = 200, 15, 32, 1e-3, 1e-2
SEEDS = 5                                 # predictions are averaged over seeds (seed ensemble)

# Per-model input configuration, chosen on the VALIDATION split by src/tune.py (test data never used).
#   window   : days of history per sample
#   market   : "compact" or "extended" market feature set (see data.MARKET_FEATURES)
#   weighted : weight training days by |next-day return| so near-zero (coin-flip) days count less
MODEL_CONFIG = {
    "MLP":         dict(window=20, market="compact",  weighted=True),
    "CNN":         dict(window=10, market="extended", weighted=True),
    "LSTM":        dict(window=5,  market="extended", weighted=True),
    "Transformer": dict(window=10, market="compact",  weighted=True),
}

SETTINGS = ("market_only", "market+sentiment", "market+sentiment+text")
