"""Validation-only search over each model's input configuration (window, market feature set, sample weighting).

The test split is never touched here, so the selected configuration is not fitted to the test set.
Copy the printed best configuration into config.MODEL_CONFIG, then run main.py.

Usage:  python src/tune.py [--models MLP CNN LSTM Transformer] [--seeds 3]"""
import argparse, itertools, time
import numpy as np
import pandas as pd
import config
from data import load, MARKET_FEATURES
from evaluate import metrics
from models import MODELS
from train import train_predict

GRID = dict(window=(5, 10, 20), market=tuple(MARKET_FEATURES), weighted=(False, True))
SETTING = "market+sentiment"   # the project's main setting

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    out_dir = config.RESULTS_DIR / "tuning"; out_dir.mkdir(parents=True, exist_ok=True)

    for name in a.models:
        rows = []
        for window, market, weighted in itertools.product(*GRID.values()):
            t0 = time.time()
            X, y, nr, _, parts = load(SETTING, window, market)
            va = parts[1]
            pv = np.mean([train_predict(name, X, y, nr, parts, s, weighted)[0] for s in range(a.seeds)], 0)
            m = metrics(y[va], pv)
            rows.append(dict(model=name, window=window, market=market, weighted=weighted,
                             val_acc=m["acc"], val_auc=m["auc"]))
            print(f"{name:12s} window={window:2d} market={market:8s} weighted={weighted!s:5s}  "
                  f"val acc {m['acc']:.3f}  val AUC {m['auc']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows).sort_values(["val_acc", "val_auc"], ascending=False)
        df.to_csv(out_dir / f"{name}.csv", index=False)
        b = df.iloc[0]
        print(f"\nBEST {name}: dict(window={b.window}, market=\"{b.market}\", weighted={b.weighted})  "
              f"val acc {b.val_acc:.3f}\n", flush=True)

if __name__ == "__main__":
    main()
