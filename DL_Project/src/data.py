"""Shared data pipeline: features, windowing, temporal split, metrics (framework independent)."""
import os, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
WINDOW = 5

def build():
    m = pd.read_csv(os.path.join(ROOT, "data", "upload_DJIA_table.csv"), parse_dates=["Date"]).sort_values("Date")
    s = pd.read_csv(os.path.join(ROOT, "data", "finbert_daily.csv"), parse_dates=["Date"])
    d = m.merge(s, on="Date").reset_index(drop=True)
    f = pd.DataFrame({"Date": d.Date})
    # scale-free market features from Open/High/Low/Close/Volume (only info available at day t)
    f["ret_close"] = np.log(d.Close / d.Close.shift(1))
    f["ret_open"] = np.log(d.Open / d.Close.shift(1))      # overnight gap
    f["intraday"] = np.log(d.Close / d.Open)
    f["hl_range"] = np.log(d.High / d.Low)
    f["vol_chg"] = np.log(d.Volume / d.Volume.shift(1))
    mk = ["ret_close", "ret_open", "intraday", "hl_range", "vol_chg"]
    sc = ["s_pos", "s_neg", "s_neu", "s_mean", "s_std", "s_min", "s_max"]
    for c in sc: f[c] = d[c]
    # target: direction of NEXT trading day's close vs today's close
    f["y"] = (d.Close.shift(-1) > d.Close).astype(float)
    f["next_ret"] = np.log(d.Close.shift(-1) / d.Close)
    f = f.iloc[1:-1].reset_index(drop=True)  # drop rows lost to shift/lead
    return f, mk, sc

def windows(f, cols, L):
    X = f[cols].values.astype(np.float32)
    idx = np.arange(L - 1, len(f))
    Xs = np.stack([X[i - L + 1:i + 1] for i in idx])
    return Xs, f.y.values[idx].astype(np.float32), f.next_ret.values[idx], f.Date.values[idx]

def split(n):  # chronological 70/15/15
    return slice(0, int(.7 * n)), slice(int(.7 * n), int(.85 * n)), slice(int(.85 * n), n)

def metrics(y, p):
    yh = (p >= .5).astype(int)
    return dict(acc=accuracy_score(y, yh), prec=precision_score(y, yh, zero_division=0),
                rec=recall_score(y, yh, zero_division=0), f1=f1_score(y, yh, zero_division=0))


if __name__ == "__main__":
    print("\n========== DATA PIPELINE STARTED ==========\n")

    # Step 1: Build the dataset
    f, mk, sc = build()

    print("1. DATASET CREATED")
    print("Shape:", f.shape)
    print("\nFirst 5 rows:")
    print(f.head())

    # Step 2: Display feature information
    print("\n2. MARKET FEATURES:")
    print(mk)

    print("\n3. SENTIMENT FEATURES:")
    print(sc)

    # Step 3: Generate windows
    all_features = mk + sc
    X, y, next_ret, dates = windows(f, all_features, WINDOW)

    print("\n4. WINDOWING RESULTS")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Returns shape:", next_ret.shape)
    print("Dates shape:", dates.shape)

    # Step 4: Chronological split
    train, val, test = split(len(X))

    print("\n5. DATA SPLIT")
    print("Training samples:", len(X[train]))
    print("Validation samples:", len(X[val]))
    print("Testing samples:", len(X[test]))

    print("\n6. TARGET DISTRIBUTION")
    print(pd.Series(y).value_counts())

    print("\n========== DATA PIPELINE COMPLETED ==========")