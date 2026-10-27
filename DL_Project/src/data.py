"""Data pipeline: feature engineering, sliding windows, chronological split and scaling (framework independent)."""
import re
from functools import lru_cache
import numpy as np
import pandas as pd
from config import DATA_DIR, SPLIT, MAX_WINDOW, CLIP, TEXT_COMPONENTS

# compact sets generalise better with ~1,350 training days; "extended" adds trend / regime features
MARKET_FEATURES = {
    "compact": ["ret_close", "ret_open", "intraday", "hl_range", "vol_chg", "mom_5", "vol_5", "rsi"],
    "extended": ["ret_close", "ret_open", "intraday", "hl_range", "close_pos", "vol_chg", "vol_z", "mom_5", "mom_20",
                 "vol_5", "vol_20", "ma_ratio", "dist_ma50", "rsi"],
}
SENTIMENT_FEATURES = ["s_pos", "s_neg", "s_mean", "s_std", "s_mean_3", "s_dev"]
TEXT_FEATURES = [f"t_{i}" for i in range(TEXT_COMPONENTS)]

@lru_cache(maxsize=1)
def build():
    """One row per trading day: market features, FinBERT sentiment, headline text components and the label (cached)."""
    m = pd.read_csv(DATA_DIR / "upload_DJIA_table.csv", parse_dates=["Date"]).sort_values("Date")
    s = pd.read_csv(DATA_DIR / "finbert_daily.csv", parse_dates=["Date"])
    d = m.merge(s, on="Date").reset_index(drop=True)
    f = pd.DataFrame({"Date": d.Date})
    C = d.Close
    r = np.log(C / C.shift(1))
    # scale-free market features, all computed from information available at the close of day t
    f["ret_close"] = r
    f["ret_open"] = np.log(d.Open / C.shift(1))            # overnight gap
    f["intraday"] = np.log(C / d.Open)
    f["hl_range"] = np.log(d.High / d.Low)
    f["close_pos"] = (C - d.Low) / (d.High - d.Low + 1e-9)  # where the close sits in the day's range
    f["vol_chg"] = np.log(d.Volume / d.Volume.shift(1))
    lv = np.log(d.Volume)
    f["vol_z"] = (lv - lv.rolling(20).mean()) / (lv.rolling(20).std() + 1e-9)
    for k in (5, 10, 20): f[f"mom_{k}"] = np.log(C / C.shift(k))
    for k in (5, 20): f[f"vol_{k}"] = r.rolling(k).std()
    f["ma_ratio"] = C.rolling(5).mean() / C.rolling(20).mean() - 1
    f["dist_ma50"] = C / C.rolling(50).mean() - 1
    g, l = r.clip(lower=0).rolling(14).mean(), (-r.clip(upper=0)).rolling(14).mean()
    f["rsi"] = g / (g + l + 1e-12) - .5                     # centred RSI(14)
    for c in ("s_pos", "s_neg", "s_neu", "s_mean", "s_std", "s_min", "s_max"): f[c] = d[c]
    # smoothed / relative sentiment: news is noisy day to day, so use rolling levels and deviation from the trend
    f["s_mean_3"] = d.s_mean.rolling(3).mean()
    f["s_dev"] = d.s_mean - d.s_mean.rolling(30).mean()
    # target: direction of the NEXT trading day's close vs today's close
    f["y"] = (C.shift(-1) > C).astype(float)
    f["next_ret"] = np.log(C.shift(-1) / C)
    f = f.iloc[50:-1].reset_index(drop=True)  # drop warm-up rows (50-day MA) and the last row (no label)
    add_text_features(f)
    return f

def add_text_features(f, k=TEXT_COMPONENTS):
    """TF-IDF of each day's 25 Reddit headlines -> k SVD components (t_0..t_{k-1}), fitted on training days only."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    n = pd.read_csv(DATA_DIR / "Combined_News_DJIA.csv", parse_dates=["Date"])
    tops = [c for c in n.columns if c.startswith("Top")]
    n["txt"] = n[tops].fillna("").astype(str).apply(lambda r: " ".join(re.sub(r"^b['\"]|['\"]$", "", s) for s in r), axis=1)
    docs = f[["Date"]].merge(n[["Date", "txt"]], on="Date", how="left").txt.fillna("").values
    fit = int(.65 * len(f))  # strictly inside the training split
    tf = TfidfVectorizer(stop_words="english", min_df=5, max_df=.5, sublinear_tf=True, ngram_range=(1, 2), max_features=20000).fit(docs[:fit])
    svd = TruncatedSVD(k, random_state=0).fit(tf.transform(docs[:fit]))
    Z = svd.transform(tf.transform(docs))
    for i in range(k): f[f"t_{i}"] = Z[:, i]

def columns(setting, market="compact"):
    """Input columns for an ablation setting."""
    cols = list(MARKET_FEATURES[market])
    if "sentiment" in setting: cols += SENTIMENT_FEATURES
    if "text" in setting: cols += TEXT_FEATURES
    return cols

def windows(f, cols, L):
    """Sliding windows of L days. Samples start at MAX_WINDOW-1 so every window length yields the same days."""
    X = f[cols].values.astype(np.float32)
    idx = np.arange(MAX_WINDOW - 1, len(f))
    Xs = np.stack([X[i - L + 1:i + 1] for i in idx])
    return Xs, f.y.values[idx].astype(np.float32), f.next_ret.values[idx], f.Date.values[idx]

def split(n):
    """Chronological train / validation / test slices."""
    a, b = int(SPLIT[0] * n), int((SPLIT[0] + SPLIT[1]) * n)
    return slice(0, a), slice(a, b), slice(b, n)

def standardise(X, tr):
    """Z-score with training-split statistics only, then clip extreme values."""
    flat = X[tr].reshape(-1, X.shape[2])
    return np.clip((X - flat.mean(0)) / (flat.std(0) + 1e-8), -CLIP, CLIP).astype(np.float32)

def load(setting, window, market="compact"):
    """Ready-to-train arrays for one setting: (X, y, next_ret, dates, (train, val, test))."""
    X, y, nr, dates = windows(build(), columns(setting, market), window)
    parts = split(len(X))
    return standardise(X, parts[0]), y, nr, dates, parts
