"""Main entry point: trains and compares MLP, 1D-CNN, LSTM and Transformer (Keras) on next-day DJIA direction,
market-only vs market+FinBERT-sentiment. Saves metrics, plots, a trading simulation and .keras model files.

Usage:  python src/main.py [--models MLP CNN LSTM Transformer] [--seeds 5] [--no-save]"""
import os, sys, json, argparse, shutil, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tensorflow as tf
from tensorflow import keras
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from data import build, windows, split, metrics, WINDOW, ROOT, RES
from models import MODELS

EPOCHS, PATIENCE, BS, LR = 100, 15, 32, 1e-3
MODEL_DIR = os.path.join(ROOT, "models_saved")

def train_predict(name, Xtr, ytr, Xva, yva, Xte, seed, save_path=None):
    keras.utils.set_random_seed(seed)
    m = MODELS[name](Xtr.shape[1], Xtr.shape[2])
    m.compile(keras.optimizers.AdamW(LR, weight_decay=1e-2), "binary_crossentropy")
    m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS, batch_size=BS, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)])
    if save_path: m.save(save_path)
    return m.predict(Xte, verbose=0).ravel()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--no-save", action="store_true", help="do not save .keras model files")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)

    f, mk, sc = build()
    print(f"{len(f)} usable days {f.Date.min().date()}..{f.Date.max().date()}, up-rate {f.y.mean():.3f}")
    settings = {"market_only": mk, "market+sentiment": mk + sc}
    rows, store = [], {}
    for sname, cols in settings.items():
        X, y, nr, _ = windows(f, cols, WINDOW)
        tr, va, te = split(len(X))
        flat = X[tr].reshape(-1, X.shape[2]); Xn = (X - flat.mean(0)) / (flat.std(0) + 1e-8)  # train-only stats
        print(f"\n== {sname}: train {tr.stop} val {va.stop - va.start} test {te.stop - te.start}")
        maj = float(y[tr].mean() >= .5)
        rows.append(dict(setting=sname, model="Majority-class", **metrics(y[te], np.full(len(y[te]), maj)), acc_std=0.0))
        for name in a.models:
            P, ms = [], []
            for seed in range(a.seeds):
                path = None if (a.no_save or seed) else os.path.join(MODEL_DIR, f"{name}_{sname.replace('+', '_')}.keras")
                p = train_predict(name, Xn[tr], y[tr], Xn[va], y[va], Xn[te], seed, path); P.append(p); ms.append(metrics(y[te], p))
            agg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}; agg["acc_std"] = float(np.std([m["acc"] for m in ms]))
            rows.append(dict(setting=sname, model=name, **agg)); store[(sname, name)] = (np.mean(P, 0), y[te], nr[te])
            print(f"  {name:12s} acc {agg['acc']:.3f}±{agg['acc_std']:.3f} P {agg['prec']:.3f} R {agg['rec']:.3f} F1 {agg['f1']:.3f}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "metrics_keras.csv"), index=False)

    # confusion matrices
    fig, ax = plt.subplots(2, len(a.models), figsize=(3.7 * len(a.models), 7), squeeze=False)
    for r, sname in enumerate(settings):
        for c, name in enumerate(a.models):
            p, y, _ = store[(sname, name)]; cm = confusion_matrix(y, (p >= .5).astype(int), labels=[0, 1]); ax_ = ax[r, c]
            ax_.imshow(cm, cmap="Blues"); ax_.set_title(f"{name}\n{sname}", fontsize=9)
            for i in range(2):
                for j in range(2): ax_.text(j, i, cm[i, j], ha="center", va="center")
            ax_.set_xticks([0, 1]); ax_.set_yticks([0, 1]); ax_.set_xlabel("pred"); ax_.set_ylabel("true")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "confusion_matrices_keras.png"), dpi=130); plt.close()

    # ablation bar chart
    piv = df[df.model != "Majority-class"].pivot(index="model", columns="setting", values="acc").loc[a.models]
    ax_ = piv.plot.bar(figsize=(7, 4), rot=0); ax_.axhline(df[df.model == "Majority-class"].acc.iloc[0], color="k", ls="--", label="majority")
    ax_.set_ylim(.4, .65); ax_.set_ylabel("test accuracy"); ax_.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RES, "ablation_accuracy_keras.png"), dpi=130); plt.close()

    # simple simulated trading (long if P(up)>=.5 else flat, no costs) with the best market+sentiment model
    best = max([k for k in store if k[0] == "market+sentiment"], key=lambda k: df[(df.setting == k[0]) & (df.model == k[1])].acc.iloc[0])
    p, y, nr = store[best]; strat = np.where(p >= .5, nr, 0.0)
    sim = dict(best_model=best[1], strategy_return=float(np.expm1(strat.sum())), buy_hold_return=float(np.expm1(nr.sum())), days_in_market=float((p >= .5).mean()))
    json.dump(sim, open(os.path.join(RES, "trading_sim_keras.json"), "w"), indent=2)
    print("\nTrading sim:", sim); print("\n", df.round(3).to_string(index=False))

    # Colab convenience: offer saved models as a browser download
    try:
        from google.colab import files
        files.download(shutil.make_archive(os.path.join(ROOT, "models_saved"), "zip", MODEL_DIR))
    except ImportError:
        print("Model files saved in", MODEL_DIR)

if __name__ == "__main__":
    main()
