"""(Legacy PyTorch version; main.py is the Keras entry point.)
Step 2: MLP / 1D-CNN / LSTM / Transformer-encoder comparison for next-day DJIA direction.
Same inputs, same temporal split, same metrics for all models; ablation: market-only vs market+FinBERT."""
import os, json, sys, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from data import ROOT, RES, WINDOW
SEEDS = [0, 1, 2, 3, 4]
EPOCHS, PATIENCE, BS, LR = 100, 15, 32, 1e-3
DEV = "cpu"  # tiny models; CPU is faster than MPS here and deterministic

from data import build, windows, split, metrics

# ---------------- models (all output one logit) ----------------
class MLP(nn.Module):
    def __init__(s, L, F, h=64, p=.3):
        super().__init__()
        s.net = nn.Sequential(nn.Flatten(), nn.Linear(L * F, h), nn.ReLU(), nn.Dropout(p),
                              nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(p), nn.Linear(h // 2, 1))
    def forward(s, x): return s.net(x).squeeze(-1)

class CNN1D(nn.Module):
    def __init__(s, L, F, c=32, p=.3):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv1d(F, c, 3, padding=1), nn.ReLU(), nn.Conv1d(c, c, 3, padding=1), nn.ReLU())
        s.head = nn.Sequential(nn.Dropout(p), nn.Linear(c, 1))
    def forward(s, x): return s.head(s.conv(x.transpose(1, 2)).mean(-1)).squeeze(-1)

class LSTM(nn.Module):
    def __init__(s, L, F, h=32, p=.3):
        super().__init__()
        s.rnn = nn.LSTM(F, h, batch_first=True); s.head = nn.Sequential(nn.Dropout(p), nn.Linear(h, 1))
    def forward(s, x): return s.head(s.rnn(x)[0][:, -1]).squeeze(-1)

class TransformerEnc(nn.Module):
    def __init__(s, L, F, d=32, heads=4, layers=2, p=.2):
        super().__init__()
        s.inp = nn.Linear(F, d); s.pos = nn.Parameter(torch.zeros(1, L, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 64, p, batch_first=True), layers)
        s.head = nn.Sequential(nn.Dropout(p), nn.Linear(d, 1))
    def forward(s, x): return s.head(s.enc(s.inp(x) + s.pos).mean(1)).squeeze(-1)

MODELS = {"MLP": MLP, "CNN": CNN1D, "LSTM": LSTM, "Transformer": TransformerEnc}

# ---------------- train / eval ----------------
def run(name, Xtr, ytr, Xva, yva, Xte, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MODELS[name](Xtr.shape[1], Xtr.shape[2]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-2); lossf = nn.BCEWithLogitsLoss()
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr); Xva_t, yva_t = torch.tensor(Xva), torch.tensor(yva)
    best, bad, state = 1e9, 0, None
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]; opt.zero_grad(); lossf(m(Xtr_t[b]), ytr_t[b]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl = lossf(m(Xva_t), yva_t).item()
        if vl < best - 1e-4: best, bad, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    m.load_state_dict(state); m.eval()
    with torch.no_grad(): p = torch.sigmoid(m(torch.tensor(Xte))).numpy()
    return p

def main():
    f, mk, sc = build()
    print(f"{len(f)} usable days {f.Date.min().date()}..{f.Date.max().date()}, up-rate {f.y.mean():.3f}")
    settings = {"market_only": mk, "market+sentiment": mk + sc}
    rows, store, curves = [], {}, {}
    for sname, cols in settings.items():
        X, y, nr, dates = windows(f, cols, WINDOW)
        tr, va, te = split(len(X))
        mu, sd = X[tr].reshape(-1, X.shape[2]).mean(0), X[tr].reshape(-1, X.shape[2]).std(0) + 1e-8  # train-only stats
        Xn = (X - mu) / sd
        print(f"\n== {sname}: train {tr.stop} val {va.stop - va.start} test {te.stop - te.start}; "
              f"test up-rate {y[te].mean():.3f}")
        maj = int(y[tr].mean() >= .5)
        rows.append(dict(setting=sname, model="Majority-class", **{k: v for k, v in metrics(y[te], np.full(len(y[te]), float(maj))).items()}, acc_std=0))
        for name in MODELS:
            P, ms = [], []
            for sd_ in SEEDS:
                p = run(name, Xn[tr], y[tr], Xn[va], y[va], Xn[te], sd_); P.append(p); ms.append(metrics(y[te], p))
            agg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
            agg["acc_std"] = float(np.std([m["acc"] for m in ms]))
            rows.append(dict(setting=sname, model=name, **agg))
            store[(sname, name)] = (np.mean(P, 0), y[te], nr[te])
            print(f"  {name:12s} acc {agg['acc']:.3f}±{agg['acc_std']:.3f} P {agg['prec']:.3f} R {agg['rec']:.3f} F1 {agg['f1']:.3f}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "metrics.csv"), index=False)

    # confusion matrices (seed-averaged probabilities, threshold .5)
    fig, ax = plt.subplots(2, 4, figsize=(15, 7))
    for r, sname in enumerate(settings):
        for c, name in enumerate(MODELS):
            p, y, _ = store[(sname, name)]
            cm = confusion_matrix(y, (p >= .5).astype(int), labels=[0, 1]); a = ax[r, c]
            a.imshow(cm, cmap="Blues"); a.set_title(f"{name}\n{sname}", fontsize=9)
            for i in range(2):
                for j in range(2): a.text(j, i, cm[i, j], ha="center", va="center")
            a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xlabel("pred"); a.set_ylabel("true")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "confusion_matrices.png"), dpi=130); plt.close()

    # accuracy bar chart: ablation
    piv = df[df.model != "Majority-class"].pivot(index="model", columns="setting", values="acc").loc[list(MODELS)]
    ax = piv.plot.bar(figsize=(7, 4), rot=0); ax.axhline(df[df.model == "Majority-class"].acc.iloc[0], color="k", ls="--", label="majority")
    ax.set_ylim(.4, .65); ax.set_ylabel("test accuracy"); ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RES, "ablation_accuracy.png"), dpi=130); plt.close()

    # simple simulated trading: long when P(up)>=.5 else flat, vs buy&hold (no costs)
    best = max([k for k in store if k[0] == "market+sentiment"], key=lambda k: df[(df.setting == k[0]) & (df.model == k[1])].acc.iloc[0])
    p, y, nr = store[best]; strat = np.where(p >= .5, nr, 0.0)
    sim = dict(best_model=best[1], strategy_return=float(np.expm1(strat.sum())), buy_hold_return=float(np.expm1(nr.sum())),
               days_in_market=float((p >= .5).mean()))
    json.dump(sim, open(os.path.join(RES, "trading_sim.json"), "w"), indent=2)
    print("\nTrading sim (best market+sentiment model, no costs):", sim)
    print("\n", df.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
