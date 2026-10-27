"""Train and compare MLP, 1D-CNN, LSTM and Transformer on next-day DJIA direction, with three input settings
(market only / + FinBERT sentiment / + headline text). Every run refreshes results/ and archives a copy.

Usage:  python src/main.py [--models MLP CNN LSTM Transformer] [--seeds 5] [--no-save]"""
import argparse, json, platform, shutil, time
from datetime import datetime
import numpy as np
import pandas as pd
import config
from config import MODEL_CONFIG, SETTINGS
from data import load
from evaluate import metrics, p_vs_majority
from models import MODELS
from train import train_predict
from report import write_report

def run_experiments(models, seeds, save):
    rows, store = [], {}
    for sname in SETTINGS:
        majority_done = False
        for mi, name in enumerate(models, 1):
            cfg = MODEL_CONFIG[name]
            X, y, nr, dates, parts = load(sname, cfg["window"], cfg["market"])
            tr, va, te = parts
            maj = float(y[tr].mean() >= .5)
            if not majority_done:   # identical test days for every model, so one baseline per setting
                rows.append(dict(setting=sname, model="Majority-class", **metrics(y[te], np.full(len(y[te]), maj)), acc_std=0.0, p_vs_majority=np.nan))
                print(f"\n== {sname}: train {tr.stop}  val {va.stop - va.start}  test {te.stop - te.start}", flush=True)
                majority_done = True
            P, E = [], []
            for seed in range(seeds):
                path = config.MODEL_DIR / f"{name}_{sname.replace('+', '_')}.keras" if (save and seed == 0) else None
                _, p, ep = train_predict(name, X, y, nr, parts, seed, cfg["weighted"], path)
                P.append(p); E.append(ep)
            pe = np.mean(P, 0); agg = metrics(y[te], pe)
            agg["acc_std"] = float(np.std([metrics(y[te], q)["acc"] for q in P]))
            agg["p_vs_majority"] = p_vs_majority(y[te], pe, maj)
            rows.append(dict(setting=sname, model=name, **agg)); store[(sname, name)] = (pe, y[te], nr[te])
            print(f"  [{mi}/{len(models)}] {name:12s} acc {agg['acc']:.3f}±{agg['acc_std']:.3f}  F1 {agg['f1']:.3f}  AUC {agg['auc']:.3f}  "
                  f"p(vs majority) {agg['p_vs_majority']:.2f}  epochs~{int(np.mean(E))}", flush=True)
        if len(models) > 1:   # averaging different architectures reduces variance further
            pe = np.mean([store[(sname, n)][0] for n in models], 0); y_te = store[(sname, models[0])][1]
            agg = metrics(y_te, pe)
            rows.append(dict(setting=sname, model="Ensemble(all)", **agg, acc_std=0.0, p_vs_majority=p_vs_majority(y_te, pe, maj)))
            store[(sname, "Ensemble(all)")] = (pe, y_te, store[(sname, models[0])][2])
            print(f"  {'Ensemble(all)':12s} acc {agg['acc']:.3f}  F1 {agg['f1']:.3f}  AUC {agg['auc']:.3f}", flush=True)
    return pd.DataFrame(rows), store

def trading_sim(df, store):
    """Long when P(up) >= .5 else flat (no costs) with the best market+sentiment model, vs buy-and-hold."""
    cands = [k for k in store if k[0] == "market+sentiment" and k[1] != "Ensemble(all)"]
    best = max(cands, key=lambda k: df[(df.setting == k[0]) & (df.model == k[1])].acc.iloc[0])
    p, _, nr = store[best]; strat = np.where(p >= .5, nr, 0.0)
    return dict(best_model=best[1], strategy_return=float(np.expm1(strat.sum())),
                buy_hold_return=float(np.expm1(nr.sum())), days_in_market=float((p >= .5).mean()))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--seeds", type=int, default=config.SEEDS)
    ap.add_argument("--no-save", action="store_true", help="do not save .keras model files")
    a = ap.parse_args()
    config.RESULTS_DIR.mkdir(exist_ok=True); config.MODEL_DIR.mkdir(exist_ok=True)

    t0 = time.time()
    df, store = run_experiments(a.models, a.seeds, not a.no_save)
    sim = trading_sim(df, store)
    meta = dict(run_at=datetime.now().isoformat(timespec="seconds"), seeds=a.seeds, models=a.models,
                runtime_min=round((time.time() - t0) / 60, 1), python=platform.python_version(), model_config=MODEL_CONFIG)

    # fixed filenames: always the latest run
    df.to_csv(config.RESULTS_DIR / "metrics.csv", index=False)
    (config.RESULTS_DIR / "trading_sim.json").write_text(json.dumps(sim, indent=2))
    (config.RESULTS_DIR / "run_info.json").write_text(json.dumps(meta, indent=2))
    write_report(df, store, sim, meta, a.models)

    # archive a copy of this run
    snap = config.HISTORY_DIR / datetime.now().strftime("%Y%m%d_%H%M%S"); snap.mkdir(parents=True)
    for f in ("metrics.csv", "trading_sim.json", "run_info.json", "report.html"):
        shutil.copy(config.RESULTS_DIR / f, snap / f)
    print("\nTrading sim:", sim)
    print(f"\nResults refreshed in {config.RESULTS_DIR}\n  open: {config.RESULTS_DIR / 'report.html'}\n  archived: {snap}")

if __name__ == "__main__":
    main()
