"""Builds the figures and a self-contained HTML report (results/report.html) from one run's results."""
import base64, html, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import config
from config import SETTINGS

# categorical slots 1-3 of the reference palette (validated all-pairs for three series)
COLORS = dict(zip(SETTINGS, ("#2a78d6", "#eb6834", "#1baf7a")))
INK, GRID = "#0b0b0b", "#d9d8d4"

def _style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors="#52514e", labelsize=9); ax.yaxis.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)

def _b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def accuracy_chart(df, models):
    rows = [m for m in models] + (["Ensemble(all)"] if "Ensemble(all)" in set(df.model) else [])
    fig, ax = plt.subplots(figsize=(9, 4.2)); w = .26
    for i, s in enumerate(SETTINGS):
        v = [df[(df.setting == s) & (df.model == m)].acc.iloc[0] for m in rows]
        bars = ax.bar(np.arange(len(rows)) + (i - 1) * w, v, w * .92, color=COLORS[s], label=s)
        for b, x in zip(bars, v): ax.text(b.get_x() + b.get_width() / 2, x + .003, f"{x:.3f}", ha="center", fontsize=7, color="#52514e")
    base = df[df.model == "Majority-class"].acc.mean()
    ax.axhline(base, color=INK, ls="--", lw=1, label=f"majority baseline {base:.3f}")
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(rows); ax.set_ylim(.40, .66); ax.set_ylabel("test accuracy")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left"); _style(ax)
    return fig

def confusion_chart(store, models):
    fig, ax = plt.subplots(len(SETTINGS), len(models), figsize=(3.2 * len(models), 3 * len(SETTINGS)), squeeze=False)
    for r, s in enumerate(SETTINGS):
        for c, m in enumerate(models):
            p, y, _ = store[(s, m)]; cm = confusion_matrix(y, (p >= .5).astype(int), labels=[0, 1]); a = ax[r, c]
            a.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.6); a.set_title(f"{m}\n{s}", fontsize=8)
            for i in range(2):
                for j in range(2): a.text(j, i, cm[i, j], ha="center", va="center", fontsize=11)
            a.set_xticks([0, 1], ["down", "up"]); a.set_yticks([0, 1], ["down", "up"]); a.set_xlabel("predicted", fontsize=8); a.set_ylabel("actual", fontsize=8)
    fig.tight_layout()
    return fig

def _table(df):
    cols = ["setting", "model", "acc", "acc_std", "prec", "rec", "f1", "auc", "p_vs_majority"]
    head = "".join(f"<th>{c}</th>" for c in cols); body = ""
    best = {s: df[(df.setting == s) & (~df.model.isin(["Majority-class"]))].acc.max() for s in SETTINGS}
    for _, r in df.iterrows():
        cls = ' class="best"' if r.model != "Majority-class" and r.acc == best[r.setting] else ""
        cells = "".join(f"<td>{r[c]:.3f}</td>" if isinstance(r[c], float) and r[c] == r[c] else f"<td>{html.escape(str(r[c])) if r[c] == r[c] else '–'}</td>" for c in cols)
        body += f"<tr{cls}>{cells}</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

CSS = """:root{--bg:#fcfcfb;--ink:#0b0b0b;--mut:#52514e;--line:#d9d8d4;--hi:#e8f1fc}
@media(prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#f0efec;--mut:#c3c2b7;--line:#3a3a38;--hi:#1f3350}}
body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,sans-serif;max-width:1000px;margin:0 auto;padding:2rem 1.2rem}
h1{margin:0 0 .2rem}h2{margin-top:2.2rem;border-bottom:1px solid var(--line);padding-bottom:.3rem}.sub{color:var(--mut)}
.kpi{display:flex;flex-wrap:wrap;gap:1rem;margin:1.2rem 0}.kpi div{border:1px solid var(--line);border-radius:8px;padding:.7rem 1rem;min-width:150px}
.kpi b{display:block;font-size:1.5rem}.kpi span{color:var(--mut);font-size:.8rem}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.85rem}th,td{padding:.35rem .6rem;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}tr.best{background:var(--hi);font-weight:600}
img{max-width:100%;background:#fff;border-radius:6px}.note{color:var(--mut);font-size:.85rem}"""

def write_report(df, store, sim, meta, models):
    """Refresh figures + HTML report for this run."""
    acc, conf = accuracy_chart(df, models), confusion_chart(store, models)
    acc.savefig(config.RESULTS_DIR / "accuracy_by_model.png", dpi=130, bbox_inches="tight", facecolor="white")
    conf.savefig(config.RESULTS_DIR / "confusion_matrices.png", dpi=130, bbox_inches="tight", facecolor="white")
    a64, c64 = _b64(accuracy_chart(df, models)), _b64(confusion_chart(store, models)); plt.close("all")

    real = df[~df.model.isin(["Majority-class", "Ensemble(all)"])]
    top = real.loc[real.acc.idxmax()]; base = df[df.model == "Majority-class"].acc.mean()
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DJIA direction – results</title><style>{CSS}</style><body>
<h1>Next-day DJIA direction: MLP vs CNN vs LSTM vs Transformer</h1>
<p class="sub">Run {meta['run_at']} · {meta['seeds']} seeds per model · {meta['runtime_min']} min · Python {meta['python']}</p>
<div class="kpi"><div><b>{top.acc:.3f}</b><span>best test accuracy ({html.escape(top.model)}, {html.escape(top.setting)})</span></div>
<div><b>{base:.3f}</b><span>majority-class baseline</span></div>
<div><b>{real.auc.max():.3f}</b><span>best AUC</span></div>
<div><b>{sim['strategy_return']:+.1%}</b><span>{html.escape(sim['best_model'])} long/flat vs buy &amp; hold {sim['buy_hold_return']:+.1%}</span></div></div>
<h2>Test accuracy</h2><img alt="Grouped bar chart of test accuracy per model and setting" src="data:image/png;base64,{a64}">
<h2>All metrics</h2><div class="wrap">{_table(df)}</div>
<p class="note">Highlighted row = best model per setting. <code>p_vs_majority</code> is a one-sided binomial test of accuracy against the majority-class baseline; values above 0.05 mean the gain is not statistically distinguishable from chance.</p>
<h2>Confusion matrices</h2><img alt="Confusion matrices per model and setting" src="data:image/png;base64,{c64}">
<h2>Trading simulation</h2><p>Long when P(up) ≥ 0.5, otherwise flat, no costs, best market+sentiment model ({html.escape(sim['best_model'])}): strategy {sim['strategy_return']:+.2%}, buy &amp; hold {sim['buy_hold_return']:+.2%}, in market {sim['days_in_market']:.0%} of days.</p>
</body></html>"""
    (config.RESULTS_DIR / "report.html").write_text(page, encoding="utf-8")
