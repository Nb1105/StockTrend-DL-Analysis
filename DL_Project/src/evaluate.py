"""Classification metrics and a significance test against the majority-class baseline."""
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def metrics(y, p, t=.5):
    yh = (p >= t).astype(int)
    auc = roc_auc_score(y, p) if len(set(y)) > 1 and len(set(p)) > 1 else .5
    return dict(acc=accuracy_score(y, yh), prec=precision_score(y, yh, zero_division=0),
                rec=recall_score(y, yh, zero_division=0), f1=f1_score(y, yh, zero_division=0), auc=auc)

def p_vs_majority(y, p, majority):
    """One-sided binomial p-value: is the accuracy genuinely above always predicting the majority class?"""
    base = float(np.mean(y == majority))
    return float(binomtest(int(((p >= .5) == y).sum()), len(y), base, alternative="greater").pvalue)
