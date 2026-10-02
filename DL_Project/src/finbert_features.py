"""Step 1: FinBERT sentiment extraction (frozen feature extractor, not one of the 4 models).
Scores the top-25 headlines per day with ProsusAI/finbert and aggregates per day.
Output: ../data/finbert_daily.csv (cached; re-run is a no-op)."""
import os, re, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "finbert_daily.csv")

MODEL_ID = "ProsusAI/finbert"
MODEL_DIR = os.path.join(ROOT, "models", "finbert")

def download_finbert():
    """Download the FinBERT model files once into ../models/finbert and reuse them afterwards."""
    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, local_dir=MODEL_DIR)
    return MODEL_DIR

def clean(s):
    if not isinstance(s, str): return ""
    s = re.sub(r"^b['\"]", "", s.strip())
    return s.rstrip("'\"").strip()

def main():
    if os.path.exists(OUT):
        print("cached:", OUT); return
    df = pd.read_csv(os.path.join(ROOT, "data", "Combined_News_DJIA.csv"))
    cols = [f"Top{i}" for i in range(1, 26)]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    path = download_finbert()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(dev).eval()
    # FinBERT label order: positive, negative, neutral
    print(model.config.id2label)
    rows = []
    for i, r in df.iterrows():
        heads = [clean(r[c]) for c in cols]
        heads = [h for h in heads if h]
        enc = tok(heads, padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits, -1).cpu().numpy()
        pos, neg, neu = p[:, 0], p[:, 1], p[:, 2]
        score = pos - neg
        rows.append(dict(Date=r["Date"], s_pos=pos.mean(), s_neg=neg.mean(), s_neu=neu.mean(),
                         s_mean=score.mean(), s_std=score.std(), s_min=score.min(), s_max=score.max()))
        if i % 200 == 0: print(i, len(df), flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("saved", OUT)

if __name__ == "__main__":
    main()
