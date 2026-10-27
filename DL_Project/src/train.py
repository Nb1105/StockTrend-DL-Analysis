"""Training routine shared by main.py (final runs) and tune.py (validation search)."""
import numpy as np
from tensorflow import keras
from config import EPOCHS, PATIENCE, BATCH_SIZE, LR, WEIGHT_DECAY
from models import MODELS

def sample_weights(y_ret):
    """|next-day return| relative to its mean: big moves carry the signal, near-zero days are mostly noise."""
    w = np.abs(y_ret)
    return (w / w.mean()).astype(np.float32)

def train_predict(name, X, y, nr, parts, seed, weighted=False, save_path=None):
    """Train one model with early stopping on validation loss; return (val_probs, test_probs, epochs_run)."""
    tr, va, te = parts
    keras.utils.set_random_seed(seed)
    m = MODELS[name](X.shape[1], X.shape[2])
    m.compile(keras.optimizers.AdamW(LR, weight_decay=WEIGHT_DECAY, clipnorm=1.0), "binary_crossentropy")
    h = m.fit(X[tr], y[tr], sample_weight=sample_weights(nr[tr]) if weighted else None,
              validation_data=(X[va], y[va]), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True),
                         keras.callbacks.ReduceLROnPlateau(factor=.5, patience=5, min_lr=1e-5)])
    if save_path: m.save(save_path)
    return m.predict(X[va], verbose=0).ravel(), m.predict(X[te], verbose=0).ravel(), len(h.epoch)
