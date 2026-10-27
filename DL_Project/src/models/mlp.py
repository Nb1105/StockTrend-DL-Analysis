"""MLP: small feed-forward baseline over the flattened window (L2 + dropout; low capacity for a noisy target)."""
from tensorflow import keras
from keras import layers, regularizers

def build_model(L, F):
    inp = keras.Input((L, F))
    x = layers.Flatten()(inp)
    for u in (32, 16):
        x = layers.Dense(u, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(x)
        x = layers.Dropout(.3)(x)
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(x), name="MLP")
