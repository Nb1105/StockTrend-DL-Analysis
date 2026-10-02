"""1D CNN: local temporal patterns across the window."""
from tensorflow import keras
from keras import layers

def build_model(L, F):
    inp = keras.Input((L, F))
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(inp)
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = layers.Dropout(.3)(layers.GlobalAveragePooling1D()(x))
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(x), name="CNN")
