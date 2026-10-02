"""MLP: feed-forward baseline over the flattened window."""
from tensorflow import keras
from keras import layers

def build_model(L, F):
    inp = keras.Input((L, F))
    x = layers.Flatten()(inp)
    x = layers.Dropout(.3)(layers.Dense(64, activation="relu")(x))
    x = layers.Dropout(.3)(layers.Dense(32, activation="relu")(x))
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(x), name="MLP")
