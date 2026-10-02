"""LSTM: sequential dependencies, last hidden state -> classifier."""
from tensorflow import keras
from keras import layers

def build_model(L, F):
    inp = keras.Input((L, F))
    x = layers.Dropout(.3)(layers.LSTM(32)(inp))
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(x), name="LSTM")
