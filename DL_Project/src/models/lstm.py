"""LSTM: a single small recurrent layer; last hidden state -> classifier."""
from tensorflow import keras
from keras import layers, regularizers

def build_model(L, F):
    inp = keras.Input((L, F))
    x = layers.LSTM(32, dropout=.2, kernel_regularizer=regularizers.l2(1e-3))(inp)
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(layers.Dropout(.3)(x)), name="LSTM")
