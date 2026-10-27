"""1D CNN: two small causal convolutions over the window, avg+max pooled."""
from tensorflow import keras
from keras import layers, regularizers

def build_model(L, F, c=16):
    inp = keras.Input((L, F))
    x = inp
    for _ in range(2):
        x = layers.Conv1D(c, 3, padding="causal", activation="relu", kernel_regularizer=regularizers.l2(1e-3))(x)
        x = layers.SpatialDropout1D(.2)(x)
    x = layers.Concatenate()([layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)])
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(layers.Dropout(.3)(x)), name="CNN")
