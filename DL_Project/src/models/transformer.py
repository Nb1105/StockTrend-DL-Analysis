"""Transformer encoder: self-attention across the time steps of the window."""
import tensorflow as tf
from tensorflow import keras
from keras import layers

@keras.utils.register_keras_serializable(package="stocknews")
class PositionalEmbedding(layers.Layer):
    """Learned positional encoding added to the inputs."""
    def __init__(self, length, dim, **kw):
        super().__init__(**kw); self.length, self.dim = length, dim
        self.emb = layers.Embedding(length, dim)
    def call(self, x): return x + self.emb(tf.range(self.length))
    def get_config(self): return {**super().get_config(), "length": self.length, "dim": self.dim}

def build_model(L, F, d=32, heads=4, blocks=2, p=.2):
    inp = keras.Input((L, F))
    x = PositionalEmbedding(L, d)(layers.Dense(d)(inp))
    for _ in range(blocks):
        a = layers.MultiHeadAttention(num_heads=heads, key_dim=d // heads, dropout=p)(x, x)
        x = layers.LayerNormalization()(x + a)
        h = layers.Dense(d)(layers.Dense(64, activation="relu")(x))
        x = layers.LayerNormalization()(x + layers.Dropout(p)(h))
    x = layers.Dropout(p)(layers.GlobalAveragePooling1D()(x))
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(x), name="Transformer")
