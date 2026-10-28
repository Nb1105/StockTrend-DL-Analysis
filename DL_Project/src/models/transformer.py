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

def build_model(L, F, d=16, heads=2, blocks=1, p=.3):
    inp = keras.Input((L, F))
    x = PositionalEmbedding(L, d)(layers.Dense(d)(inp))
    for _ in range(blocks):
        n = layers.LayerNormalization()(x)  # pre-norm blocks train more stably on small data
        x = x + layers.Dropout(p)(layers.MultiHeadAttention(num_heads=heads, key_dim=d // heads, dropout=p)(n, n))
        h = layers.Dense(d)(layers.Dropout(p)(layers.Dense(32, activation="gelu")(layers.LayerNormalization()(x))))
        x = x + layers.Dropout(p)(h)
    x = layers.LayerNormalization()(x)
    x = layers.Concatenate()([layers.GlobalAveragePooling1D()(x), x[:, -1]])  # mean + most recent step
    x = layers.Dropout(p)(x)
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(x), name="Transformer")
