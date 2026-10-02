from .mlp import build_model as build_mlp
from .cnn import build_model as build_cnn
from .lstm import build_model as build_lstm
from .transformer import build_model as build_transformer

MODELS = {"MLP": build_mlp, "CNN": build_cnn, "LSTM": build_lstm, "Transformer": build_transformer}
