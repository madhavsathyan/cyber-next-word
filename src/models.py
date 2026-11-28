import json
from tensorflow import keras
from tensorflow.keras import layers

cfg = json.load(open("artifacts/config.json"))
VOCAB_SIZE = cfg["vocab_size"]
MAX_CONTEXT = cfg["max_context"]

def build_model(kind="lstm", embed_dim=128, units=128, dropout=0.3, lr=1e-3):
    """kind: 'simplernn', 'lstm' or 'gru'. Everything else is identical."""
    rnn_layers = {
        "simplernn": layers.SimpleRNN,
        "lstm": layers.LSTM,
        "gru": layers.GRU,
    }
    keras.utils.set_random_seed(42)  # same starting conditions for fairness

    model = keras.Sequential([
        keras.Input(shape=(MAX_CONTEXT,)),
        layers.Embedding(VOCAB_SIZE, embed_dim),
        rnn_layers[kind](units),
        layers.Dropout(dropout),
        layers.Dense(VOCAB_SIZE, activation="softmax"),
    ], name=kind)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="top1"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
        ],
    )
    return model

if __name__ == "__main__":
    for kind in ["simplernn", "lstm", "gru"]:
        build_model(kind).summary()