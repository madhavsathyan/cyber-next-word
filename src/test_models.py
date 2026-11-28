import time
import numpy as np
from models import build_model, VOCAB_SIZE

X = np.load("data/processed/X_train.npy")[:50000]
y = np.load("data/processed/y_train.npy")[:50000]
Xv = np.load("data/processed/X_val.npy")[:10000]
yv = np.load("data/processed/y_val.npy")[:10000]

print("Expected starting loss for random guessing: ln(5000) =", round(float(np.log(VOCAB_SIZE)), 2))

for kind in ["simplernn", "lstm", "gru"]:
    print(f"\n===== {kind.upper()} =====")
    model = build_model(kind)
    print("Parameters:", f"{model.count_params():,}")
    t = time.time()
    hist = model.fit(X, y, validation_data=(Xv, yv),
                     epochs=2, batch_size=256, verbose=1)
    print(f"Time: {time.time() - t:.1f}s | "
          f"val_loss={hist.history['val_loss'][-1]:.2f} | "
          f"val_top1={hist.history['val_top1'][-1]:.3f} | "
          f"val_top5={hist.history['val_top5'][-1]:.3f}")