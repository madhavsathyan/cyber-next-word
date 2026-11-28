import re, os, json, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

VOCAB_SIZE = 5000
MAX_CONTEXT = 20
SEED = 42

os.makedirs("data/processed", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 1. Load corpus and make full stops their own token
lines = open("data/corpus.txt", encoding="utf-8").read().split("\n")
lines = [re.sub(r"\s+", " ", l.replace(".", " . ")).strip() for l in lines]
lines = [l for l in lines if l]

# 2. Split by description (before building sequences)
train_lines, val_lines = train_test_split(lines, test_size=0.1, random_state=SEED)
print("Train descriptions:", len(train_lines), "| Val descriptions:", len(val_lines))

# 3. Tokenizer: learn the vocabulary from TRAIN text only
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>", filters="", lower=True)
tokenizer.fit_on_texts(train_lines)

# 4. Build (context -> next word) examples
def make_sequences(text_lines):
    X, y = [], []
    for seq in tokenizer.texts_to_sequences(text_lines):
        for i in range(1, len(seq)):
            X.append(seq[max(0, i - MAX_CONTEXT):i])
            y.append(seq[i])
    X = pad_sequences(X, maxlen=MAX_CONTEXT, padding="pre")
    return X.astype("int32"), np.array(y, dtype="int32")

X_train, y_train = make_sequences(train_lines)
X_val, y_val = make_sequences(val_lines)
print("X_train:", X_train.shape, "| y_train:", y_train.shape)
print("X_val:  ", X_val.shape, "| y_val:  ", y_val.shape)

# 5. Vocabulary coverage
counts = sorted(tokenizer.word_counts.values(), reverse=True)
coverage = sum(counts[:VOCAB_SIZE - 2]) / sum(counts)
print(f"Top {VOCAB_SIZE} words cover {coverage:.1%} of all words")

# 6. Sanity check: decode one example back to words
rev = {i: w for w, i in tokenizer.word_index.items()}
k = 100
print("Context:", " ".join(rev[i] for i in X_train[k] if i != 0))
print("Next word:", rev[y_train[k]])

# 7. Word-frequency chart
top = sorted(tokenizer.word_counts.items(), key=lambda x: -x[1])[:20]
plt.figure(figsize=(10, 5))
plt.bar([w for w, _ in top], [c for _, c in top])
plt.xticks(rotation=60, ha="right")
plt.title("Top 20 most frequent words")
plt.tight_layout()
plt.savefig("results/word_frequency.png", dpi=150)

# 8. Save everything for Day 3 and the Streamlit app
np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/X_val.npy", X_val)
np.save("data/processed/y_val.npy", y_val)
pickle.dump(tokenizer, open("artifacts/tokenizer.pkl", "wb"))
json.dump({"vocab_size": VOCAB_SIZE, "max_context": MAX_CONTEXT},
          open("artifacts/config.json", "w"))
print("Saved.")