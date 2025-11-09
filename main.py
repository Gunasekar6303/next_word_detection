from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
import numpy as np

# Load dataset
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
train_text = dataset["train"]["text"]
corpus = " ".join(train_text)

# Slice text BEFORE tokenizing
raw_words = corpus.split()
raw_words = raw_words[:100000]
small_corpus = " ".join(raw_words)

# Tokenize only on small corpus
tokenizer = Tokenizer()
tokenizer.fit_on_texts([small_corpus])
tokens = tokenizer.texts_to_sequences([small_corpus])[0]

# Save tokenizer
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Vocabulary
vocab_size = len(tokenizer.word_index) + 1
print("vocab_size =", vocab_size)

# Build sequences
sequence_length = 20
sequences = []
for i in range(sequence_length, len(tokens)):
    sequences.append(tokens[i-sequence_length:i])

sequences = np.array(sequences)
print("shape:", sequences.shape)

# Split X, y
X = sequences[:, :-1]
y = sequences[:, -1]

# Model
model = Sequential()
model.add(Embedding(vocab_size, 256, input_length=sequence_length - 1))
model.add(GRU(512))
model.add(Dense(vocab_size, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

model.summary()

# Train
model.fit(X, y, epochs=5, batch_size=256)

# Save model
model.save("next_word_model.h5")

print("Training completed. Model and tokenizer saved.")
