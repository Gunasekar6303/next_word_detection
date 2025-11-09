import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model
model = load_model("next_word_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

sequence_length = 20

def predict_next_word(model, tokenizer, text):
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])[0]

    seq = seq[-(sequence_length - 1):]
    seq = pad_sequences([seq], maxlen=sequence_length - 1, padding='pre')

    pred = model.predict(seq, verbose=0)[0]
    next_word_id = np.argmax(pred)

    for word, index in tokenizer.word_index.items():
        if index == next_word_id:
            return word

    return None

# Testing
print(predict_next_word(model, tokenizer, "the united states"))
print(predict_next_word(model, tokenizer, "the cat sat on"))
print(predict_next_word(model, tokenizer, "during the second world"))
