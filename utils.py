from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

word_index = imdb.get_word_index()
max_length = 200

def encode_review(text):
    words = text_to_word_sequence(text)
    encoded = [word_index.get(word, 2) for word in words]
    padded = pad_sequences([encoded], maxlen=max_length)
    return padded
