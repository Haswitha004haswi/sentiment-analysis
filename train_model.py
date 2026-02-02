# train_model.py
# Step 1: Import Libraries
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import text_to_word_sequence

# Step 2: Set Parameters
vocab_size = 10000      # top 10k words
max_length = 200        # max review length

# Step 3: Load IMDB Dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Step 4: Pad Sequences
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

# Step 5: Build LSTM Model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

# Step 6: Compile Model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Step 7: Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# Step 8: Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Step 9: Save Model
model.save("sentiment_lstm_model.h5")
print("Model saved as sentiment_lstm_model.h5")

# ---------------------------
# Step 10: Test Custom Review
# ---------------------------
word_index = imdb.get_word_index()

def encode_review(text):
    words = text_to_word_sequence(text)
    encoded = [word_index.get(word, 2) for word in words]  # 2 for unknown words
    padded = pad_sequences([encoded], maxlen=max_length)
    return padded

# Example review
review = "The movie was not good"
prediction = model.predict(encode_review(review))
print(f"Review: {review}")
print("Sentiment:", "Positive" if prediction > 0.5 else "Negative")
