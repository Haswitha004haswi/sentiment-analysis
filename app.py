import streamlit as st
import tensorflow as tf
from utils import encode_review

# Load model
model = tf.keras.models.load_model("sentiment_lstm_model.h5")

# UI
st.title("🎬 Sentiment Analysis using LSTM")
st.write("Enter a movie review and check its sentiment")

user_input = st.text_area("Enter your review here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        encoded = encode_review(user_input)
        prediction = model.predict(encoded)

        if prediction > 0.5:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😠 Negative Sentiment")
