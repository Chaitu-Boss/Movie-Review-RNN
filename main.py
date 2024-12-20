import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

model=load_model('rnn_model.h5')

word_indexes=imdb.get_word_index()
reverse_word_index=dict([(value,key) for (key,value) in word_indexes.items()])

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_indexes.get(word, 2) + 3 for word in words]
    encoded_review = [min(index, 9999) for index in encoded_review]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def sentiment(value):
    if value<0.2:
        return "Very negative"
    elif value<0.4:
        return "Negative"
    elif value<0.6:
        return "Neutral"
    elif value<0.8:
        return "Positive"
    else:
        return "Very positive"
    
def predict_sentiment(review):
    preprocessed_test=preprocess_text(review)
    prediction=model.predict(preprocessed_test)
    user_sentiment=sentiment(prediction[0][0])
    return user_sentiment,prediction[0][0]

import streamlit as st

st.title("Sentiment Analysis of Movie Reviews")

input_text = st.text_area("Enter your review")

if st.button("Predict"):
    user_sentiment, sentiment_value = predict_sentiment(input_text)
    st.write(f"Sentiment: {user_sentiment}")
    st.write(f"Prediction Score: {sentiment_value}")
else:
    st.write("Please enter a review.")
