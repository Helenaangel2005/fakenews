import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = joblib.load("pac_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("📰 Fake News Detector")
st.write("Enter a news article below to check whether it's **Fake** or **Real**.")

user_input = st.text_area("News Article Text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        vect_input = vectorizer.transform([user_input])
        prediction = model.predict(vect_input)[0]
        st.success(f"The news is **{prediction}**.")
