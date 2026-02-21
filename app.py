import streamlit as st
import pandas as pd
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detection App")
st.write("Enter a news article or headline, and we'll predict whether it's Real or Fake.")

user_input = st.text_area("Paste your article or headline here:")

if st.button("Predict"):
    if user_input.strip() != "":
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        confidence = max(model.predict_proba(input_vector)[0]) * 100

        if prediction == 1:
            st.error(f"🚨 Prediction: Fake News ({confidence:.2f}% confidence)")
        else:
            st.success(f"✅ Prediction: Real News ({confidence:.2f}% confidence)")
    else:
        st.warning("Please enter some text to analyze.")
