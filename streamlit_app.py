import streamlit as st
import pickle

# load model + vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Twitter Sentiment Analysis")

user_input = st.text_area("Enter a tweet:")

if st.button("Predict Sentiment"):
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)

    if prediction[0] == 1:
        st.success("Positive 😊")
    else:
        st.error("Negative 😡")
