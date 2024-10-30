import streamlit as st
import pickle
import string
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def text_transform(text):
    text = text.lower()
    text = text.split()
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)  # Return as a single string for vectorization

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

if os.path.exists('model.pkl'):
    model = pickle.load(open('model.pkl', 'rb'))
else:
    st.error("Model file not found.")

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict') and input_sms:
    # Preprocessing
    transform_sms = text_transform(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transform_sms])

    # Predict
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
