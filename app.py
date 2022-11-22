import streamlit as st
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
st.title("Predicting Stocks App")
st.write("Welcome to predicting stocks!")
article = st.text_input('Todays news: ')


tokenizer = AutoTokenizer.from_pretrained("icyGS/FinancePredictor")

model = AutoModelForSequenceClassification.from_pretrained("icyGS/FinancePredictor")

classifier = pipeline("text-classification", model = model, tokenizer = tokenizer)
prediction = classifier(article)
st.text(prediction)
