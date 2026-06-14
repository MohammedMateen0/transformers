import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

tokenizer=AutoTokenizer.from_pretrained(
    "day5_distilbert_sentiment/saved_model"
)
model=AutoModelForSequenceClassification.from_pretrained(
    "day5_distilbert_sentiment/saved_model"
)

st.title(
    "Financial Sentiment Analysis"
)
text=st.text_area(
    "Enter financial news"
)

if st.button("Predict"):
    inputs=tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs=model(**inputs)
    probs=torch.softmax(
        outputs.logits,
        dim=1
    )
    confidence = torch.max(probs).item()
    pred=torch.argmax(
        probs,
        dim=1
    ).item()

    labels={
        0:"Negative",
        1:"Neutral",
        2:"Positive"
    }
    st.write(
        f"Prediction: {labels[pred]}"
       
    )
    st.write(
         f"Confidence: {confidence}"
       
    )