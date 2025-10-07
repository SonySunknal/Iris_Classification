# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

st.set_page_config(page_title="Iris Classifier", layout="centered")

@st.cache_resource
def load_model(path="models/iris_pipeline.joblib"):
    if not os.path.exists(path):
        raise FileNotFoundError("Model file not found: " + path)
    return joblib.load(path)

model = load_model()

st.title("Iris Species Prediction")
st.write("Enter measurements to predict the iris species.")

sepal_length = st.number_input("Sepal length (cm)", 0.0, 10.0, 5.0)
sepal_width  = st.number_input("Sepal width (cm)", 0.0, 10.0, 3.0)
petal_length = st.number_input("Petal length (cm)", 0.0, 10.0, 1.4)
petal_width  = st.number_input("Petal width (cm)", 0.0, 10.0, 0.2)

input_df = pd.DataFrame([{
    "sepal_length": sepal_length,
    "sepal_width": sepal_width,
    "petal_length": petal_length,
    "petal_width": petal_width
}])

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    st.success(f"Predicted species: **{pred}**")
    st.write("Probabilities:", dict(zip(model.classes_, [round(float(p),3) for p in proba])))
