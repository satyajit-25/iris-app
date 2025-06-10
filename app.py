import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load("logistic_regression_iris.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("🌸 Iris Flower Species Prediction")
st.write("Enter the flower measurements below to predict the species:")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width  = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.35)
petal_width  = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

# Prepare input
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    st.success(f"🌼 The predicted Iris species is: **{species_map[prediction]}**")
