import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import svm_model, knn_model, rf_model, lstm_model, rnn_model, scaler

# App Title
st.title("Electricity Bill Prediction")

# Sidebar for model selection
model_choice = st.sidebar.selectbox("Choose Model", ["SVM", "KNN", "Random Forest", "LSTM", "RNN"])

# Input Fields
st.subheader("Enter Customer Details")
city = st.text_input("City Name")
company = st.text_input("Company Name")
monthly_units = st.number_input("Monthly Units Consumed", min_value=0)
previous_bill = st.number_input("Previous Month's Bill", min_value=0)
peak_hours = st.number_input("Peak Hours Consumption", min_value=0)

# Convert inputs to dataframe
input_data = pd.DataFrame({
    "City": [city],
    "Company": [company],
    "MonthlyUnits": [monthly_units],
    "PreviousBill": [previous_bill],
    "PeakHours": [peak_hours]
})

# Load Encoder & Scale Input
input_data_scaled = scaler.transform(input_data)

# Prediction Function
def predict_bill(model_name, input_data):
    if model_name == "SVM":
        return svm_model.predict(input_data)[0]
    elif model_name == "KNN":
        return knn_model.predict(input_data)[0]
    elif model_name == "Random Forest":
        return rf_model.predict(input_data)[0]
    elif model_name == "LSTM":
        return lstm_model.predict(np.reshape(input_data, (1, 1, -1)))[0][0]
    elif model_name == "RNN":
        return rnn_model.predict(np.reshape(input_data, (1, 1, -1)))[0][0]

# Predict Button
if st.button("Predict"):
    prediction = predict_bill(model_choice, input_data_scaled)
    st.success(f"Predicted Electricity Bill: ₹{prediction:.2f}")
