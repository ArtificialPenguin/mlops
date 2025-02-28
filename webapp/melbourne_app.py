import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model

# Load the best trained model
model_path = r"C:\Users\jysh2\OneDrive\Documents\code\mlops_repo\models\sriram_best_model.pkl"
best_model = load_model(model_path)

# Function to make predictions using the best model
def predict_price(input_data):
    expected_features = best_model.feature_names_in_
    
    # Create full input DataFrame with expected features
    input_data_full = pd.DataFrame(columns=expected_features)
    
    # Assign known input values
    for col in input_data.columns:
        if col in expected_features:
            input_data_full[col] = input_data[col]
    
    # Convert categorical columns to string type to avoid numerical conversion issues
    for col in input_data_full.columns:
        if input_data_full[col].dtype == 'object':
            input_data_full[col] = input_data_full[col].astype(str).fillna("Unknown")
    
    # Fill missing numerical columns with median values
    for col in input_data_full.columns:
        if pd.api.types.is_numeric_dtype(input_data_full[col]):
            input_data_full[col].fillna(input_data_full[col].median(), inplace=True)
    
    print("Final Input Data for Model:", input_data_full.head())  # Debugging Step
    
    # Make prediction
    prediction = predict_model(best_model, data=input_data_full).iloc[0, -1]
    return max(0, prediction)  # Ensure no negative predictions

# Streamlit App
st.title("🏡 Melbourne Property Price Prediction")
st.markdown("Enter property details below to predict the price.")

# User Input Form
with st.form("prediction_form"):
    rooms = st.number_input("Number of Rooms", min_value=1, max_value=10, value=3)
    distance = st.number_input("Distance from CBD (km)", min_value=0.0, max_value=50.0, value=5.0)
    building_area = st.number_input("Building Area (sqm)", min_value=10.0, max_value=1000.0, value=150.0)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
    submit_button = st.form_submit_button("Predict Price")

# Handle Prediction
if submit_button:
    expected_features = best_model.feature_names_in_
    input_data = pd.DataFrame(columns=expected_features)
    
    input_data.at[0, "Rooms"] = rooms
    input_data.at[0, "Distance"] = distance
    input_data.at[0, "BuildingArea"] = building_area
    input_data.at[0, "YearBuilt"] = year_built
    
    # Convert categorical columns to string type and fill missing values with "Unknown"
    for col in input_data.columns:
        if input_data[col].dtype == 'object':
            input_data[col] = input_data[col].astype(str).fillna("Unknown")
    
    # Fill missing numerical columns with median values
    for col in input_data.columns:
        if pd.api.types.is_numeric_dtype(input_data[col]):
            input_data[col].fillna(input_data[col].median(), inplace=True)
    
    print("Final User Input for Prediction:", input_data)  # Debugging Step
    
    prediction = predict_price(input_data)
    st.success(f"🏠 Estimated Property Price: **${prediction:,.2f}**")