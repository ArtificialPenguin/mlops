from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import os

# Base directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define relative paths to the models folder
car_model_path = os.path.join(base_dir, "models", "used_car_sales")
best_model_path = os.path.join(base_dir, "models", "sriram_best_model")

# Load the models
car_model = load_model(car_model_path)
best_model = load_model(best_model_path)

def predict(model, input_df):
    # Apply log transformation to match training preprocessing
    cols_to_log = ["Kilometers_Driven", "Engine (CC)", "Power (bhp)"]
    input_df[cols_to_log] = input_df[cols_to_log].apply(lambda x: np.log1p(x))  # Apply log1p

    # PyCaret applies its preprocessing pipeline automatically
    predictions_df = predict_model(estimator=model, data=input_df)

    # Extract the transformed prediction value
    predicted_value = predictions_df['prediction_label'][0]

    # Reverse the target transformation
    original_prediction = np.expm1(predicted_value)

    return original_prediction

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


def run():
    add_selectbox = st.sidebar.selectbox(
        "What would you like to predict?",
        ("Used Car Prices", "Property Price Prediction")
    )

    st.sidebar.info('This app is created to predict used car prices')
    st.title("Used Car Price Prediction App")

    if add_selectbox == 'Used Car Prices':
        brand = st.selectbox("Brand", ['Maruti', 'Hyundai', 'Honda', 'Audi', 'Nissan', 'Toyota',
       'Volkswagen', 'Tata', 'Land', 'Mitsubishi', 'Renault',
       'Mercedes-Benz', 'BMW', 'Mahindra', 'Ford', 'Porsche', 'Datsun',
       'Jaguar', 'Volvo', 'Chevrolet', 'Skoda', 'Mini', 'Fiat', 'Jeep',
       'Smart', 'Ambassador', 'Isuzu', 'Force', 'Bentley',
       'Lamborghini'])

        model_name = st.text_input("e.g. Model Name", "Creta 1.6 CRDi SX Option")
        location = st.selectbox("Location", ['Mumbai', 'Pune', 'Chennai', 'Coimbatore', 'Hyderabad', 'Jaipur', 'Kochi', 'Kolkata', 'Delhi', 'Bangalore', 'Ahmedabad'])
        year = st.slider("Year of Manufacture", min_value=2000, max_value=2025, value=2015)
        kilometers_driven = st.number_input("Kilometers Driven", min_value=0, max_value=300000, value=30000)
        fuel_type = st.selectbox("Fuel Type", ["CNG", "Diesel", "Petrol", "LPG"])
        transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
        owner_type = st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth & Above"])
        mileage = st.number_input("Mileage (in kmpl)", min_value=5.0, max_value=60.0, value=18.0)
        engine = st.number_input("Engine Capacity (CC)", min_value=50, max_value=6000, value=1500)
        power = st.number_input("Power (bhp)", min_value=30.0, max_value=1000.0, value=100.0)
        seats = st.selectbox("Seats", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        output = ""

        # Create input DataFrame
        input_df = pd.DataFrame([{
        "Brand": brand,
        "Model": model_name,  
        "Location": location,
        "Year": year,
        "Kilometers_Driven": kilometers_driven,
        "Fuel_Type": fuel_type,
        "Transmission": transmission,
        "Owner_Type": owner_type,
        "Mileage (kmpl)": mileage,  
        "Engine (CC)": engine,      
        "Power (bhp)": power,   
        "Seats": seats
    }])

        
        if st.button("Predict"):
            output = predict(model=car_model, input_df=input_df)
            st.success(f'The estimated car price is: ₹{output:,.2f}')
    
    if add_selectbox == 'Property Price Prediction':
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

# Run the Streamlit app
if __name__ == '__main__':
    run()