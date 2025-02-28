from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

# Load the used car sales model
car_model = load_model(r'C:\Users\jysh2\OneDrive\Documents\code\mlops_repo\models\used_car_sales')

# Load the Melbourne property price prediction model
property_model_path = r"C:\Users\jysh2\OneDrive\Documents\code\mlops_repo\models\sriram_best_model"
property_model = load_model(property_model_path)

def predict_car_price(model, input_df):
    cols_to_log = ["Kilometers_Driven", "Engine (CC)", "Power (bhp)"]
    input_df[cols_to_log] = input_df[cols_to_log].apply(lambda x: np.log1p(x))
    predictions_df = predict_model(estimator=model, data=input_df)
    predicted_value = predictions_df['prediction_label'][0]
    return np.expm1(predicted_value)

def predict_property_price(input_data):
    expected_features = property_model.feature_names_in_
    input_data_full = pd.DataFrame(columns=expected_features)
    for col in input_data.columns:
        if col in expected_features:
            input_data_full[col] = input_data[col]
    for col in input_data_full.columns:
        if input_data_full[col].dtype == 'object':
            input_data_full[col] = input_data_full[col].astype(str).fillna("Unknown")
    for col in input_data_full.columns:
        if pd.api.types.is_numeric_dtype(input_data_full[col]):
            input_data_full[col].fillna(input_data_full[col].median(), inplace=True)
    prediction = predict_model(property_model, data=input_data_full).iloc[0, -1]
    return max(0, prediction)

def run():
    add_selectbox = st.sidebar.selectbox(
        "What would you like to predict?",
        ("Used Car Prices", "Housing Prices")
    )
    
    if add_selectbox == 'Used Car Prices':
        st.sidebar.info('This app predicts used car prices')
        st.title("Used Car Price Prediction App")

        brand = st.selectbox("Brand", ['Maruti', 'Hyundai', 'Honda', 'Audi', 'Nissan', 'Toyota',
           'Volkswagen', 'Tata', 'Land', 'Mitsubishi', 'Renault',
           'Mercedes-Benz', 'BMW', 'Mahindra', 'Ford', 'Porsche', 'Datsun',
           'Jaguar', 'Volvo', 'Chevrolet', 'Skoda', 'Mini', 'Fiat', 'Jeep',
           'Smart', 'Ambassador', 'Isuzu', 'Force', 'Bentley',
           'Lamborghini'])
        model_name = st.text_input("Model Name", "Creta 1.6 CRDi SX Option")
        location = st.selectbox("Location", ['Mumbai', 'Pune', 'Chennai', 'Coimbatore', 'Hyderabad', 'Jaipur', 'Kochi', 'Kolkata', 'Delhi', 'Bangalore', 'Ahmedabad'])
        year = st.slider("Year of Manufacture", 2000, 2025, 2015)
        kilometers_driven = st.number_input("Kilometers Driven", 0, 300000, 30000)
        fuel_type = st.selectbox("Fuel Type", ["CNG", "Diesel", "Petrol", "LPG"])
        transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
        owner_type = st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth & Above"])
        mileage = st.number_input("Mileage (kmpl)", 5.0, 60.0, 18.0)
        engine = st.number_input("Engine Capacity (CC)", 50, 6000, 1500)
        power = st.number_input("Power (bhp)", 30.0, 1000.0, 100.0)
        seats = st.selectbox("Seats", list(range(1, 11)))

        input_df = pd.DataFrame([{ "Brand": brand, "Model": model_name, "Location": location,
            "Year": year, "Kilometers_Driven": kilometers_driven, "Fuel_Type": fuel_type,
            "Transmission": transmission, "Owner_Type": owner_type, "Mileage (kmpl)": mileage,
            "Engine (CC)": engine, "Power (bhp)": power, "Seats": seats }])

        if st.button("Predict"):
            output = predict_car_price(car_model, input_df)
            st.success(f'The estimated car price is: ₹{output:,.2f}')
    
    if add_selectbox == 'Housing Prices':
        st.title("🏡 Melbourne Property Price Prediction")
        st.markdown("Enter property details below to predict the price.")
        
        with st.form("prediction_form"):
            rooms = st.number_input("Number of Rooms", 1, 10, 3)
            distance = st.number_input("Distance from CBD (km)", 0.0, 50.0, 5.0)
            building_area = st.number_input("Building Area (sqm)", 10.0, 1000.0, 150.0)
            year_built = st.number_input("Year Built", 1800, 2025, 2000)
            submit_button = st.form_submit_button("Predict Price")

        if submit_button:
            expected_features = property_model.feature_names_in_
            input_data = pd.DataFrame(columns=expected_features)
            input_data.at[0, "Rooms"] = rooms
            input_data.at[0, "Distance"] = distance
            input_data.at[0, "BuildingArea"] = building_area
            input_data.at[0, "YearBuilt"] = year_built
            prediction = predict_property_price(input_data)
            st.success(f'🏠 Estimated Property Price: **${prediction:,.2f}**')

if __name__ == '__main__':
    run()
