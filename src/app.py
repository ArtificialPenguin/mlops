from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

# header/layout
model = load_model('used_car_sales_pipeline')

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

def run():
    add_selectbox = st.sidebar.selectbox(
        "What would you like to predict?",
        ("Used Car Prices", "Sriram stuff")
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
            output = predict(model=model, input_df=input_df)
            st.success(f'The estimated car price is: ₹{output:,.2f}')
    
    if add_selectbox == 'Sriram stuff':
        st.write("Hello")

# Run the Streamlit app
if __name__ == '__main__':
    run()