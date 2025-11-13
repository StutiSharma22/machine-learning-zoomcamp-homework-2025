import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open("Model.pkl", "rb"))

# Load processed data for dropdowns
car = pd.read_csv("processed_data.csv")

st.set_page_config(page_title="Used Car Price Predictor", layout="centered")

# Title with navy blue background
st.markdown(
    """
    <div style="background-color:#001f5b;padding:15px;border-radius:5px">
        <h1 style="color:white;text-align:center;">Used Car Price Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("##")  # spacing

# Dropdowns for inputs
brand = st.selectbox("Select Car Brand", sorted(car['brand'].unique()))
year = st.selectbox("Select Year", sorted(car['make_year'].unique(), reverse=True))
transmission = st.selectbox("Select Transmission Type", sorted(car['transmission'].unique()))
fuel_type = st.selectbox("Select Fuel Type", sorted(car['fuel_type'].unique()))
engine_cc = st.selectbox("Engine CC", sorted(car['engine_cc'].unique()))
owner_count = st.selectbox("Owner Count", sorted(car['owner_count'].unique()))
car_age = st.selectbox("Car Age", sorted(car['car_age'].unique()))
insurance_valid = st.selectbox("Insurance Valid", ["Yes", "No"])
color = st.selectbox("Color", sorted(car['color'].unique()))
service_history = st.selectbox("Service History", ["Full", "Partial", "None"])
accidents_reported = st.selectbox("Accidents Reported", sorted(car['accidents_reported'].unique()))
mileage_kmpl = st.selectbox("Mileage (kmpl)", sorted(car['mileage_kmpl'].unique()))

st.write("##")  # spacing

# Predict button
if st.button("Predict Price"):
    # Prepare input for model
    X = [[brand, year, transmission, fuel_type, engine_cc,
          owner_count, car_age, insurance_valid, color,
          service_history, accidents_reported, mileage_kmpl]]
    
    # Predict
    prediction = model.predict(X)
    
    st.success(f"Predicted Price: USD {np.round(prediction[0], 2)}")