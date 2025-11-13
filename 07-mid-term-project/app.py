from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("Model.pkl", "rb"))

# Load preprocessed car data for populating dropdowns
car_data = pd.read_csv("processed_data.csv")

# Prepare dropdown lists safely
brands = sorted(car_data['brand'].astype(str).unique())
years = sorted(car_data['make_year'].astype(int).unique(), reverse=True)
transmission = sorted(car_data['transmission'].astype(str).unique())
fuel_types = sorted(car_data['fuel_type'].astype(str).unique())
engine_ccs = sorted(car_data['engine_cc'].astype(int).unique())
owner_counts = sorted(car_data['owner_count'].astype(int).unique())
insurance_options = sorted(car_data['insurance_valid'].astype(str).unique())
colors = sorted(car_data['color'].astype(str).unique())
service_history_options = sorted(car_data['service_history'].astype(str).unique())
accidents = sorted(car_data['accidents_reported'].astype(str).unique())
mileages = sorted(car_data['mileage_kmpl'].astype(float).unique())

@app.route("/")
def index():
    return render_template(
        "index.html",
        brands=brands,
        years=years,
        transmission=transmission,
        fuel_types=fuel_types,
        engine_ccs=engine_ccs,
        owner_counts=owner_counts,
        insurance_options=insurance_options,
        colors=colors,
        service_history_options=service_history_options,
        accidents=accidents,
        mileages=mileages
    )

@app.route("/predict", methods=["POST"])
def predict():
    # Extract form data
    form = request.form
    X = pd.DataFrame([{
        'brand': form['brand'],
        'make_year': int(form['year']),
        'transmission': form['transmission'],
        'fuel_type': form['fuel'],
        'engine_cc': float(form['engine_cc']),
        'owner_count': int(form['owner_count']),
        'insurance_valid': form['insurance_valid'],
        'color': form['color'],
        'service_history': form['service_history'],
        'accidents_reported': int(form['accidents_reported']),
        'mileage_kmpl': float(form['mileage_kmpl'])
    }])
    # Make prediction
    pred = model.predict(X)
    print(pred[0])
    return str(round(pred[0], 2))

if __name__ == "__main__":
    app.run(debug=True)