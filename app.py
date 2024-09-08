from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import os
from flask_pymongo import PyMongo
from bson import ObjectId
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import traceback  # For logging errors

# Initialize the Flask app
app = Flask(__name__)

# Configure MongoDB
app.config["MONGO_URI"] = "mongodb://localhost:27017/accident_reports_db"
mongo = PyMongo(app)

# Accident reports collection
accident_reports = mongo.db.accident_reports

# Function to load CSV data from any file
def load_csv_data(filename):
    path = os.path.join('static', 'data', filename)
    try:
        df = pd.read_csv(path)
        print(f"CSV Columns in {filename}:", df.columns)
        return df
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return pd.DataFrame()

# Get accident reports from MongoDB
def get_mongo_data():
    try:
        mongo_reports = list(accident_reports.find({}))
        if mongo_reports:
            mongo_df = pd.DataFrame(mongo_reports)
            mongo_df['latitude'] = mongo_df['latitude'].astype(float)
            mongo_df['longitude'] = mongo_df['longitude'].astype(float)
            return mongo_df
        return pd.DataFrame()  # Return empty DataFrame if no MongoDB data
    except Exception as e:
        print(f"Error fetching MongoDB data: {str(e)}")
        return pd.DataFrame()

# Search for location in the NODE.csv and MongoDB datasets
def search_location(location_name):
    # Load NODE.csv data
    node_data = load_csv_data('NODE.csv')
    mongo_data = get_mongo_data()

    # Search in NODE.csv first, by LGA_NAME
    if 'LGA_NAME' in node_data.columns:
        node_matches = node_data[node_data['LGA_NAME'].str.contains(location_name, case=False, na=False)]
    else:
        node_matches = pd.DataFrame()

    # If we find a match in NODE.csv, return the first result
    if not node_matches.empty:
        first_match = node_matches.iloc[0]
        return first_match['LATITUDE'], first_match['LONGITUDE']

    # If no match in NODE.csv, fallback to MongoDB search by 'address' field
    mongo_matches = mongo_data[mongo_data['address'].str.contains(location_name, case=False, na=False)] if 'address' in mongo_data.columns else pd.DataFrame()

    # If we find a match in MongoDB, return the first result
    if not mongo_matches.empty:
        first_match = mongo_matches.iloc[0]
        return first_match['latitude'], first_match['longitude']

    # No match found in either dataset
    return None, None

# Train the machine learning model (Random Forest)
def train_accident_model():
    try:
        csv_data = load_csv_data('NODE.csv')  # Loading NODE.csv
        if csv_data.empty or 'LATITUDE' not in csv_data.columns or 'LONGITUDE' not in csv_data.columns:
            print("Error: LATITUDE and LONGITUDE columns not found in NODE.csv")
            return None

        mongo_data = get_mongo_data()
        if mongo_data.empty:
            print("Warning: No data found in MongoDB.")
        
        # Create dummy class labels (0 for no accident, 1 for accidents)
        combined_data = pd.concat([csv_data[['LATITUDE', 'LONGITUDE']], mongo_data[['latitude', 'longitude']]])
        combined_data['accident_occurred'] = 1  # All records are accidents

        # Add some synthetic negative examples (optional step)
        # Here, we're assuming a scenario where some points don't have accidents
        synthetic_data = pd.DataFrame({
            'LATITUDE': combined_data['LATITUDE'].sample(10).values + 0.1,
            'LONGITUDE': combined_data['LONGITUDE'].sample(10).values + 0.1,
            'accident_occurred': 0
        })
        combined_data = pd.concat([combined_data, synthetic_data], ignore_index=True)

        # Ensure the columns are named consistently
        X = combined_data[['LATITUDE', 'LONGITUDE']].rename(columns={'LATITUDE': 'latitude', 'LONGITUDE': 'longitude'})
        y = combined_data['accident_occurred']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a RandomForest classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        return model

    except Exception as e:
        print(f"Error training model: {str(e)}")
        return None

# Route to get accident prediction for a given location name
@app.route('/predict-accident', methods=['POST'])
def predict_accident():
    try:
        data = request.json
        location_name = data.get("location_name")

        if not location_name:
            return jsonify({"error": "Please provide a location name"}), 400

        latitude, longitude = search_location(location_name)

        if latitude is None or longitude is None:
            return jsonify({"error": "Could not find the location in our data"}), 400

        model = train_accident_model()
        if not model:
            return jsonify({"error": "Model training failed"}), 500

        # Predict the likelihood of an accident for the given location
        prediction = model.predict([[latitude, longitude]])
        probability = model.predict_proba([[latitude, longitude]])

        # Handle the case where only one class exists
        if len(probability[0]) > 1:
            accident_probability = probability[0][1]
        else:
            accident_probability = probability[0][0]

        return jsonify({
            "location": location_name,
            "latitude": latitude,
            "longitude": longitude,
            "accident_probability": accident_probability,
            "prediction": int(prediction[0])
        })

    except Exception as e:
        traceback.print_exc()  # Log the error
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

# Route for Crash Hotspot Identification
@app.route('/hotspots', methods=['GET', 'POST'])
def hotspots():
    atmospheric_df = load_csv_data('ATMOSPHERIC_COND.csv')
    location_df = load_csv_data('ACCIDENT_LOCATION.csv')
    node_df = load_csv_data('NODE.csv')

    missing_datasets = []
    if atmospheric_df.empty:
        missing_datasets.append('Atmospheric data')
    if location_df.empty:
        missing_datasets.append('Location data')
    if node_df.empty:
        missing_datasets.append('Node data')

    if missing_datasets:
        return f"Missing datasets: {', '.join(missing_datasets)}", 400

    try:
        df = pd.merge(pd.merge(atmospheric_df, location_df, on='ACCIDENT_NO'), node_df, on='ACCIDENT_NO', how='inner')
    except KeyError:
        return "Data merging error: Missing keys in datasets.", 400

    lga_filter = request.args.get('lga', '').lower()
    if lga_filter and 'LGA_NAME' in df.columns:
        df = df[df['LGA_NAME'].str.lower().str.contains(lga_filter, case=False, na=False)]
        if df.empty:
            return f"No accidents found for the LGA: {lga_filter}", 404

    csv_hotspot_data = df[['LATITUDE', 'LONGITUDE']].dropna().values.tolist()

    mongo_hotspot_data = []
    for report in accident_reports.find():
        if 'latitude' in report and 'longitude' in report:
            mongo_hotspot_data.append([report['latitude'], report['longitude']])

    combined_hotspot_data = csv_hotspot_data + mongo_hotspot_data
    combined_hotspot_data_json = json.dumps(combined_hotspot_data)

    return render_template('hotspots.html', hotspot_data=combined_hotspot_data_json)

# Route to handle accident report submission
@app.route('/report-incident', methods=['POST'])
def report_incident():
    try:
        data = request.json
        if not data.get("type") or not data.get("latitude") or not data.get("longitude"):
            return jsonify({"success": False, "message": "Invalid data"}), 400

        report_id = accident_reports.insert_one({
            "type": data["type"],
            "notes": data.get("notes", ""),
            "latitude": data["latitude"],
            "longitude": data["longitude"],
            "address": data.get("address", "Unknown location")
        }).inserted_id

        return jsonify({
            "success": True,
            "report_id": str(report_id),
            "address": data.get("address", "Unknown location")
        })

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"success": False, "message": "Internal Server Error"}), 500

# Route for accident reports retrieval
@app.route('/get-reports', methods=['GET'])
def get_reports():
    reports = []
    try:
        for report in accident_reports.find():
            reports.append({
                "id": str(report["_id"]),
                "type": report["type"],
                "notes": report["notes"],
                "latitude": report["latitude"],
                "longitude": report["longitude"],
                "address": report.get("address", "")
            })
    except Exception as e:
        print(f"Error fetching reports: {str(e)}")
        return jsonify({"error": "Could not fetch reports"}), 500

    return jsonify({"reports": reports})

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# Route for accident prediction page
@app.route('/prediction', methods=['GET'])
def prediction():
    return render_template('prediction.html')

# Route for report submission page
@app.route('/report')
def report():
    return render_template('report.html')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
