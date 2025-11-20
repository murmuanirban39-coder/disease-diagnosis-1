# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np
import json
import mysql.connector
from datetime import datetime
import os


app = Flask(__name__)


# Load model and feature list
model_path = 'model/model.pkl'
with open(model_path, 'rb') as f:
    saved = pickle.load(f)
model = saved['model']
features = saved['features']


with open('model/suggestions.json', 'r') as f:
    suggestions_map = json.load(f)


# MySQL config - change to your DB credentials
DB_CONFIG = {
'host': 'localhost',
'user': 'root',
'password': 'Anirban123',
'database': 'disease_db'
}




def save_to_db(patient_name, age, gender, symptoms_dict, predicted, confidence):
# symptoms_dict is a mapping of feature->0/1
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        insert_sql = (
        "INSERT INTO patients (name, age, gender, symptoms_json, predicted_disease, confidence, created_at)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s)"
        )
        cursor.execute(insert_sql, (
        patient_name,
        age,
        gender,
        json.dumps(symptoms_dict),
        predicted,
        float(confidence),
        datetime.utcnow()
        ))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print('DB save error:', e)




@app.route('/')
def index():
# Render checkboxes for each feature
    return render_template('index.html', features=features)


@app.route('/predict', methods=['POST'])
def predict():
    form = request.form
    
    name = form.get('name', 'Unknown')
    age = form.get('age', None)
    gender = form.get('gender', '')

    # Build feature vector
    x = []
    symptoms_dict = {}

    for feat in features:
        value = 1 if form.get(feat) == 'on' else 0
        x.append(value)
        symptoms_dict[feat] = value

    arr = np.array(x).reshape(1, -1)

    # Predict disease
    pred = model.predict(arr)[0]

    # Predict confidence
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(arr)
        class_index = list(model.classes_).index(pred)
        confidence = float(probs[0][class_index])
    else:
        confidence = 0.0

    # Save to DB
    save_to_db(name, age, gender, symptoms_dict, pred, confidence)

    # Suggestions
    suggestion = suggestions_map.get(pred, "No suggestion found")

    return render_template(
        'result.html',
        name=name,
        predicted=pred,
        confidence=round(confidence, 3),
        suggestion=suggestion
    )

app.run(debug=True, host="0.0.0.0", port=5000)

