# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import json


# Load dataset (user should prepare `data/disease_data.csv`)
# CSV format example:
# fever,cough,fatigue,headache,nausea,disease
# 1,1,0,0,0,Flu


def main():
    df = pd.read_csv('data/disease_data.csv')
    # assume last column is 'disease'
    X = df.drop(columns=['disease'])
    y = df['disease']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)


    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))


    # Save model and feature names
    with open('model/model.pkl', 'wb') as f:
        pickle.dump({'model': clf, 'features': list(X.columns)}, f)


    # Optionally save a disease -> suggestion mapping template
    suggestions = {
    'Flu': 'Rest, stay hydrated, paracetamol for fever, see a doctor if breathless or symptoms worsen.',
    'Common Cold': 'Rest, fluids, warm fluids, decongestant if needed, consult if symptoms last >10 days.',
    'Gastritis': 'Avoid spicy/fried food, small meals, antacids; consult for prescription if severe.',
    'Migraine': 'Dark room, NSAID (if safe), identify triggers; seek medical help if severe.',
    'COVID-19': 'Isolate, rest, fluids, monitor oxygen; seek care for breathing difficulty.'
    }
    with open('model/suggestions.json', 'w') as f:
        json.dump(suggestions, f, indent=2)


main()