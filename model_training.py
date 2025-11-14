import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import re

df = pd.read_csv('backend/merged_data.csv')
print(df.head())
print(df.dtypes)

# Split blood_pressure into systolic and diastolic
df['systolic_bp'] = df['blood_pressure'].str.split('/').str[0].astype(float)
df['diastolic_bp'] = df['blood_pressure'].str.split('/').str[1].astype(float)

targets = ['disease', 'medication_name', 'dosage']
features = ['age', 'height', 'weight', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature']
df_encoded = df[features].copy()

# Create models directory if it doesn't exist
models_dir = 'backend/models'
os.makedirs(models_dir, exist_ok=True)

def sanitize_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

for target in targets:
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.2, random_state=42)
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Classification report for {model_name} on target '{target}':")
        print(classification_report(y_test, y_pred))
        
        filename = f"{sanitize_filename(model_name)}_{sanitize_filename(target)}.joblib"
        filepath = os.path.join(models_dir, filename)
        joblib.dump(model, filepath)
