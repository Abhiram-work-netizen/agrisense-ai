import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Setup Paths
# Get the directory of this script (training_scripts) and go up one level to Project Root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Crop_recommendation.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"🔄 Loading data from: {DATA_PATH}")

# 2. Load Data
try:
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data loaded! Shape: {df.shape}")
except FileNotFoundError:
    print(f"❌ Error: 'Crop_recommendation.csv' not found at {DATA_PATH}")
    print("Please ensure the 'data' folder is in the project root.")
    exit()

# 3. Prepare Data (Inputs: 7 features)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# 4. Convert Labels (Rice -> 1, Maize -> 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 5. Train Model
print("🧠 Training AI Model (Random Forest)...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the Brains
pickle.dump(model, open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb'))
pickle.dump(le, open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb'))

print(f"💾 Success! New 7-feature model saved to '{MODEL_DIR}'")