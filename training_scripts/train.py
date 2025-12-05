import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
try:
    data = pd.read_csv('weather_forecast_data.csv')
except FileNotFoundError:
    print("Error: 'weather_forecast_data.csv' not found. Make sure the dataset is in the same directory.")
    exit()

# Handle potential non-numeric data if any and encode the target variable
le = LabelEncoder()
# Assuming 'Rain' is the target. It's good practice to handle the case where it might already be numeric
if data['Rain'].dtype == 'object':
    data['Rain'] = le.fit_transform(data['Rain'])

# Define features (X) and target (y)
X = data.drop('Rain', axis=1)
y = data['Rain']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n")
# Get string representation of classification_report
# We need to get the class names from the label encoder for a readable report
if 'le' in locals() and hasattr(le, 'classes_'):
    report = classification_report(y_test, y_pred, target_names=le.classes_)
else:
    report = classification_report(y_test, y_pred)
print(report)


# Save the trained model and the label encoder
joblib.dump(model, 'weather_model.pkl')
if 'le' in locals():
    joblib.dump(le, 'label_encoder.pkl')
    print("Label encoder saved as label_encoder.pkl")

print("\nModel trained and saved as weather_model.pkl")