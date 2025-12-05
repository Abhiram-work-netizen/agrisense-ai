import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Create a synthetic dataset based on the provided soil types
soil_types = ['Black Soil', 'Cinder Soil', 'Laterite Soil', 'Peat Soil', 'Yellow Soil']
data = {
    'moisture': np.random.rand(1000) * 50,
    'ph': np.random.rand(1000) * 3 + 4,  # pH typically between 4 and 7 for these soils
    'organic_matter': np.random.rand(1000) * 10,
    'Soil_Type': np.random.choice(soil_types, 1000)
}
df = pd.DataFrame(data)

# Separate features (X) and target (y)
X = df.drop('Soil_Type', axis=1)
y = df['Soil_Type']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
with open('model2.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("\nTrained model saved as model2.pkl")