import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load the dataset
try:
    df = pd.read_csv('district wise rainfall normal.csv')
except FileNotFoundError:
    print("Error: 'district wise rainfall normal.csv' not found.")
    print("Please make sure the dataset file is in the same directory as the script.")
    exit()

# Drop rows with missing values for simplicity
df.dropna(inplace=True)

# Select features (monthly rainfall) and target (annual rainfall)
features = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
target = 'ANNUAL'

X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) Score: {r2}")

# Save the trained model
with open('model3.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("\nTrained model saved as model3.pkl")