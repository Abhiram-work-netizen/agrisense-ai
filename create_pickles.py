import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# --- This script creates dummy .pkl files for a sample machine learning model. ---
# --- You should replace this with your own logic to generate your actual .pkl files. ---

def create_dummy_pickles():
    """
    Creates and saves a dummy scaler and a linear regression model.
    This simulates a scenario where you have a pre-trained model and a scaler.
    """
    print("Generating dummy data and training model...")

    # 1. Create some sample data
    # Let's pretend we are predicting a target value based on three input features.
    # X should be a 2D array (samples, features), and y should be a 1D array.
    X_train = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
        [25, 35, 45],
        [55, 65, 75]
    ])
    y_train = np.array([5, 12, 19, 8, 15]) # A simple linear relationship for demonstration

    # 2. Create and train the StandardScaler
    # The scaler standardizes features by removing the mean and scaling to unit variance.
    print("Training scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 3. Create and train the Linear Regression model
    print("Training model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # 4. Save the scaler to a .pkl file
    try:
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("✅ scaler.pkl created successfully.")
    except Exception as e:
        print(f"❌ Error creating scaler.pkl: {e}")


    # 5. Save the model to a .pkl file
    try:
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("✅ model.pkl created successfully.")
    except Exception as e:
        print(f"❌ Error creating model.pkl: {e}")

if __name__ == '__main__':
    create_dummy_pickles()
