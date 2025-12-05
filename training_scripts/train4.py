import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
from datetime import datetime, timedelta

# --- Step 1: Generate Synthetic Data (Replace with pd.read_csv in real use) ---
# In a real scenario, you would load your CSVs like this:
# df_w_food = pd.read_csv('Weekly_Food_Retail_Prices.csv')
# df_w_non_food = pd.read_csv('Weekly_Non_Food_Retail_Prices.csv')
# ... and so on.

def create_synthetic_data(start_date, periods, freq, item_type, num_items=50):
    """Creates a sample retail price dataframe."""
    dates = pd.to_datetime(pd.date_range(start=start_date, periods=periods, freq=freq))
    df_data = []
    for item_id in range(num_items):
        # Create a base price and some random fluctuations
        base_price = np.random.rand() * 100 + 5
        prices = base_price + np.random.randn(len(dates)) * 2 + np.sin(np.arange(len(dates)))
        for i, date in enumerate(dates):
            df_data.append([date, f'{item_type}_{item_id}', prices[i], item_type])
    
    df = pd.DataFrame(df_data, columns=['Date', 'ItemID', 'Price', 'Type'])
    df['Price'] = df['Price'].round(2)
    return df

# Generate four sample dataframes
df_w_food = create_synthetic_data('2023-01-01', 52, 'W', 'Food')
df_w_non_food = create_synthetic_data('2023-01-01', 52, 'W', 'Non-Food')
df_m_food = create_synthetic_data('2023-01-01', 12, 'M', 'Food')
df_m_non_food = create_synthetic_data('2023-01-01', 12, 'M', 'Non-Food')

# --- Step 2: Combine and Preprocess Data ---

# Combine all dataframes into one
df = pd.concat([df_w_food, df_w_non_food, df_m_food, df_m_non_food], ignore_index=True)

# Feature Engineering from Date
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Encode Categorical Features
# Using LabelEncoder for simplicity here. OneHotEncoder is often better.
le_item = LabelEncoder()
le_type = LabelEncoder()

df['ItemID_Encoded'] = le_item.fit_transform(df['ItemID'])
df['Type_Encoded'] = le_type.fit_transform(df['Type'])


# --- Step 3: Train the Model ---

# Select features and target
features = ['Year', 'Month', 'Day', 'DayOfWeek', 'ItemID_Encoded', 'Type_Encoded']
target = 'Price'

X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForest Regressor model
# Using RandomForest as it's versatile for this kind of tabular data
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)


# --- Step 4: Evaluate the Model ---

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")


# --- Step 5: Save the Model ---

# Save the trained model and the encoders
with open('model4.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('item_encoder.pkl', 'wb') as file:
    pickle.dump(le_item, file)

with open('type_encoder.pkl', 'wb') as file:
    pickle.dump(le_type, file)

print("\\nTrained model and encoders saved as model4.pkl, item_encoder.pkl, and type_encoder.pkl")