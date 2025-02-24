from gettext import install

import yfinance as yf

# Download stock data for Apple from January 1, 2018, to January 1, 2023
data = yf.download('AAPL', start='2018-01-01', end='2023-01-01')

# View the first few rows of the data
print(data.head())

import yfinance as yf
import pandas as pd
import numpy as np

# Download stock data for Apple from January 1, 2018, to January 1, 2023
data = yf.download('AAPL', start='2018-01-01', end='2023-01-01')

# Drop rows with missing values (optional)
data = data.dropna()

# Add a column for the daily percentage change in the 'Close' price
data['Pct_Change'] = data['Close'].pct_change()

# Remove the first row (because the percentage change is NaN)
data = data.dropna()

# Display the first few rows of the preprocessed data
print(data.head())

# Create lag features (e.g., previous day's close price)
data['Lag_1'] = data['Close'].shift(1)

# Create moving averages (5-day and 30-day)
data['MA_5'] = data['Close'].rolling(window=5).mean()
data['MA_30'] = data['Close'].rolling(window=30).mean()

# Create a feature for the daily volume
data['Volume'] = data['Volume']

# Drop any rows with NaN values (because of shifting or moving averages)
data = data.dropna()

# Display the first few rows of the data with new features
print(data.head())

from sklearn.model_selection import train_test_split

# Define the features (X) and target variable (y)
X = data[['Lag_1', 'MA_5', 'MA_30', 'Volume']]  # Features
y = data['Close']  # Target variable (next day's close)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Display the shape of the training and testing sets
print("Training set:", X_train.shape, y_train.shape)
print("Testing set:", X_test.shape, y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate the performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Display the results
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)


from sklearn.linear_model import LinearRegression

# Initialize the Linear Regression model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Make predictions
lr_pred = lr_model.predict(X_test)

# Calculate the performance metrics for Linear Regression
from sklearn.metrics import mean_absolute_error, mean_squared_error

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)


# Train Random Forest model
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions with Random Forest
rf_pred = rf_model.predict(X_test)

# Calculate performance metrics for Random Forest
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)

# Print the comparison of both models
print("\nRandom Forest Mean Absolute Error:", rf_mae)
print("Random Forest Mean Squared Error:", rf_mse)

print("\nLinear Regression Mean Absolute Error:", lr_mae)
print("Linear Regression Mean Squared Error:", lr_mse)


# Import the necessary visualization library
import matplotlib.pyplot as plt

# Plotting Random Forest predictions vs actual
plt.figure(figsize=(14, 7))

# Random Forest
plt.subplot(1, 2, 1)  # (rows, columns, position)
plt.plot(y_test.index, y_test, color='blue', label='Actual Prices')
plt.plot(y_test.index, rf_pred, color='red', label='Random Forest Predictions')
plt.title('Random Forest: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.legend()

# Linear Regression
plt.subplot(1, 2, 2)
plt.plot(y_test.index, y_test, color='blue', label='Actual Prices')
plt.plot(y_test.index, lr_pred, color='green', label='Linear Regression Predictions')
plt.title('Linear Regression: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()


# Predicting for the next day
# Get the most recent data from the test set (or latest available data point)
recent_data = X_test.tail(1)

# Use the trained model to predict the next stock price
rf_future_pred = rf_model.predict(recent_data)
lr_future_pred = lr_model.predict(recent_data)

# Print the predictions
print("Random Forest Predicted Next Day Price:", rf_future_pred[0])
print("Linear Regression Predicted Next Day Price:", lr_future_pred[0])


# Importing the necessary visualization library
import matplotlib.pyplot as plt

# Plotting Random Forest predictions vs actual
plt.figure(figsize=(14, 7))

# Random Forest
plt.subplot(1, 2, 1)  # (rows, columns, position)
plt.plot(y_test.index, y_test, color='blue', label='Actual Prices')
plt.plot(y_test.index, rf_pred, color='red', label='Random Forest Predictions')
plt.title('Random Forest: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.legend()

# Linear Regression
plt.subplot(1, 2, 2)
plt.plot(y_test.index, y_test, color='blue', label='Actual Prices')
plt.plot(y_test.index, lr_pred, color='green', label='Linear Regression Predictions')
plt.title('Linear Regression: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()


import joblib

# Save the models to disk
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(lr_model, 'linear_regression_model.pkl')

from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained models
rf_model = joblib.load('random_forest_model.pkl')
lr_model = joblib.load('linear_regression_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define a route to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the POST request
        data = request.get_json()

        # Extract features from the incoming data (adjust according to your feature set)
        lag_1 = data['Lag_1']
        ma_5 = data['MA_5']
        ma_30 = data['MA_30']
        volume = data['Volume']

        # Create the feature array
        features = np.array([[lag_1, ma_5, ma_30, volume]])

        # Make predictions using both models
        rf_pred = rf_model.predict(features)[0]
        lr_pred = lr_model.predict(features)[0]

        # Return the predictions as a JSON response
        return jsonify({
            'Random Forest Prediction': rf_pred,
            'Linear Regression Prediction': lr_pred
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True)












