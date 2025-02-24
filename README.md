# Stock Price Prediction using Machine Learning

## Overview
This project predicts stock prices for Apple (AAPL) using machine learning algorithms like Linear Regression and Random Forest. The model is designed to help investors make informed decisions by predicting future stock movements.

## Techniques Used
- **Linear Regression**: A simple linear model used for prediction based on historical stock prices.
- **Random Forest**: A more advanced ensemble learning method for improving prediction accuracy.
- **Feature Engineering**: Features like lag variables and moving averages were created to enhance model performance.

## How to Run the Code
1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/Stock-Price-Prediction.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the main script to train the model and make predictions:
    ```bash
    python main.py
    ```

## Performance Metrics
- **Random Forest MAE**: ~3.5
- **Linear Regression MAE**: ~2.5

## Future Improvements
- Incorporate additional features like technical indicators.
- Experiment with deep learning models (e.g., LSTM) for better prediction accuracy.

