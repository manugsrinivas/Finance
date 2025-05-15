
# Stock Price Prediction Model

This repository contains a time-series prediction project that uses machine learning to forecast next-day stock prices based on the past week's closing price data. Initially centered on accurately predicting the next day's numerical price, the project's focus evolved toward predicting broader market trends and directional movements of stock prices. The model leverages historical daily price data collected via the Polygon.io API and implements both regression modeling and trend analysis for evaluation.

## Project Description

The dataset was obtained through the Polygon.io API, which provided two years' worth of daily trading data for a selected stock (AAPL). The project primarily utilizes the closing prices of each day, structured into sliding windows that represent the previous 7 days. These rolling windows are used as input features to predict the closing price of the next day. This approach enables the model to learn short-term temporal dependencies in stock movements.

The notebook includes data fetching, preprocessing, model training, evaluation, and visualization of predictions versus actual prices. In addition to numerical accuracy, the model also attempts to capture directional movement trends, comparing whether the predicted price correctly signals a rise or fall in stock price compared to the previous day.

## Methods and Models

Two primary regression models were implemented to predict next-day closing prices:

1. **Linear Regression:** A simple baseline model that assumes a linear relationship between past closing prices and the next day's price.
2. **Random Forest Regressor:** An ensemble learning method that creates multiple decision trees and aggregates their predictions for improved accuracy and robustness.

After training the models, predictions were compared to actual outcomes using standard regression evaluation metrics. The project includes visual plots to show the alignment between predicted and actual stock prices, highlighting both the accuracy and the directionality of the forecasts.

## Evaluation Metrics

To assess model performance, the following metrics were used:

- **Mean Squared Error (MSE)**: Measures the average of the squares of the errors between predicted and actual values.
- **Mean Absolute Error (MAE)**: Captures the average absolute difference.
- **R² Score**: Indicates how well the model captures variance in the data.
- **Trend Accuracy**: Measures whether the model correctly predicts the direction (up/down) of the next day's price.

Findings indicated that while raw price prediction is inherently noisy due to market volatility, the models showed meaningful success in identifying short-term trend directions, with the Random Forest model outperforming the linear baseline in both accuracy and interpretability.

## Libraries Used

- `numpy`, `pandas` – numerical and tabular data manipulation
- `matplotlib.pyplot` – visualization of predicted vs. actual prices
- `sklearn.model_selection.train_test_split` – train/test data splitting
- `sklearn.linear_model.LinearRegression` – linear baseline model
- `sklearn.ensemble.RandomForestRegressor` – ensemble modeling
- `sklearn.metrics` – for MSE, MAE, and R² score calculations
- `requests`, `datetime` – API access and date handling
