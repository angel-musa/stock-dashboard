import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import yfinance as yf
from datetime import datetime, timedelta
import xgboost as xgb
import optuna
from sklearn.ensemble import BaggingRegressor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def forecast_boost(stock_symbol):
    # Load data
    start_date = '2021-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    data = data.reset_index()

    # Prepare the data
    data = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    data['unique_id'] = stock_symbol  # Add unique_id column

    # Train-test split
    train_size = int(len(data) * 0.66)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Manual Feature Engineering
    def create_lags(df, lags):
        df = df.copy()
        for lag in lags:
            df[f'lag_{lag}'] = df['y'].shift(lag)
        return df

    def create_ewm(df, spans):
        df = df.copy()
        for span in spans:
            df[f'ewm_{span}'] = df['y'].ewm(span=span).mean()
        return df

    def add_date_parts(df):
        df = df.copy()
        df['year'] = df['ds'].dt.year
        df['month'] = df['ds'].dt.month
        df['day'] = df['ds'].dt.day
        df['weekday'] = df['ds'].dt.weekday
        return df

    # Feature engineering for train and test data
    train_data = create_lags(train_data, lags=[1, 2, 3, 4, 5, 6, 7])
    train_data = create_ewm(train_data, spans=[7, 14, 21])
    train_data = add_date_parts(train_data)
    train_data = train_data.dropna().reset_index(drop=True)  # Drop rows with NaN values

    test_data = create_lags(test_data, lags=[1, 2, 3, 4, 5, 6, 7])
    test_data = create_ewm(test_data, spans=[7, 14, 21])
    test_data = add_date_parts(test_data)
    test_data = test_data.dropna().reset_index(drop=True)  # Drop rows with NaN values

    # Define feature columns
    feature_columns = [col for col in train_data.columns if col.startswith('lag_') or col.startswith('ewm_') or col in ['year', 'month', 'day', 'weekday']]

    # Define the objective function for Bayesian Optimization
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1.0),
            'gamma': trial.suggest_uniform('gamma', 0, 1),
            'lambda': trial.suggest_loguniform('lambda', 1e-4, 1.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-4, 1.0)
        }
        
        model = xgb.XGBRegressor(**params)
        
        # Train the model
        model.fit(train_data[feature_columns], train_data['y'])
        
        # Predict
        predictions = model.predict(test_data[feature_columns])
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(test_data['y'], predictions))
        return rmse

    # Optimize hyperparameters
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    # Initialize the XGBoost model with best parameters
    best_params = study.best_params
    xgb_model = xgb.XGBRegressor(**best_params)
    model = BaggingRegressor(xgb_model, n_estimators=30)

    # Fit the model with the best parameters
    model.fit(train_data[feature_columns], train_data['y'])

    # Predict on the test_data range
    predictions = model.predict(test_data[feature_columns])

    # Calculate residuals
    residuals = test_data['y'] - predictions

    # Create ACF and PACF plots
    acf_fig, acf_ax = plt.subplots(figsize=(10, 6))
    plot_acf(residuals, ax=acf_ax)
    acf_ax.set_title('Autocorrelation Function of Residuals')

    pacf_fig, pacf_ax = plt.subplots(figsize=(10, 6))
    plot_pacf(residuals, ax=pacf_ax)
    pacf_ax.set_title('Partial Autocorrelation Function of Residuals')

    # Define forecast horizon (number of days you want to forecast beyond test_data)
    forecast_horizon = 30  # Example: forecast 30 days into the future

    # Create future dates
    future_dates = pd.date_range(start=data['ds'].iloc[-1] + timedelta(days=1), periods=forecast_horizon)
    future_data = pd.DataFrame({
        'ds': future_dates,
        'unique_id': stock_symbol
    })

    # Ensure future_data has 'y' column and fill it with the last known 'y' value
    future_data['y'] = data['y'].iloc[-1]

    # Add placeholders for required features
    future_data = create_lags(future_data, lags=[1, 2, 3, 4, 5, 6, 7])
    future_data = create_ewm(future_data, spans=[7, 14, 21])
    future_data = add_date_parts(future_data)

    # Fill NaN values with suitable default values
    for col in feature_columns:
        if col not in future_data.columns:
            future_data[col] = 0

    # Predict future prices
    future_predictions = model.predict(future_data[feature_columns])

    # Plot the results
    forecast_fig, forecast_ax = plt.subplots(figsize=(14, 7))
    
    # Plot actual prices
    forecast_ax.plot(data['ds'], data['y'], label='Actual Prices', color='blue')

    # Plot test data (actual prices within the test period)
    forecast_ax.plot(test_data['ds'], test_data['y'], label='Test Data', color='orange')

    # Plot model predictions
    forecast_ax.plot(test_data['ds'], predictions, label='Predicted Prices', color='red')

    # Plot future forecast prices
    forecast_ax.plot(future_dates, future_predictions, label='Future Forecast', color='green', linestyle='--')

    forecast_ax.set_title(f'{stock_symbol} Stock Price Forecast')
    forecast_ax.set_xlabel('Date')
    forecast_ax.set_ylabel('Price')
    forecast_ax.legend()
    forecast_ax.grid(True)

    # Show all plots
    plt.show()

    return forecast_fig, acf_fig, pacf_fig, residuals