import yfinance as yf
import pandas as pd
from prophet import Prophet
import xgboost as xgb
from datetime import datetime
from skopt import gp_minimize
from skopt.space import Real, Integer
import matplotlib.pyplot as plt
import statsmodels.api as sm

def forecast_pb(ticker): 
    def add_technical_indicators(df):
        df['SMA_20'] = df['y'].rolling(window=20).mean()
        df['SMA_50'] = df['y'].rolling(window=50).mean()
        df['Volatility'] = df['y'].rolling(window=20).std()
        df = df.dropna()  # Drop NaNs created by rolling functions
        df = df[df['Volatility'] > 0]  # Ensure positive volatility
        return df

    def prepare_data(ticker):
        data = yf.download(ticker, start='2021-01-01', end='2024-07-22')
        df = add_technical_indicators(data)  # Fixed reference to data
        df.reset_index(drop=True, inplace=True)
        split_index = int(len(df) * split_ratio)
        train_data_with_features = df[:split_index]
        train_data = df[:split_index]
        return df, train_data_with_features, train_data

    def fit_and_forecast_prophet(train_data, test_data, params):
        model = Prophet(**params)
        model.fit(train_data)
        forecast = model.predict(test_data)
        return model, forecast

    def add_prophet_features(df, forecast):
        df = df.copy()
        df = df.merge(forecast[['ds', 'trend', 'yhat']], on='ds', how='left')
        return df

    def fit_xgboost(train_data_with_features, train_data):
        xgb_model = xgb.XGBRegressor()
        X_train = train_data_with_features[['trend', 'SMA_20', 'SMA_50', 'Volatility']]
        y_train = train_data['y']
        xgb_model.fit(X_train, y_train)
        return xgb_model

    def combine_predictions(test_data_with_features):
        test_data_with_features['combined_pred'] = (test_data_with_features['yhat'] + test_data_with_features['xgb_pred']) / 2
        return test_data_with_features

    def plot_acf(residuals):
        plt.figure(figsize=(10, 4))
        sm.graphics.tsa.plot_acf(residuals, lags=40)
        plt.title('ACF of Residuals')
        plt.tight_layout()
        return plt

    def plot_pacf(residuals):
        plt.figure(figsize=(10, 4))
        sm.graphics.tsa.plot_pacf(residuals, lags=40)
        plt.title('PACF of Residuals')
        plt.tight_layout()
        return plt

    def plot_residuals(residuals):
        plt.figure(figsize=(10, 4))
        plt.plot(residuals)
        plt.axhline(0, linestyle='--', color='red')
        plt.title('Residuals Plot')
        plt.tight_layout()
        return plt

    def plot_forecast(df, forecast):
        plt.figure(figsize=(12, 6))
        plt.plot(df['ds'], df['y'], label='Actual', color='blue')
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='orange')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightgray', alpha=0.5, label='Uncertainty Interval')
        plt.title('Forecast vs Actuals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        return plt

    def calculate_mean_price_diff(params, split_ratio):
        df, train_data_with_features, train_data = prepare_data(ticker)
        
        # Fit Prophet and forecast
        prophet_model, forecast_test = fit_and_forecast_prophet(train_data, df.iloc[int(len(df) * split_ratio):], params)

        # Combine predictions
        test_data_with_features = add_prophet_features(df.iloc[int(len(df) * split_ratio):], forecast_test)
        xgb_model = fit_xgboost(train_data_with_features, train_data)
        test_data_with_features['xgb_pred'] = xgb_model.predict(test_data_with_features[['trend', 'SMA_20', 'SMA_50', 'Volatility']])
        test_data_with_features = combine_predictions(test_data_with_features)

        last_closing_price = df['y'].iloc[-1]
        mean_forecast_price = test_data_with_features['combined_pred'].mean()
        return abs(mean_forecast_price - last_closing_price)

    # Download and prepare data
    ticker = 'MSFT'
    split_ratios = [0.66, 0.7, 0.75, 0.8, 0.85, 0.9]
    best_split_ratio = None
    min_difference = float('inf')
    best_params = None

    for split_ratio in split_ratios:
        search_space = [
            Real(0.001, 0.05, name='changepoint_prior_scale'),
            Real(1.0, 10.0, name='seasonality_prior_scale'),
            Real(1.0, 10.0, name='holidays_prior_scale'),
            Integer(10, 50, name='n_changepoints'),
            Real(0.6, 0.9, name='changepoint_range'),
            Real(0, 1, name='seasonality_mode')  # 0 for additive, 1 for multiplicative
        ]

        def objective(params):
            return calculate_mean_price_diff(params, split_ratio)

        res = gp_minimize(objective, search_space, n_calls=20, random_state=42)

        current_best_params = {
            'changepoint_prior_scale': res.x[0],
            'seasonality_prior_scale': res.x[1],
            'holidays_prior_scale': res.x[2],
            'n_changepoints': res.x[3],
            'changepoint_range': res.x[4],
            'seasonality_mode': 'additive' if res.x[5] < 0.5 else 'multiplicative'
        }

        # Fit Prophet and forecast with best parameters
        df, train_data_with_features, train_data = prepare_data(ticker)
        prophet_model, forecast_test = fit_and_forecast_prophet(df.iloc[:int(len(df) * split_ratio)], df.iloc[int(len(df) * split_ratio):], current_best_params)

        train_data_with_features = add_prophet_features(df.iloc[:int(len(df) * split_ratio)], prophet_model.predict(df.iloc[:int(len(df) * split_ratio)]))
        test_data_with_features = add_prophet_features(df.iloc[int(len(df) * split_ratio):], forecast_test)

        xgb_model = fit_xgboost(train_data_with_features, df.iloc[:int(len(df) * split_ratio)])
        test_data_with_features['xgb_pred'] = xgb_model.predict(test_data_with_features[['trend', 'SMA_20', 'SMA_50', 'Volatility']])

        test_data_with_features = combine_predictions(test_data_with_features)

        last_closing_price = df['y'].iloc[-1]
        mean_forecast_price = test_data_with_features['combined_pred'].mean()
        mean_diff = abs(mean_forecast_price - last_closing_price)

        if mean_diff < min_difference:
            min_difference = mean_diff
            best_split_ratio = split_ratio
            best_params = current_best_params

    print(f"Best split ratio: {best_split_ratio}")

    # Use the best split ratio for final training and forecasting
    train_size = int(len(df) * best_split_ratio)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:].copy()

    prophet_model, forecast_test = fit_and_forecast_prophet(train_data, test_data, best_params)

    train_data_with_features = add_prophet_features(train_data, prophet_model.predict(train_data))
    test_data_with_features = add_prophet_features(test_data, forecast_test)

    xgb_model = fit_xgboost(train_data_with_features, train_data)
    test_data_with_features['xgb_pred'] = xgb_model.predict(test_data_with_features[['trend', 'SMA_20', 'SMA_50', 'Volatility']])

    test_data_with_features = combine_predictions(test_data_with_features)

    residuals = test_data_with_features['combined_pred'] - test_data_with_features['y']

    # Return final graphs only
    acf_plot = plot_acf(residuals)
    pacf_plot = plot_pacf(residuals)
    residuals_plot = plot_residuals(residuals)
    forecast_plot = plot_forecast(test_data, forecast_test)

    return forecast_plot, acf_plot, pacf_plot, residuals_plot 