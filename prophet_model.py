import yfinance as yf
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import plotly.graph_objs as go
import math
from datetime import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt

def forecast_prophet(ticker): 
    # Download historical data from Yahoo Finance
    end_date = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(ticker, start='2021-01-01', end=end_date)

    # Prepare data for Prophet
    df = pd.DataFrame()
    df['ds'] = data.index
    df['y'] = data['Close'].values

    # Split data into train and test sets
    train_size = int(len(df) * 0.66)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:].copy()

    # Define the objective function for Bayesian Optimization
    def objective(params):
        changepoint_prior_scale, seasonality_prior_scale, holidays_prior_scale, n_changepoints, changepoint_range, seasonality_mode = params
        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            n_changepoints=n_changepoints,
            changepoint_range=changepoint_range,
            seasonality_mode=seasonality_mode
        )
        model.fit(train_data)
        forecast = model.predict(test_data)
        mse = mean_squared_error(test_data['y'], forecast['yhat'])
        return mse

    # Define the search space
    search_space = [
        Real(0.001, 0.5, name='changepoint_prior_scale'),
        Real(1.0, 20.0, name='seasonality_prior_scale'),
        Real(1.0, 20.0, name='holidays_prior_scale'),
        Integer(10, 30, name='n_changepoints'),
        Real(0.5, 1.0, name='changepoint_range'),
        Categorical(['additive', 'multiplicative'], name='seasonality_mode')
    ]

    # Perform Bayesian Optimization
    res = gp_minimize(objective, search_space, n_calls=20, random_state=42)

    # Extract best parameters
    best_params = {
        'changepoint_prior_scale': res.x[0],
        'seasonality_prior_scale': res.x[1],
        'holidays_prior_scale': res.x[2],
        'n_changepoints': res.x[3],
        'changepoint_range': res.x[4],
        'seasonality_mode': res.x[5]
    }

    # Initialize Prophet model with best parameters
    model = Prophet(**best_params)
    model.fit(train_data)

    # Make predictions on test data
    forecast = model.predict(test_data)

    # Forecast the next 7 days starting after test data
    forecast_end_date = df['ds'].max()  # Last date in test_data
    future_dates = pd.date_range(start=forecast_end_date + pd.Timedelta(days=1), periods=7, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})
    forecast_7days = model.predict(future_df)

    # Extend the forecast into 2024
    future_dates_2024 = pd.date_range(start='2024-07-18', end='2024-12-31', freq='D')
    future_df_2024 = pd.DataFrame({'ds': future_dates_2024})
    forecast_2024 = model.predict(future_df_2024)

    # Update test_data with predicted values using .loc to avoid SettingWithCopyWarning
    test_data.loc[:, 'yhat'] = forecast['yhat'].values

    # Calculate residuals using .loc
    test_data.loc[:, 'residuals'] = test_data['y'] - test_data['yhat']

    # Calculate Root Mean Squared Error (RMSE) on test data
    rmse = math.sqrt(mean_squared_error(test_data['y'], test_data['yhat']))
    print(f"Root Mean Squared Error (RMSE) on test data: {rmse:.2f}")

    # Plot results using Plotly for interactive plot
    fig = go.Figure()

    # Actual data
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual', hovertemplate='Date: %{x}<br>Price: %{y:.2f}'))

    # Predicted on test data
    fig.add_trace(go.Scatter(x=test_data['ds'], y=test_data['yhat'], mode='lines', name='Predicted on Test Data', hovertemplate='Date: %{x}<br>Price: %{y:.2f}'))

    # Forecasted next 7 days
    fig.add_trace(go.Scatter(x=forecast_7days['ds'], y=forecast_7days['yhat'], mode='lines', name='Forecasted (Next 7 Days)', hovertemplate='Date: %{x}<br>Price: %{y:.2f}'))

    # Extended forecast for 2024
    fig.add_trace(go.Scatter(x=forecast_2024['ds'], y=forecast_2024['yhat'], mode='lines', name='Extended Forecast (2024)', line=dict(dash='dash', color='orange'), hovertemplate='Date: %{x}<br>Price: %{y:.2f}'))

    # Add range slider and buttons for interactive features
    fig.update_layout(
        title=f'{ticker} Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Stock Price',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    # Create ACF and PACF plots
    fig_acf, ax_acf = plt.subplots()
    sm.tsa.stattools.acf(test_data['residuals'], nlags=20, fft=False)
    sm.graphics.tsa.plot_acf(test_data['residuals'], ax=ax_acf, lags=20)
    ax_acf.set_title('Autocorrelation Function (ACF)')
    plt.close(fig_acf)  # Prevent displaying it immediately

    fig_pacf, ax_pacf = plt.subplots()
    sm.graphics.tsa.plot_pacf(test_data['residuals'], ax=ax_pacf, lags=20)
    ax_pacf.set_title('Partial Autocorrelation Function (PACF)')
    plt.close(fig_pacf)  # Prevent displaying it immediately

    # Create Residuals plot
    fig_residuals, ax_residuals = plt.subplots()
    ax_residuals.plot(test_data['ds'], test_data['residuals'], label='Residuals', color='purple')
    ax_residuals.axhline(y=0, color='r', linestyle='--')
    ax_residuals.set_title('Residuals Plot')
    ax_residuals.set_xlabel('Date')
    ax_residuals.set_ylabel('Residuals')
    ax_residuals.legend()
    plt.close(fig_residuals)  # Prevent displaying it immediately

    return fig, fig_acf, fig_pacf, fig_residuals
