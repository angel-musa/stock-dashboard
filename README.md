# Time Series Forecasting Dashboard 📊

The **Time Series Forecasting Dashboard** is a web-based tool built with **Streamlit** to help users forecast stock prices using advanced machine learning models. This project provides a user-friendly interface for exploring time series data, visualizing historical trends, and generating accurate forecasts using various models.

## Features

- **Stock Selection**: Users can select stock tickers using a search bar to retrieve and visualize the corresponding historical data.
- **Forecasting Models**: Includes three powerful forecasting models:
  - **ARIMA**: AutoRegressive Integrated Moving Average
  - **LSTM**: Long Short-Term Memory neural network
  - **Prophet**: Facebook Prophet model
- **Interactive Visualization**: View training, testing, and forecasted data in interactive plots.
- **Model Performance**: Compare model accuracy by visualizing performance metrics (e.g., RMSE, MAPE).
- **Residual Analysis**: Visualize residual plots, Auto-Correlation Function (ACF), and Partial Auto-Correlation Function (PACF) for each model.
- **Short-term Forecasting**: Users can generate short-term future forecasts for selected stocks.
  
## How It Works

1. **Stock Data**: Upon entering a stock ticker, the application retrieves historical stock data.
2. **Model Selection**: Users select one of the available forecasting models from the sidebar.
3. **Forecast Visualization**: The app generates and displays forecasts along with historical data. Residuals and error metrics are visualized to assess the accuracy of predictions.
4. **ACF/PACF Plots**: ACF and PACF plots are generated to analyze correlations in the residuals.
   
