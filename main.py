import streamlit as st
import yfinance as yf
import importlib
import matplotlib.pyplot as plt
from datetime import datetime
from arima import forecast_arima
from boost import forecast_boost
from prophet_model import forecast_prophet
from lstm import forecast_lstm

# Function to display explanations for ACF and PACF plots
def display_plots_and_explanations(acf_fig, pacf_fig, residuals_fig):
    # Display ACF and PACF plots
        st.pyplot(acf_fig)
        st.write("""
        **Autocorrelation Function (ACF)**: The ACF plot helps identify the correlation between a time series and its lagged values. Significant spikes in the ACF indicate the presence of autocorrelation, which suggests that past values have an influence on current values.
        """)

        st.pyplot(pacf_fig)
        st.write("""
        **Partial Autocorrelation Function (PACF)**: The PACF plot is used to determine the extent of lag that influences the time series. The first significant lag indicates how many past values should be included in the model. 
        """)

        # Display residuals plot
        st.pyplot(residuals_fig)
        st.write("""
        **Residuals Plot**: The residuals plot shows the difference between the predicted and actual values. Ideally, the residuals should be randomly scattered around zero, indicating that the model has captured all underlying patterns in the data. Patterns in the residuals may suggest that the model can be improved. 
        """)


# Main screen for ticker search
st.title("Time-Series Forecasting Dashboard")

# Model selection section
st.subheader("Model Selection")
model_option = st.selectbox("Choose forecasting model:", ("ARIMA", "XGBoost", "Prophet", "LSTM"))

# Ticker input section
ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL):")
end_date = datetime.now().strftime('%Y-%m-%d')

if ticker:
    # Fetching data from yfinance
    data = yf.download(ticker, start="2021-01-01", end=end_date)
    st.write(f"Stock data for {ticker}")
    st.line_chart(data['Close'])

    # Short-term forecasting placeholder
    st.subheader("Short-Term Forecast")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Actual Data')

    # Load the selected model's code from the corresponding file
    if model_option == "ARIMA":
        final_forecast_fig, acf_fig1, pacf_fig1, residuals_fig1 = forecast_arima(ticker)
        st.pyplot(final_forecast_fig)
        display_plots_and_explanations(acf_fig1, pacf_fig1, residuals_fig1)
    elif model_option == "XGboost":
        final_forecast_fig, acf_fig2, pacf_fig2, residuals_fig2 = forecast_boost(ticker)
        st.pyplot(final_forecast_fig)
        display_plots_and_explanations(acf_fig2, pacf_fig2, residuals_fig2)
    elif model_option == "Prophet":
        final_forecast_fig, acf_fig2, pacf_fig2, residuals_fig2 = forecast_prophet(ticker)
        st.plotly_chart(final_forecast_fig)
        display_plots_and_explanations(acf_fig2, pacf_fig2, residuals_fig2)
    elif model_option == "LSTM":
        final_forecast_fig, acf_fig2, pacf_fig2, residuals_fig2 = forecast_lstm(ticker)
        st.plotly_chart(final_forecast_fig)
        display_plots_and_explanations(acf_fig2, pacf_fig2, residuals_fig2)
         

    
    # # Run the forecast and get the predicted data
    # predictions, train, test = model_module.forecast(data['Close'])
    
    # # Plot training, testing, and predicted data
    # ax.plot(train.index, train, label='Training Data')
    # ax.plot(test.index, test, label='Testing Data')
    # ax.plot(predictions.index, predictions, label='Predicted Data')

    # ax.legend()
    # st.pyplot(fig)

