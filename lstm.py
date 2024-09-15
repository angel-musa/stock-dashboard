import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from datetime import date, timedelta
from keras._tf_keras.keras.layers import Dense, LSTM, Dropout
from keras._tf_keras.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas_ta as ta
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import sys

def forecast_lstm(ticker): 

    # Download stock data
    end_date=datetime.now().strftime('%Y-%m-%d')
    azn_df = yf.download(ticker, start="2021-01-01", end=end_date)

    # # Plot Adjusted Close price
    # sns.set(rc={'figure.figsize':(16, 8)})
    # azn_df['Adj Close'].plot(grid=True)
    # plt.title('MSFT Adjusted Close Price', color='black', fontsize=20)
    # plt.xlabel('Year', color='black', fontsize=15)
    # plt.ylabel('Stock price', color='black', fontsize=15)
    # plt.show()

    # Prepare data
    azn_adj = azn_df[['Adj Close']]
    azn_adj_arr = azn_adj.values
    training_data_len = int(0.8 * len(azn_adj_arr))

    # Create train and test data sets
    train = azn_adj_arr[:training_data_len]
    test = azn_adj_arr[training_data_len:]

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train)

    # Create training data structure with 60 time-steps
    X_train, y_train = [], []
    for i in range(60, len(train_scaled)):
        X_train.append(train_scaled[i-60:i, 0])
        y_train.append(train_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=64)

    # Prepare test data
    total_data = np.concatenate((train, test), axis=0)
    inputs = total_data[len(total_data) - len(test) - 60:]
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i-60:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predict and inverse transform the predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Evaluate the model
    rmse = np.sqrt(np.mean(predictions - test)**2)
    print(f"Root Mean Squared Error: {rmse}")

    # Plot the results
    train = azn_adj[:training_data_len]
    test = azn_adj[training_data_len:]
    test['Predictions'] = predictions

    # Create the Plotly figure
    fig = go.Figure()

    # Add traces for the training data
    fig.add_trace(go.Scatter(x=train.index, y=train['Adj Close'], mode='lines', name='Training'))

    # Add traces for the actual test data
    fig.add_trace(go.Scatter(x=test.index, y=test['Adj Close'], mode='lines', name='Actual'))

    # Add traces for the predicted test data
    fig.add_trace(go.Scatter(x=test.index, y=test['Predictions'], mode='lines', name='Predicted'))

    # Update the layout of the figure
    fig.update_layout(
        title=ticker + " Time Series Analysis",
        xaxis_title="Year",
        yaxis_title="Stock Price",
        legend_title="Legend",
        template="plotly_white"
    )

    # Assuming 'fig' is your plotly figure

    fig.update_layout(
        hovermode='x unified',  # This adds a vertical line across the graph
        xaxis=dict(
            showspikes=True,    # This will show a spike (line) at the x-value
            spikemode='across', # This will make the spike go across the entire plot
            spikesnap='cursor', # This makes the spike snap to the cursor
            showline=True,      # Ensures the x-axis line is visible
        ),
        yaxis=dict(
            showspikes=True,    # This will show a spike (line) at the y-value
            spikemode='across', # This will make the spike go across the entire plot
            spikesnap='cursor', # This makes the spike snap to the cursor
            showline=True,      # Ensures the y-axis line is visible
        )
    )

    fig.update_traces(
        hoverinfo="x+y",  # This ensures both x and y values are shown on hover
        mode='lines',     # Sets the mode to lines for continuous graphs
    )

        # fig.show()

    # # Predict Adjusted Close price for the next day
    # last_60_days = azn_adj[-60:].values
    # last_60_days_scaled = scaler.transform(last_60_days)
    # X_test = np.array([last_60_days_scaled])
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # pred_price = model.predict(X_test)
    # pred_price = scaler.inverse_transform(pred_price)
    # print(f"Predicted Adjusted Close Price for the next day: {pred_price[0][0]}")

    # Predict Adjusted Close price for the next 5 days
    pred_prices = []
    last_60_days = azn_adj[-60:].values

        # Calculate ATR
    atr = ta.atr(azn_df['High'], azn_df['Low'], azn_df['Adj Close'], length=14).iloc[-1]

    for _ in range(5):
        # Scale the last 60 days data
        last_60_days_scaled = scaler.transform(last_60_days)

        # Prepare the input to the model
        X_test = np.array([last_60_days_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Predict the next day's price
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)

        # Append the prediction to the list
        pred_prices.append(pred_price[0][0])

        # Update the last 60 days data by removing the oldest value and adding the predicted price
        last_60_days = np.append(last_60_days[1:], pred_price, axis=0)

    # # Calculate the average of the predicted prices
    # average_pred_price = np.mean(pred_prices)

    # # Get the last actual closing price from the data
    # last_actual_close = azn_adj['Adj Close'].iloc[-1]

    # # Compare the average predicted price to the ATR range
    # if average_pred_price > last_actual_close + atr:
    #     result = "Above ATR"
    # elif average_pred_price < last_actual_close - atr:
    #     result = "Below ATR"
    # else:
    #     result = "Within ATR"

    # # # Compare the average predicted price to the last actual closing price
    # # if average_pred_price > last_actual_close:
    # #     result = "Positive"
    # # else:
    # #     result = "Negative"

    # # Output the result
    # print(f"The average forecasted price for the next 5 days is {average_pred_price:.2f}.")
    # print(f"The last actual closing price is {last_actual_close:.2f}.")
    # print(f"The forecast is {result}.")

    # Residuals
    residuals = test['Adj Close'].values - test['Predictions']

    # Plot ACF and PACF of residuals
    fig_acf = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_acf(residuals, lags=20)
    plt.title('ACF of Residuals')

    fig_pacf = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_pacf(residuals, lags=20)
    plt.title('PACF of Residuals')

    fig_residuals = plt.figure(figsize=(10, 5))
    plt.plot(residuals)
    plt.title('Residuals')
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.axhline(0, color='red', linestyle='--')

    return fig, fig_acf, fig_pacf, fig_residuals
