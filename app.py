from flask import Flask, request, render_template
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Prevent GUI backend warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

app = Flask(__name__)

# Ensure directories exist
os.makedirs('static', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker'].upper()
    data = yf.download(ticker, start='2015-01-01', end='2024-12-31')

    if data.empty:
        return "Invalid ticker symbol or no data found."

    # Save data
    data.to_csv(f"data/{ticker}_data.csv")

    # --- EDA Section ---
    # Closing price plot
    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'], label='Closing Price', color='blue')
    plt.title(f"{ticker} Closing Price History")
    plt.xlabel('Date')
    plt.ylabel('Price USD ($)')
    plt.legend()
    eda_plot_path = f'static/{ticker}_eda.png'
    plt.savefig(eda_plot_path)
    plt.close()

    # Moving averages
    data['MA50'] = data['Close'].rolling(50).mean()
    data['MA200'] = data['Close'].rolling(200).mean()
    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'], label='Close', color='blue')
    plt.plot(data['MA50'], label='50-Day MA', color='red')
    plt.plot(data['MA200'], label='200-Day MA', color='green')
    plt.title(f'{ticker} - 50 & 200 Day Moving Averages')
    plt.legend()
    ma_plot_path = f'static/{ticker}_ma.png'
    plt.savefig(ma_plot_path)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title(f'{ticker} - Correlation Heatmap')
    corr_plot_path = f'static/{ticker}_corr.png'
    plt.savefig(corr_plot_path)
    plt.close()

    # --- LSTM Model ---
    close_data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)

    def create_sequences(dataset, time_step=60):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60
    X, y = create_sequences(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    model.save(f"models/{ticker}_lstm_model.h5")

    train_size = int(len(scaled_data) * 0.8)
    test_data = scaled_data[train_size - time_step:]
    X_test, y_test = create_sequences(test_data, time_step)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = math.sqrt(mean_squared_error(y_test_unscaled, predictions))
    mae = mean_absolute_error(y_test_unscaled, predictions)

    # Plot actual vs predicted
    plt.figure(figsize=(10, 5))
    plt.plot(data.index[-len(y_test_unscaled):], y_test_unscaled, label='Actual')
    plt.plot(data.index[-len(y_test_unscaled):], predictions, label='Predicted')
    plt.title(f'{ticker} - LSTM Predicted vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    lstm_plot_path = f'static/{ticker}_lstm.png'
    plt.savefig(lstm_plot_path)
    plt.close()

    # --- Prophet Forecast ---
    df_prophet = data.reset_index()[['Date', 'Close']]
    df_prophet.columns = ['ds', 'y']
    prophet_model = Prophet()
    prophet_model.fit(df_prophet)

    future = prophet_model.make_future_dataframe(periods=180)
    forecast = prophet_model.predict(future)

    plt.figure(figsize=(10, 5))
    plt.plot(df_prophet['ds'], df_prophet['y'], label='Actual')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
    plt.title(f'{ticker} - Prophet Forecast (6 months ahead)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    prophet_plot_path = f'static/{ticker}_prophet.png'
    plt.savefig(prophet_plot_path)
    plt.close()

    # Save results
    final_df = pd.DataFrame({
        'Date': data.index[-len(y_test_unscaled):],
        'Actual': y_test_unscaled.flatten(),
        'Predicted_LSTM': predictions.flatten()
    })
    final_df.to_csv(f"data/{ticker}_predictions.csv", index=False)

    return render_template(
        'result.html',
        ticker=ticker,
        rmse=round(rmse, 2),
        mae=round(mae, 2),
        eda_img=eda_plot_path,
        ma_img=ma_plot_path,
        corr_img=corr_plot_path,
        lstm_img=lstm_plot_path,
        prophet_img=prophet_plot_path
    )

if __name__ == '__main__':
    app.run(debug=True)
