# Stock Price Prediction using LSTM and Prophet 

## Overview 
A deep learning and time series forecasting project that predicts stock prices using LSTM (Long-Short-Term-Memory) and Prophet models. It also includes a Flask web app for interactive stock forecasting and visual insights.

## Features 
- Fetches live stock data using Yahoo Finance (YFinance)
- Performs EDA with trend, correlation and moving average
- Trains LSTM for time-series prediction
- Uses Prophet for seasonal and trend-based forecasting
- Generates insightful plots automatically

## Tech Stack 
| Category | Tools / Libraries |
|-----------|------------------|
| **Backend** | Flask |
| **ML / DL** | TensorFlow, Prophet |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Frontend** | HTML, CSS (Bootstrap) |

## Prokect Structure 
<pre> STOCK_PRICE_PREDICTION/
│
├── data/ 
│ ├── AAPL_data.csv
│ ├── AAPL_predictions.csv
│ ├── GOOG_data.csv
│ └── GOOG_predictions.csv
│
├── models/ 
│ ├── AAPL_lstm_model.h5
│ └── GOOG_lstm_model.h5
│
├── static/ 
│ ├── AAPL_corr.png
│ ├── AAPL_lstm.png
│ ├── AAPL_prophet.png
│ └── ...
│
├── templates/ 
│ ├── index.html
│ └── result.html
│
├── app.py 
├── requirements.txt </pre>
