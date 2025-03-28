from flask import Flask, render_template, request
from pycoingecko import CoinGeckoAPI
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import plotly.graph_objects as go
import logging

app = Flask(__name__)
cg = CoinGeckoAPI()
models = {horizon: joblib.load(f'xgb_model_{horizon}.pkl') for horizon in ['1d', '2d', '1w', '1m']}
scaler = joblib.load('scaler.pkl')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/')
def index():
    coins = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'USDT-USD', 'BNB-USD']
    coin_ids = {
        'BTC-USD': 'bitcoin', 'ETH-USD': 'ethereum', 'LTC-USD': 'litecoin',
        'USDT-USD': 'tether', 'BNB-USD': 'binancecoin'
    }
    live_prices = {}
    for coin in coins:
        try:
            price = cg.get_price(ids=coin_ids[coin], vs_currencies='usd')[coin_ids[coin]]['usd']
            live_prices[coin] = price
        except Exception as e:
            live_prices[coin] = f"Error: {e}"
    return render_template('index.html', prices=live_prices)

@app.route('/predict', methods=['POST'])
def predict():
    coin = request.form['coin']
    df = pd.read_csv('featured_data.csv')
    coin_data = df[df['Coin'] == coin].tail(30)  # Last 30 days for context
    if len(coin_data) < 30:
        return render_template('result.html', coin=coin, predictions={}, chart=None, indicators=None)
    
    features = ['Open', 'Close', 'Volume', 'Coin_Encoded', 
                'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_7', 'Close_Lag_14', 'Close_Lag_30',
                'Volume_Lag_1', 'Volume_Lag_2', 'Volume_Lag_7', 'Volume_Lag_14', 'Volume_Lag_30',
                'SMA10', 'SMA50', 'EMA12', 'EMA26', 'MACD', 'MACD_Signal', 'RSI', 
                'BB_Mid', 'BB_Upper', 'BB_Lower', 'Momentum']
    X = scaler.transform(coin_data[features].iloc[-1:].values)
    
    predictions = {}
    for horizon in ['1d', '2d', '1w', '1m']:
        prob = models[horizon].predict_proba(X)[0][1]
        predictions[horizon] = {
            'direction': 'Up' if prob > 0.5 else 'Down',
            'confidence': f"{prob * 100:.2f}%"
        }

    indicators = coin_data.iloc[-1][features].to_dict()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=coin_data['Date'], y=coin_data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=coin_data['Date'], y=coin_data['BB_Upper'], mode='lines', name='BB Upper', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=coin_data['Date'], y=coin_data['BB_Lower'], mode='lines', name='BB Lower', line=dict(dash='dash')))
    fig.update_layout(title=f'{coin} Price with Bollinger Bands (Last 30 Days)', xaxis_title='Date', yaxis_title='Price (USD)')
    chart = fig.to_html(full_html=False)

    return render_template('result.html', coin=coin, predictions=predictions, chart=chart, indicators=indicators)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
