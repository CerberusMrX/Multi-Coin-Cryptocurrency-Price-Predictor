import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

df = pd.read_csv('processed_data.csv')
logging.info(f"Initial shape: {df.shape}")

# Time-lagged features (days)
lags = [1, 2, 7, 14, 30]
for lag in lags:
    df[f'Close_Lag_{lag}'] = df.groupby('Coin')['Close'].shift(lag)
    df[f'Volume_Lag_{lag}'] = df.groupby('Coin')['Volume'].shift(lag)

# Technical Indicators
df['SMA10'] = df.groupby('Coin')['Close'].rolling(10).mean().reset_index(0, drop=True)
df['SMA50'] = df.groupby('Coin')['Close'].rolling(50).mean().reset_index(0, drop=True)
df['EMA12'] = df.groupby('Coin')['Close'].ewm(span=12).mean().reset_index(0, drop=True)
df['EMA26'] = df.groupby('Coin')['Close'].ewm(span=26).mean().reset_index(0, drop=True)
df['MACD'] = df['EMA12'] - df['EMA26']
df['MACD_Signal'] = df.groupby('Coin')['MACD'].ewm(span=9).mean().reset_index(0, drop=True)
df['RSI'] = df.groupby('Coin')['Close'].apply(lambda x: 100 - (100 / (1 + (x.diff().where(lambda x: x > 0, 0).rolling(14).mean() / -x.diff().where(lambda x: x < 0, 0).rolling(14).mean())))).reset_index(0, drop=True)
df['BB_Mid'] = df.groupby('Coin')['Close'].rolling(20).mean().reset_index(0, drop=True)
df['BB_Std'] = df.groupby('Coin')['Close'].rolling(20).std().reset_index(0, drop=True)
df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']
df['Momentum'] = df.groupby('Coin')['Close'].diff(10).reset_index(0, drop=True)

df = df.dropna()
logging.info(f"Shape after feature engineering: {df.shape}")

features = ['Open', 'Close', 'Volume', 'Coin_Encoded', 
            'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_7', 'Close_Lag_14', 'Close_Lag_30',
            'Volume_Lag_1', 'Volume_Lag_2', 'Volume_Lag_7', 'Volume_Lag_14', 'Volume_Lag_30',
            'SMA10', 'SMA50', 'EMA12', 'EMA26', 'MACD', 'MACD_Signal', 'RSI', 
            'BB_Mid', 'BB_Upper', 'BB_Lower', 'Momentum']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

df.to_csv('featured_data.csv', index=False)
logging.info(f"Featured shape: {df.shape}")
