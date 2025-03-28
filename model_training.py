import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

df = pd.read_csv('featured_data.csv')
features = ['Open', 'Close', 'Volume', 'Coin_Encoded', 
            'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_7', 'Close_Lag_14', 'Close_Lag_30',
            'Volume_Lag_1', 'Volume_Lag_2', 'Volume_Lag_7', 'Volume_Lag_14', 'Volume_Lag_30',
            'SMA10', 'SMA50', 'EMA12', 'EMA26', 'MACD', 'MACD_Signal', 'RSI', 
            'BB_Mid', 'BB_Upper', 'BB_Lower', 'Momentum']
X = df[features]
scaler = joblib.load('scaler.pkl')
X_scaled = scaler.transform(X)

horizons = {'1d': 1, '2d': 2, '1w': 7, '1m': 30}
models = {}

for horizon, shift in horizons.items():
    y = df[f'Target_{horizon}'].values
    train_size = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # XGBoost with tuning
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"{horizon} Test Accuracy: {accuracy:.4f}")
    print(f"{horizon} Test Accuracy: {accuracy:.4f}")
    
    models[horizon] = xgb_model
    joblib.dump(xgb_model, f'xgb_model_{horizon}.pkl')

joblib.dump(models, 'models.pkl')
