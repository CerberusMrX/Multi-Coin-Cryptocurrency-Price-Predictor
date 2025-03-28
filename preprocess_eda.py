import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

df = pd.read_csv('multi_coin_data.csv')
logging.info(f"Initial shape: {df.shape}")

# Multi-horizon Targets (daily)
horizons = {'1d': 1, '2d': 2, '1w': 7, '1m': 30}
for name, shift in horizons.items():
    df[f'Target_{name}'] = df.groupby('Coin')['Close'].shift(-shift) > df['Close']
    df[f'Target_{name}'] = df[f'Target_{name}'].astype(int)

# Encode Coin
le = LabelEncoder()
df['Coin_Encoded'] = le.fit_transform(df['Coin'])
joblib.dump(le, 'label_encoder.pkl')

# Scale initial features
features = ['Open', 'Close', 'Volume', 'Coin_Encoded']
scaler = StandardScaler()
df_clean = df.dropna(subset=[f'Target_{name}' for name in horizons])
X = df_clean[features]
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

# EDA
print(df_clean.groupby('Coin').describe()['Close'])
df_clean.pivot(columns='Coin', values='Close').plot(title='Daily Closing Prices (Last 365 Days)')
plt.savefig('price_plot.png')
plt.close()
sns.countplot(x='Target_1d', hue='Coin', data=df_clean)
plt.title('1-Day Up/Down Distribution by Coin')
plt.savefig('target_dist.png')
plt.close()

df_clean.to_csv('processed_data.csv', index=False)
logging.info(f"Processed shape: {df_clean.shape}")
