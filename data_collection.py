import yfinance as yf
import pandas as pd
from pycoingecko import CoinGeckoAPI

# Historical Data with Error Handling
coins = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'USDT-USD', 'BNB-USD']
data = {}
for coin in coins:
    try:
        data[coin] = yf.download(coin, start='2020-01-01', end='2025-03-24', progress=False)
        data[coin]['Coin'] = coin
        print(f"Downloaded {coin}: {data[coin].shape[0]} rows")
    except Exception as e:
        print(f"Failed to download {coin}: {e}")

# Combine and Save
if data:
    df = pd.concat(data.values(), ignore_index=True)
    df.to_csv('multi_coin_data.csv', index=False)
    print("Data shape:", df.shape)
    print(df.head())
else:
    print("No historical data collected.")

# Real-Time Data
cg = CoinGeckoAPI()
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
        print(f"Error fetching {coin} price: {e}")
print("Live Prices:", live_prices)
