import pandas as pd
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
cg = CoinGeckoAPI()
coins = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'USDT-USD', 'BNB-USD']
coin_ids = {
    'BTC-USD': 'bitcoin', 'ETH-USD': 'ethereum', 'LTC-USD': 'litecoin',
    'USDT-USD': 'tether', 'BNB-USD': 'binancecoin'
}

def collect_historical_data():
    data = {}
    end_date = datetime(2025, 3, 24)
    start_date = end_date - timedelta(days=365)
    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())

    for coin, coin_id in coin_ids.items():
        try:
            history = cg.get_coin_market_chart_range_by_id(
                id=coin_id, vs_currency='usd', from_timestamp=start_ts, to_timestamp=end_ts
            )  # Default is daily for >90 days
            df_coin = pd.DataFrame(history['prices'], columns=['Date', 'Close'])
            df_coin['Date'] = pd.to_datetime(df_coin['Date'], unit='ms')
            df_coin['Open'] = df_coin['Close'].shift(1)
            df_coin['Volume'] = [v[1] for v in history['total_volumes']]
            df_coin['Coin'] = coin
            data[coin] = df_coin
            logging.info(f"Downloaded {coin}: {df_coin.shape[0]} rows")
        except Exception as e:
            logging.error(f"Failed to download {coin}: {e}")
    
    if data:
        df = pd.concat(data.values(), ignore_index=True)
        df.to_csv('multi_coin_data.csv', index=False)
        logging.info(f"Data shape: {df.shape}")
        print(df.head())
    else:
        logging.warning("No historical data collected.")

if __name__ == "__main__":
    collect_historical_data()
