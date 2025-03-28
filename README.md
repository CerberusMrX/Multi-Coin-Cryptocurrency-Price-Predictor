# Advanced Multi-Coin Cryptocurrency Price Movement Predictor

![home](https://github.com/user-attachments/assets/8d689ca7-c9f8-43ab-83fc-3ea03799af5e)

![pre](https://github.com/user-attachments/assets/d497e97b-98ce-4ac8-b907-6162d6e5b91e)

A tool that predicts if BTC, ETH, LTC, USDT, and BNB prices will rise or fall in 1 day, 2 days, 1 week, or 1 month, using XGBoost on a year of data with 60–75% accuracy. It shows live prices and forecasts on a Flask website with charts.

## About
This project predicts price movements for five cryptocurrencies—Bitcoin (BTC), Ethereum (ETH), Litecoin (LTC), Tether (USDT), and Binance Coin (BNB)—over four time periods. Built for my [Course Name/Code] at the National Institute of Business Management (NIBM), supervised by Mr. Ravilal, it uses machine learning and a web app. Submitted by [Your Full Name] on March 26, 2025.

## Features
- **Data Collection**: Grabs 1 year of daily prices (March 2024–2025) from CoinGecko for BTC, ETH, LTC, USDT, and BNB.
- **Data Processing**: Cleans data, adds up/down targets for 1d, 2d, 1w, 1m, and scales numbers for better predictions.
- **Feature Engineering**: Adds 25 features like past prices, simple moving averages (SMA), RSI, Bollinger Bands, and momentum.
- **Model Training**: Tests 6 algorithms (Random Forest, Logistic Regression, Neural Network, SVM, Decision Tree, XGBoost), picks XGBoost with 60–75% accuracy.
- **Website Interface**: Flask app shows live prices in animated cards and predictions with a 30-day interactive chart.
- **Cross-Platform**: Runs on Kali Linux, Windows, and Jupyter Notebook.

## Prerequisites
- Python 3.13 (or 3.x)
- Git (to clone the repo)
- A web browser (e.g., Chrome)

## How to Install and Run

**Clone the Repository**
    
   Open a terminal and run:  
python3 -m venv crypto_venv
source crypto_venv/bin/activate
pip install -r requirements.txt
python data_collection_cg.py
python preprocess_eda.py
python feature_engineering.py
python model_training.py
python app.py

   

   
   
