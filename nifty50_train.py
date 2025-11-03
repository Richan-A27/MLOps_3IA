# nifty50_train.py
# Trains NIFTY50 Linear Regression model and saves it as model.pkl

import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

print("📡 Fetching NIFTY50 data...")

nifty = yf.download("^NSEI", start="2020-01-01", end="2025-11-01", auto_adjust=True)

if isinstance(nifty.columns, pd.MultiIndex):
    nifty.columns = [col[0] for col in nifty.columns]

nifty["Prev_Close"] = nifty["Close"].shift(1)
nifty["Price_Change"] = nifty["Close"] - nifty["Prev_Close"]
nifty.dropna(inplace=True)

X = nifty[["Prev_Close", "Open", "High", "Low", "Volume"]]
y = nifty["Close"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")
