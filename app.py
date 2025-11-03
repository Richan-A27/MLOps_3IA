# nifty50_app.py
# Streamlit web app for NIFTY50 close price prediction

import streamlit as st
import joblib
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="NIFTY50 Predictor", page_icon="📈", layout="centered")

st.title("📊 NIFTY50 Stock Market Predictor")
st.markdown("Predict the next day's **closing price** of the NIFTY50 index using historical data and a simple ML model.")

# Load trained model
model = joblib.load("model.pkl")

# Fetch latest data
nifty = yf.download("^NSEI", period="5d", interval="1d", auto_adjust=True)
latest = nifty.iloc[-1]

st.subheader("🗓️ Latest Market Data")
st.dataframe(latest.to_frame().T.style.highlight_max(axis=1))

st.markdown("---")

st.subheader("🧮 Enter Prediction Inputs")

with st.form("predict_form"):
    prev_close = st.number_input("Previous Close (₹)", value=float(latest["Close"]), step=1.0)
    open_price = st.number_input("Open (₹)", value=float(latest["Open"]), step=1.0)
    high = st.number_input("High (₹)", value=float(latest["High"]), step=1.0)
    low = st.number_input("Low (₹)", value=float(latest["Low"]), step=1.0)
    volume = st.number_input("Volume", value=float(latest["Volume"]), step=1000000.0)

    submitted = st.form_submit_button("🔮 Predict")

if submitted:
    features = np.array([[prev_close, open_price, high, low, volume]])
    prediction = model.predict(features)[0]

    st.success(f"📈 **Predicted Next Day Close:** ₹{prediction:.2f}")
    st.caption("This prediction is based on a linear regression model trained on past NIFTY50 data.")

    # Simple visualization
    fig, ax = plt.subplots(figsize=(8,4))
    st.markdown("### 📉 Historical NIFTY50 Trend")
    nifty['Close'].tail(30).plot(ax=ax, linewidth=2, label='Last 30 Days Close')
    ax.axhline(prediction, color='orange', linestyle='--', label='Predicted Close')
    ax.legend()
    st.pyplot(fig)
