import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import talib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Grok Private Trader 2025", layout="wide")
st.title("Grok Private Trader 2025 — Personal Only")
st.sidebar.header("Pilih Aset & Setting")

ticker = st.sidebar.selectbox("Aset", ["1155.KL", "5183.KL", "PETGAS.KL", "CL=F", "BZ=F", "PALM.OIL"])
timeframe = st.sidebar.selectbox("Timeframe", ["1h", "4h", "1d"])
min_conf = st.sidebar.slider("Min Confidence %", 70, 95, 85)

@st.cache_data(ttl=300)
def get_data(tck, tf):
    return yf.download(tck, period="6mo", interval=tf)

data = get_data(ticker, timeframe)
if data.empty:
    st.error("Tiada data")
    st.stop()

def train_model(df):
    df = df.copy()
    df['rsi'] = talib.RSI(df['Close'], 14)
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], 14)
    df['fvg'] = ((df['Low'] > df['High'].shift(2)) | (df['High'] < df['Low'].shift(2))).astype(int)
    df['target'] = (df['Close'].shift(-5) > df['Close'] * 1.03).astype(int)
    df = df.dropna()
    if len(df) < 50:
        return None, 50
    X = df[['rsi','atr','fvg']]
    y = df['target']
    model = XGBClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X, y)
    acc = model.score(X, y)
    return model, acc*100

model, accuracy = train_model(data)

latest = data.iloc[-1]
current_price = latest['Close']

if model is not None:
    prob = model.predict_proba(data[['rsi','atr','fvg']].dropna())[-1][1] * 100
    signal = "BUY" if prob >= min_conf else "HOLD"
else:
    prob = 50
    signal = "HOLD"

col1, col2 = st.columns([3,1])
with col1:
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                    open=data['Open'], high=data['High'],
                    low=data['Low'], close=data['Close'])])
    fig.update_layout(height=600, title=f"{ticker} – Live Chart")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric("Harga Sekarang", f"{current_price:.2f}")
    st.metric("AI Signal", signal, delta=f"{prob:.1f}% confidence")
    st.metric("ML Accuracy", f"{accuracy:.1f}%")
    if signal == "BUY":
        st.success("Masuk trade sekarang!")
    else:
        st.info("Tunggu setup lebih baik")

if st.button("Refresh Live Data"):
    st.rerun()  
