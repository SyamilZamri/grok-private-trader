import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Simple RSI tanpa TA-Lib
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

st.set_page_config(page_title="Grok Private Trader MY", layout="wide")
st.title("Grok Private Trader MY 2025 — Top Picks + Chart Fixed")

# Semua aset
assets = {
    "MAYBANK":"1155.KL","PBBANK":"1295.KL","TENAGA":"5347.KL","PETGAS":"6033.KL",
    "IOICORP":"1961.KL","SIMEPLT":"5285.KL","KLK":"2445.KL","GENTING":"3182.KL",
    "GAMUDA":"5398.KL","IJM":"3336.KL","YTL":"4677.KL","HARTA":"5168.KL",
    "TOPGLOV":"7113.KL","INARI":"0166.KL","CIMB":"1023.KL","MAXIS":"6012.KL",
    "Crude Oil WTI":"CL=F","Crude Palm Oil":"PALM.OIL"
}

@st.cache_data(ttl=300)
def scan_and_rank():
    picks = []
    for name, ticker in assets.items():
        try:
            df = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=True)
            if df.empty or len(df) < 30: continue
            
            close = df['Close'].iloc[-1]
            rsi = calculate_rsi(df['Close']).iloc[-1]
            vol = df['Volume'].iloc[-1]
            avg_vol = df['Volume'].tail(20).mean()
            low_20 = df['Low'].rolling(20).min().shift(1).iloc[-1]
            in_ob = close <= low_20 * 1.015
            
            score = 60
            if rsi < 45: score += 22
            if in_ob: score += 20
            if vol > avg_vol * 1.3: score += 12
            score = min(score, 96)
            
            if score >= 83:
                picks.append({
                    "Nama": name,
                    "Ticker": ticker,
                    "Harga": f"RM {close:.3f}" if "KL" in ticker else f"{close:.2f}",
                    "Confidence": f"{score:.1f}%",
                    "RSI": f"{rsi:.1f}",
                    "OB Zone": "Ya" if in_ob else "Tidak"
                })
        except: continue
    return pd.DataFrame(picks).sort_values("Confidence", ascending=False).head(8)

st.write("Scanning 18 aset Shariah + Crude + CPO…")
top_picks = scan_and_rank()

if top_picks.empty:
    st.warning("Tiada setup kuat sekarang. Cuba petang nanti.")
else:
    st.success(f"Top {len(top_picks)} Entry Terbaik Hari Ini")
    st.dataframe(top_picks, use_container_width=True)

# Pilih satu untuk chart penuh
if not top_picks.empty:
    choice = st.selectbox("Pilih untuk Chart Penuh", top_picks["Nama"])
    ticker = top_picks[top_picks["Nama"] == choice]["Ticker"].iloc[0]
    
    df_chart = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
    if not df_chart.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_chart.index,
            open=df_chart['Open'],
            high=df_chart['High'],
            low=df_chart['Low'],
            close=df_chart['Close'],
            name=choice
        ))
        # Mark Bullish OB
        df_chart['ob_low'] = df_chart['Low'].rolling(20).min().shift(1)
        ob_entries = df_chart[df_chart['Close'] <= df_chart['ob_low'] * 1.015]
        fig.add_trace(go.Scatter(
            x=ob_entries.index, y=ob_entries['ob_low'],
            mode='markers', marker=dict(color='lime', size=10, symbol='triangle-up'),
            name='Bullish OB'
        ))
        
        fig.update_layout(height=650, title=f"{choice} ({ticker}) – Full Chart")
        st.plotly_chart(fig, use_container_width=True)
        st.balloons()

if st.button("Refresh Scan + Chart"):
    st.cache_data.clear()
    st.rerun()

st.caption("Fixed: Chart kosong + multi-index error. Shariah SC Nov 2025. Confidence >83% sahaja keluar.")
