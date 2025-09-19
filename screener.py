import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, inspect
import pandas_ta as ta
import joblib
import numpy as np
import json

# --- FUNGSI-FUNGSI BANTU ---
@st.cache_data
def load_data(ticker, timeframe='daily', db_file_path=None):
    _engine = create_engine(f"sqlite:///{db_file_path}")
    table_name = f"{ticker}_weekly" if timeframe == 'weekly' else ticker
    try:
        df = pd.read_sql(f"SELECT * FROM '{table_name}'", _engine, index_col='Date', parse_dates=['Date'])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def get_available_stocks(db_file_path):
    _engine = create_engine(f"sqlite:///{db_file_path}")
    inspector = inspect(_engine)
    stock_names = [name for name in inspector.get_table_names() if not name.endswith(('_weekly', '_sentiment')) and 'news' not in name and 'broker' not in name]
    return sorted(stock_names)

@st.cache_resource
def load_ai_model(ticker):
    model_filename = f'models/{ticker}_model.joblib'
    try:
        model = joblib.load(model_filename)
        return model
    except FileNotFoundError:
        return None

@st.cache_data
def load_optimal_params():
    try:
        with open('optimal_params.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def calculate_all_features(_df, _df_weekly, params):
    df = _df.copy()
    df_weekly = _df_weekly.copy()
    
    df_weekly_features = df_weekly[['SMA_20_weekly', 'RSI_14_weekly']]
    df_merged = pd.merge_asof(df, df_weekly_features, left_index=True, right_index=True)
    df = df_merged
    
    df.ta.rsi(length=params.get('rsi_length', 14), append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=params.get('bbands_length', 20), append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)
    df.ta.adx(length=14, append=True)
    df.fillna(0, inplace=True)
    return df
# Fungsi inti screener (diperbarui untuk menerima parameter risiko jangka panjang)
@st.cache_data(ttl=3600)
def run_screener(stock_list, db_file_path, atr_multiplier, risk_reward_ratio, rrr_long_term, _status_callback):
    _engine = create_engine(f"sqlite:///{db_file_path}")
    all_optimal_params = load_optimal_params()
    short_term_picks, long_term_picks = [], []

    for i, ticker in enumerate(stock_list):
        _status_callback(ticker, (i + 1) / len(stock_list))
        
        df_daily = load_data(ticker, timeframe='daily', db_file_path=db_file_path)
        df_weekly = load_data(ticker, timeframe='weekly', db_file_path=db_file_path)
        
        if len(df_daily) < 250 or len(df_weekly) < 52: continue

        df_weekly['SMA_20_weekly'] = df_weekly.ta.sma(length=20)
        df_weekly['RSI_14_weekly'] = df_weekly.ta.rsi(length=14)
        params = all_optimal_params.get(ticker, {'rsi_length': 14, 'bbands_length': 20})
        df = calculate_all_features(df_daily, df_weekly, params)
        model = load_ai_model(ticker)
        if model is None: continue

        required_features = model.feature_names_in_
        for col in required_features:
            if col not in df.columns: df[col] = 0
        X = df[required_features]
        df['Prediksi_Sinyal'] = model.predict(X)
        last_day = df.iloc[-1]
        
        # Logika Peringkat Jangka Pendek (Short Term)
        if last_day['Prediksi_Sinyal'] == 1:
            score = 0
            if last_day['ADX_14'] > 25 and last_day['DMP_14'] > last_day['DMN_14']: score += last_day['ADX_14']
            if last_day[f'RSI_{params.get("rsi_length", 14)}'] > 50: score += last_day[f'RSI_{params.get("rsi_length", 14)}']
            
            harga_beli = last_day['Close']
            atr_rupiah = last_day['ATRr_14']
            risiko = atr_multiplier * atr_rupiah
            harga_sl = harga_beli - risiko
            harga_tp = harga_beli + (risiko * risk_reward_ratio)

            short_term_picks.append({
                "Saham": ticker,
                "Harga Terakhir": last_day['Close'],
                "Rekomendasi AI": "BELI",
                "Harga Beli": harga_beli,
                "Harga Take Profit": harga_tp,
                "Harga Stop Loss": harga_sl,
                "Skor Rekomendasi": score
            })

        # Logika Peringkat Jangka Panjang (Long Term) - DIPERBARUI
        if last_day['Close'] > last_day['SMA_20_weekly'] and last_day['RSI_14_weekly'] > 55:
            harga_beli_long = last_day['Close']
            harga_sl_long = last_day['SMA_20_weekly']
            # Pastikan risiko tidak nol untuk menghindari pembagian dengan nol
            if (harga_beli_long - harga_sl_long) > 0:
                risiko_long = harga_beli_long - harga_sl_long
                harga_tp_long = harga_beli_long + (risiko_long * rrr_long_term)
                
                long_term_picks.append({
                    "Saham": ticker, "Harga Terakhir": last_day['Close'], "Rekomendasi Sistem": "TAHAN/BELI",
                    "Harga Beli": harga_beli_long, "Harga Take Profit": harga_tp_long, "Harga Stop Loss": harga_sl_long,
                    "RSI Mingguan": last_day['RSI_14_weekly']
                })
            
    df_short = pd.DataFrame(short_term_picks).sort_values(by="Skor Rekomendasi", ascending=False).head(10)
    df_long = pd.DataFrame(long_term_picks).sort_values(by="RSI Mingguan", ascending=False).head(10)
    
    return df_short, df_long