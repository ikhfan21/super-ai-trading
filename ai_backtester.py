import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect
import matplotlib.pyplot as plt
import joblib
import pandas_ta as ta
import argparse
import os
import time
from datetime import datetime
import json
import streamlit as st # <-- PERBAIKAN: Menambahkan import yang hilang

# --- FUNGSI-FUNGSI BANTU ---
def get_available_models():
    """Mendapatkan daftar semua model yang tersedia di folder /models."""
    if not os.path.exists('models'):
        return []
    model_files = os.listdir('models')
    tickers = [f.replace('_model.joblib', '') for f in model_files if f.endswith('_model.joblib')]
    return sorted(tickers)

@st.cache_data
def load_optimal_params():
    """Memuat file parameter optimal."""
    try:
        with open('optimal_params.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def jalankan_ai_backtesting(ticker_symbol, engine, all_optimal_params, modal_awal=100_000_000, show_chart=True):
    """
    Menjalankan simulasi backtesting untuk satu saham dan mengembalikan hasilnya.
    """
    model_filename = f'models/{ticker_symbol}_model.joblib'
    
    # --- 1. MEMUAT DATA & MEMBUAT FITUR (SINKRON DENGAN TRAINER FINAL) ---
    try:
        df_daily = pd.read_sql(f"SELECT * FROM '{ticker_symbol}'", engine, index_col='Date', parse_dates=['Date'])
        if len(df_daily) < 250: return None
    except Exception:
        return None

    params = all_optimal_params.get(ticker_symbol, {'rsi_length': 14, 'bbands_length': 20})

    # Replikasi feature engineering dari trainer.py secara lengkap
    df_weekly = pd.read_sql(f"SELECT * FROM '{ticker_symbol}_weekly'", engine, index_col='Date', parse_dates=['Date'])
    df_weekly['SMA_20_weekly'] = df_weekly.ta.sma(length=20)
    df_weekly['RSI_14_weekly'] = df_weekly.ta.rsi(length=14)
    df_weekly_features = df_weekly[['SMA_20_weekly', 'RSI_14_weekly']]
    df = pd.merge_asof(df_daily, df_weekly_features, left_index=True, right_index=True)
    
    try:
        df_sentiment = pd.read_sql(f"SELECT * FROM news_sentiment WHERE ticker='{ticker_symbol}'", engine, parse_dates=['date'])
        if not df_sentiment.empty:
            sentiment_daily = df_sentiment.groupby(df_sentiment['date'].dt.date).agg(sentiment_sum=('sentiment', 'sum')).reset_index()
            sentiment_daily['date'] = pd.to_datetime(sentiment_daily['date'])
            sentiment_daily.set_index('date', inplace=True)
            df = df.merge(sentiment_daily, left_index=True, right_index=True, how='left')
    except:
        df['sentiment_sum'] = 0

    df.ta.rsi(length=params.get('rsi_length', 14), append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=params.get('bbands_length', 20), append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)
    df.ta.adx(length=14, append=True)
    df.ta.cdl_pattern(name="all", append=True)
    df.ta.pivots(append=True)
    pivot_levels_raw = ['PIVOTS_TRAD_D_P','PIVOTS_TRAD_D_S1','PIVOTS_TRAD_D_R1','PIVOTS_TRAD_D_S2','PIVOTS_TRAD_D_R2']
    pivot_levels_simple = ['p','s1','r1','s2','r2']
    rename_dict = {old: new for old, new in zip(pivot_levels_raw, pivot_levels_simple)}
    df.rename(columns=rename_dict, inplace=True)
    df.fillna(0, inplace=True)
    
    # --- 2. MEMUAT MODEL AI & MEMBUAT PREDIKSI ---
    try:
        model = joblib.load(model_filename)
    except FileNotFoundError:
        return None

    required_features = model.feature_names_in_
    for col in required_features:
        if col not in df.columns: df[col] = 0
    X = df[required_features]
    df['Sinyal'] = model.predict(X)
    # --- 3. SIMULASI BACKTESTING ---
    kas = modal_awal
    jumlah_saham = 0
    nilai_portfolio_harian = []
    
    for i in range(len(df)):
        harga_hari_ini = df['Close'].iloc[i]
        if df['Sinyal'].iloc[i] == 1 and kas > 0:
            jumlah_saham = kas / harga_hari_ini
            kas = 0
        elif df['Sinyal'].iloc[i] == 0 and jumlah_saham > 0:
            kas = jumlah_saham * harga_hari_ini
            jumlah_saham = 0
        nilai_portfolio_harian.append(kas + (jumlah_saham * harga_hari_ini))
        
    df['Nilai_Portfolio'] = nilai_portfolio_harian

    # --- 4. HASIL ---
    nilai_akhir = df['Nilai_Portfolio'].iloc[-1]
    total_return_pct = ((nilai_akhir - modal_awal) / modal_awal) * 100
    
    df_filtered = df[df.index >= df.index[0]]
    buy_and_hold_return_pct = ((df_filtered['Close'].iloc[-1] - df_filtered['Close'].iloc[0]) / df_filtered['Close'].iloc[0]) * 100
    
    hasil = {
        "ticker": ticker_symbol,
        "ai_return_pct": total_return_pct,
        "bh_return_pct": buy_and_hold_return_pct,
        "alpha_pct": total_return_pct - buy_and_hold_return_pct,
        "nilai_akhir": nilai_akhir
    }
    
    if show_chart:
        print(f"\n--- HASIL AKHIR BACKTESTING (AI) UNTUK {ticker_symbol} ---")
        print(f"Modal Awal          : Rp {modal_awal:,.0f}")
        print(f"Nilai Akhir Portfolio : Rp {nilai_akhir:,.0f}")
        print(f"Total Return Strategi AI: {total_return_pct:.2f}%")
        print(f"Total Return Buy & Hold : {buy_and_hold_return_pct:.2f}%")
        
        plt.style.use('dark_background')
        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df['Nilai_Portfolio'], label='Kinerja Strategi AI', color='purple')
        plt.plot(df.index, (df['Close'] / df['Close'].iloc[0]) * modal_awal, label='Kinerja Buy & Hold', color='gray', linestyle='--')
        plt.title(f'Kinerja Strategi AI vs. Buy & Hold pada {ticker_symbol}', fontsize=16)
        plt.legend()
        plt.grid(True)
        plt.show()
        
    return hasil

# --- BAGIAN EKSEKUSI UTAMA (DENGAN MODE GANDA) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtester Model AI (Mode Pabrik & Spesialis).")
    parser.add_argument("--tickers", nargs='+', help="(Opsional) Daftar ticker spesifik yang akan di-backtest (contoh: BBCA.JK ASII.JK)")
    args = parser.parse_args()
    
    db_file_path = "sqlite:///data_saham.db"
    engine = create_engine(db_file_path)
    
    all_optimal_params = load_optimal_params()

    tickers_to_process = []
    
    if args.tickers:
        tickers_to_process = [ticker.upper() for ticker in args.tickers]
        print(f"--- MENJALANKAN DALAM MODE SPESIALIS UNTUK {len(tickers_to_process)} SAHAM ---")
        
        for ticker in tickers_to_process:
            model_file = f'models/{ticker}_model.joblib'
            if not os.path.exists(model_file):
                print(f"Error: File model '{model_file}' tidak ditemukan. Jalankan 'trainer.py --ticker {ticker}' terlebih dahulu.")
                continue
            jalankan_ai_backtesting(ticker, engine, all_optimal_params, show_chart=True)

    else:
        tickers_to_process = get_available_models()
        print(f"--- MENJALANKAN DALAM MODE PABRIK UNTUK {len(tickers_to_process)} MODEL YANG DITEMUKAN ---")
        
        semua_hasil = []
        start_time = time.time()
        
        for i, ticker in enumerate(tickers_to_process):
            print(f"({i+1}/{len(tickers_to_process)}) Backtesting: {ticker}", end='\r')
            
            hasil = jalankan_ai_backtesting(ticker, engine, all_optimal_params, show_chart=False)
            
            if hasil:
                semua_hasil.append(hasil)
        
        if semua_hasil:
            df_hasil = pd.DataFrame(semua_hasil)
            df_hasil.sort_values(by='alpha_pct', ascending=False, inplace=True)
            
            output_filename = 'hasil_backtest_semua.csv'
            df_hasil.to_csv(output_filename, index=False, float_format='%.2f')
            
            end_time = time.time()
            total_waktu_menit = (end_time - start_time) / 60
            
            print("\n\n" + "="*54)
            print("--- PROSES BACKTESTING MASSAL SELESAI ---")
            print(f"Laporan Akhir:")
            print(f"Total waktu            : {total_waktu_menit:.2f} menit")
            print(f"Total saham diuji      : {len(semua_hasil)}")
            print("\n--- 20 SAHAM DENGAN KINERJA AI TERBAIK (vs. Buy & Hold) ---")
            print(df_hasil.head(20).to_string())
            print(f"\nLaporan peringkat lengkap tersimpan di file: '{output_filename}'")
            print("="*54)

            with open('logs/ai_backtester_last_run.log', 'w') as f:
                f.write(datetime.now().isoformat())
            print("\nStempel waktu backtesting berhasil dicatat.")
        else:
            print("Tidak ada hasil backtest yang bisa diproses.")