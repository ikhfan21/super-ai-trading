import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect
import pandas_ta as ta
# Impor Regressor dan Classifier
from sklearn.model_selection import train_test_split # Tambahkan ini
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import joblib
import os
import argparse
import time
import json
from datetime import datetime

# --- FUNGSI-FUNGSI BANTU ---
# (Fungsi get_available_stocks tetap sama)
def get_available_stocks(engine):
    inspector = inspect(engine)
    stock_names = [name for name in inspector.get_table_names() if not name.endswith(('_weekly', '_sentiment')) and 'news' not in name and 'broker' not in name]
    return sorted(stock_names)

def train_model_for_ticker(ticker_symbol, engine, all_optimal_params):
    """
    Fungsi untuk melatih TIGA model (Arah, Stop Loss, Take Profit) untuk satu ticker.
    """
    # Nama file model sekarang ada 3
    model_arah_filename = f'models/{ticker_symbol}_arah_model.joblib'
    model_sl_filename = f'models/{ticker_symbol}_sl_model.joblib'
    model_tp_filename = f'models/{ticker_symbol}_tp_model.joblib'

    default_params = {
        'rsi_length': 14, 'bbands_length': 20, 'n_estimators': 100,
        'max_depth': 20, 'min_samples_leaf': 1
    }
    params = all_optimal_params.get(ticker_symbol, default_params)
    
    try:
        df_daily = pd.read_sql(f"SELECT * FROM '{ticker_symbol}'", engine, index_col='Date', parse_dates=['Date'])
        if len(df_daily) < 250:
            print(f"-> Data untuk {ticker_symbol} tidak cukup. Melewati.")
            return False
    except Exception as e:
        print(f"-> Gagal memuat data harian untuk {ticker_symbol}. Error: {e}")
        return False

    # --- REKAYASA FITUR (LENGKAP) ---
    # Logika ini tetap sama, karena fiturnya digunakan oleh ketiga model
    df_weekly = pd.read_sql(f"SELECT * FROM '{ticker_symbol}_weekly'", engine, index_col='Date', parse_dates=['Date'])
    df_weekly['SMA_20_weekly'] = df_weekly.ta.sma(length=20)
    df_weekly['RSI_14_weekly'] = df_weekly.ta.rsi(length=14)
    df_weekly_features = df_weekly[['SMA_20_weekly', 'RSI_14_weekly']]
    df = pd.merge_asof(df_daily, df_weekly_features, left_index=True, right_index=True)
    
    # (Semua feature engineering lainnya sama persis)
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
    for level in pivot_levels_simple:
        if level in df.columns: df[f'Jarak_ke_{level.upper()}'] = (df['Close'] - df[level]) / df['Close']
    for level in ['p', 's1', 'r1']:
        if level in df.columns: df[f'Posisi_vs_{level.upper()}'] = np.where(df['Close'] > df[level], 1, 0)
    try:
        df_sentiment = pd.read_sql(f"SELECT * FROM news_sentiment WHERE ticker='{ticker_symbol}'", engine, parse_dates=['date'])
        if not df_sentiment.empty:
            sentiment_daily = df_sentiment.groupby(df_sentiment['date'].dt.date).agg(sentiment_sum=('sentiment', 'sum')).reset_index()
            sentiment_daily['date'] = pd.to_datetime(sentiment_daily['date'])
            sentiment_daily.set_index('date', inplace=True)
            df = df.merge(sentiment_daily, left_index=True, right_index=True, how='left')
    except:
        df['sentiment_sum'] = 0
    
    # --- MEMBUAT TIGA TARGET VARIABLE BARU ---
    future_period = 5
    profit_threshold = 0.02
    
    # Target 1: Arah (Classifier)
    df['Target_Arah'] = np.where(df['Close'].shift(-future_period) > df['Close'] * (1 + profit_threshold), 1, 0)
    
    # Target 2: Harga Terendah (Regressor untuk Stop Loss)
    df['Target_SL'] = df['Low'].rolling(window=future_period).min().shift(-future_period)
    
    # Target 3: Harga Tertinggi (Regressor untuk Take Profit)
    df['Target_TP'] = df['High'].rolling(window=future_period).max().shift(-future_period)
    
    df.fillna(0, inplace=True)
        # --- MEMISAHKAN FITUR DAN TIGA TARGET BERBEDA ---
    print("\nMemisahkan data menjadi Fitur (X) dan 3 Target (y)...")
    
    kolom_non_fitur = [col for col in df.columns if col in pivot_levels_simple + ['Harga_Masa_Depan', 'Perubahan_Harga', 'Target_Arah', 'Target_SL', 'Target_TP']]
    X = df.drop(columns=kolom_non_fitur)
    y_arah = df['Target_Arah']
    y_sl = df['Target_SL']
    y_tp = df['Target_TP']

    # --- MEMBAGI DATA UNTUK SETIAP TARGET ---
    # Ukuran test set tetap sama untuk semua
    X_train, X_test, y_train_arah, y_test_arah = train_test_split(X, y_arah, test_size=0.2, shuffle=False)
    _, _, y_train_sl, y_test_sl = train_test_split(X, y_sl, test_size=0.2, shuffle=False)
    _, _, y_train_tp, y_test_tp = train_test_split(X, y_tp, test_size=0.2, shuffle=False)
    
    # --- MELATIH & MENYIMPAN TIGA MODEL ---
    # Model 1: Arah (Classifier)
    print("Melatih Model Arah...")
    model_arah = RandomForestClassifier(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 20),
        min_samples_leaf=params.get('min_samples_leaf', 1),
        random_state=42, n_jobs=-1
    )
    model_arah.fit(X_train, y_train_arah)
    if os.path.exists(model_arah_filename): os.remove(model_arah_filename)
    joblib.dump(model_arah, model_arah_filename)
    print("-> Model Arah berhasil dilatih dan disimpan.")

    # Model 2: Stop Loss (Regressor)
    print("Melatih Model Stop Loss...")
    model_sl = RandomForestRegressor(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 20),
        min_samples_leaf=params.get('min_samples_leaf', 1),
        random_state=42, n_jobs=-1
    )
    model_sl.fit(X_train, y_train_sl)
    if os.path.exists(model_sl_filename): os.remove(model_sl_filename)
    joblib.dump(model_sl, model_sl_filename)
    print("-> Model Stop Loss berhasil dilatih dan disimpan.")

    # Model 3: Take Profit (Regressor)
    print("Melatih Model Take Profit...")
    model_tp = RandomForestRegressor(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 20),
        min_samples_leaf=params.get('min_samples_leaf', 1),
        random_state=42, n_jobs=-1
    )
    model_tp.fit(X_train, y_train_tp)
    if os.path.exists(model_tp_filename): os.remove(model_tp_filename)
    joblib.dump(model_tp, model_tp_filename)
    print("-> Model Take Profit berhasil dilatih dan disimpan.")
    
    return True

# --- BAGIAN EKSEKUSI UTAMA (DENGAN MODE GANDA) ---
if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
        
    parser = argparse.ArgumentParser(description="Trainer Model AI (Arah, SL, TP).")
    parser.add_argument("--tickers", nargs='+', help="Daftar ticker spesifik yang akan dilatih.")
    args = parser.parse_args()
    
    engine = create_engine("sqlite:///data_saham.db")
    
    try:
        with open('optimal_params.json', 'r') as f:
            all_optimal_params = json.load(f)
        print("Buku resep 'optimal_params.json' berhasil dimuat.")
    except FileNotFoundError:
        all_optimal_params = {}
        print("PERINGATAN: File 'optimal_params.json' tidak ditemukan.")

    tickers_to_process = []
    
    if args.tickers:
        tickers_to_process = [ticker.upper() for ticker in args.tickers]
        print(f"--- MENJALANKAN DALAM MODE SPESIALIS UNTUK {len(tickers_to_process)} SAHAM ---")
    else:
        tickers_to_process = get_available_stocks(engine)
        print(f"--- MENJALANKAN DALAM MODE PABRIK UNTUK {len(tickers_to_process)} SAHAM ---")

    sukses_count = 0
    gagal_count = 0
    start_time = time.time()
    
    for i, ticker in enumerate(tickers_to_process):
        print(f"\n({i+1}/{len(tickers_to_process)}) Memproses: {ticker}")
        sukses = train_model_for_ticker(ticker, engine, all_optimal_params)
        
        if sukses:
            sukses_count += 1
            print(f"-> {ticker} BERHASIL melatih 3 model.")
        else:
            gagal_count += 1

    end_time = time.time()
    total_waktu_menit = (end_time - start_time) / 60
    
    print("\n\n--- PROSES PELATIHAN MASSAL SELESAI ---")
    print("="*54)
    print(f"Laporan Akhir:")
    print(f"Total waktu            : {total_waktu_menit:.2f} menit")
    print(f"Total saham diproses   : {len(tickers_to_process)}")
    print(f"Berhasil dilatih       : {sukses_count}")
    print(f"Gagal dilatih          : {gagal_count}")
    
    with open('logs/trainer_last_run.log', 'w') as f:
        f.write(datetime.now().isoformat())
    print("\nStempel waktu training berhasil dicatat.")
    print("="*54)
