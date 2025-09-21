import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import argparse
import time
import json
from datetime import datetime
from github_sync import sync_to_github # <-- Impor kurir kita

# --- FUNGSI-FUNGSI BANTU ---
def get_available_stocks(engine):
    """Mendapatkan daftar semua tabel (saham) dari database."""
    inspector = inspect(engine)
    stock_names = [name for name in inspector.get_table_names() if not name.endswith(('_weekly', '_sentiment')) and 'news' not in name and 'broker' not in name]
    return sorted(stock_names)

def train_model_for_ticker(ticker_symbol, engine, all_optimal_params):
    """
    Fungsi untuk menjalankan seluruh proses training untuk satu ticker
    menggunakan parameter yang sudah dioptimasi.
    """
    model_filename = f'models/{ticker_symbol}_model.joblib'

    default_params = {
        'rsi_length': 14, 'bbands_length': 20, 'n_estimators': 100,
        'max_depth': 20, 'min_samples_leaf': 1
    }
    params = all_optimal_params.get(ticker_symbol, default_params)
    
    try:
        df_daily = pd.read_sql(f"SELECT * FROM '{ticker_symbol}'", engine, index_col='Date', parse_dates=['Date'])
        if len(df_daily) < 250:
            # print(f"-> Data untuk {ticker_symbol} tidak cukup panjang ({len(df_daily)} baris). Melewati.")
            return False, None
    except Exception as e:
        # print(f"-> Gagal memuat data harian untuk {ticker_symbol}. Error: {e}")
        return False, None

    # --- REKAYASA FITUR (LENGKAP) ---
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
    for level in pivot_levels_simple:
        if level in df.columns: df[f'Jarak_ke_{level.upper()}'] = (df['Close'] - df[level]) / df['Close']
    for level in ['p', 's1', 'r1']:
        if level in df.columns: df[f'Posisi_vs_{level.upper()}'] = np.where(df['Close'] > df[level], 1, 0)
    
    future_period = 5; profit_threshold = 0.02
    df['Target'] = np.where(df['Close'].shift(-future_period) > df['Close'] * (1 + profit_threshold), 1, 0)
    df.fillna(0, inplace=True)
    
    kolom_non_fitur = [col for col in df.columns if col in pivot_levels_simple + ['Target']]
    X = df.drop(columns=kolom_non_fitur)
    y = df['Target']

    if len(X) < 100:
        # print(f"-> Data untuk {ticker_symbol} tidak cukup setelah diproses. Melewati.")
        return False, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # --- MELATIH & MENYIMPAN MODEL ---
    model = RandomForestClassifier(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 20),
        min_samples_leaf=params.get('min_samples_leaf', 1),
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    if os.path.exists(model_filename): os.remove(model_filename)
    joblib.dump(model, model_filename)
    
    # Evaluasi dan kembalikan rapornya
    predictions = model.predict(X_test)
    
    # Menangani kasus di mana data tes hanya memiliki satu kelas
    if len(np.union1d(y_test.unique(), predictions)) < 2:
        report = {"accuracy": accuracy_score(y_test, predictions)}
    else:
        report = classification_report(
            y_test, 
            predictions, 
            target_names=['Bukan Peluang (0)', 'Peluang Bagus (1)'], 
            output_dict=True, 
            labels=[0, 1],
            zero_division=0
        )
    
    return True, report

# --- BAGIAN EKSEKUSI UTAMA (DENGAN AUTO-SYNC) ---
if __name__ == "__main__":
    # Buat folder 'models' jika belum ada
    if not os.path.exists('models'):
        os.makedirs('models')
        
    parser = argparse.ArgumentParser(description="Trainer Model AI.")
    parser.add_argument("--tickers", nargs='+', help="Daftar ticker spesifik yang akan dilatih.")
    args = parser.parse_args()
    
    db_file_path = "sqlite:///data_saham.db"
    engine = create_engine(db_file_path)
    
    try:
        with open('optimal_params.json', 'r') as f:
            all_optimal_params = json.load(f)
        print("Buku resep 'optimal_params.json' berhasil dimuat.")
    except FileNotFoundError:
        all_optimal_params = {}
        print("PERINGATAN: File 'optimal_params.json' tidak ditemukan. Parameter default akan digunakan.")

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
        
        sukses, rapor = train_model_for_ticker(ticker, engine, all_optimal_params)
        
        if sukses:
            sukses_count += 1
            f1_score_1 = rapor.get('Peluang Bagus (1)', {}).get('f1-score', 0)
            print(f"-> {ticker} BERHASIL dilatih. F1-Score (Peluang Bagus): {f1_score_1:.2f}")
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
    
    # --- PANGGIL "KURIR" UNTUK SINKRONISASI OTOMATIS ---
    if sukses_count > 0: # Hanya sync jika ada model baru yang berhasil dilatih
        sync_to_github(f"Auto-sync: Latih ulang {sukses_count} model AI")