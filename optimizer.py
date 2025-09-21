import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import json
from itertools import product
import time
import os
import argparse
from datetime import datetime
from github_sync import sync_to_github # <-- Impor kurir kita

# --- FUNGSI UNTUK MEMPERSIAPKAN DATA DENGAN PARAMETER DINAMIS ---
def prepare_features_and_target(df, params):
    """
    Fungsi ini membuat fitur dan target berdasarkan parameter yang diberikan.
    """
    df_copy = df.copy()

    # Buat fitur teknikal dengan parameter dari 'params'
    df_copy.ta.rsi(length=params['rsi_length'], append=True)
    df_copy.ta.bbands(length=params['bbands_length'], append=True)
    df_copy.ta.atr(length=14, append=True)
    df_copy.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    # Membuat target variable
    future_period = 5
    profit_threshold = 0.02
    df_copy['Target'] = np.where(df_copy['Close'].shift(-future_period) > df_copy['Close'] * (1 + profit_threshold), 1, 0)
    
    df_copy.fillna(0, inplace=True)
    
    # Hanya pilih fitur yang relevan untuk optimasi
    feature_columns = [
        'Close', 'High', 'Low', 'Open', 'Volume', 
        f'RSI_{params["rsi_length"]}', 
        f'BBM_{params["bbands_length"]}_2.0_2.0',
        'ATRr_14', 
        'MACD_12_26_9'
    ]
    
    for col in feature_columns:
        if col not in df_copy.columns:
            return pd.DataFrame(), pd.Series()

    X = df_copy[feature_columns]
    y = df_copy['Target']
    
    return X, y

# --- BAGIAN EKSEKUSI UTAMA ---
if __name__ == "__main__":
    start_time = time.time()
    engine = create_engine("sqlite:///data_saham.db")
    
    # Siapkan argumen parser
    parser = argparse.ArgumentParser(description="Hyperparameter Optimizer untuk Model AI Saham.")
    parser.add_argument("--tickers", nargs='+', help="(Opsional) Daftar ticker spesifik yang akan dioptimasi.")
    args = parser.parse_args()

    tickers_to_process = []
    
    # Tentukan mode kerja
    if args.tickers:
        tickers_to_process = [ticker.upper() for ticker in args.tickers]
        print(f"--- MENJALANKAN DALAM MODE TARGET UNTUK {len(tickers_to_process)} SAHAM ---")
    else:
        tickers_to_process = ['BBCA.JK']
        print(f"--- MENJALANKAN DALAM MODE CONTOH UNTUK {tickers_to_process[0]} ---")
        print("Untuk menjalankan lebih dari satu saham, gunakan argumen --tickers.")

    # 1. DEFINISIKAN "MENU" PARAMETER (PARAMETER GRID)
    param_grid = {
        'rsi_length': [10, 14, 21],
        'bbands_length': [15, 20, 30],
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_leaf': [1, 5]
    }
    
    keys, values = zip(*param_grid.items())
    all_param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    total_combinations = len(all_param_combinations)
    print(f"Total kombinasi parameter per saham: {total_combinations}")
    
    output_file = 'optimal_params.json'
    
    try:
        with open(output_file, 'r') as f:
            all_best_params = json.load(f)
        print(f"File '{output_file}' ditemukan. Melanjutkan dan akan memperbarui jika ditemukan hasil lebih baik.")
    except FileNotFoundError:
        all_best_params = {}
        # --- LOOPING UTAMA UNTUK SEMUA SAHAM YANG DIPILIH ---
    saham_yang_dioptimasi_kali_ini = []
    
    for i, ticker in enumerate(tickers_to_process):
        
        # FITUR RESUME: Lewati saham yang sudah ada di file JSON
        if ticker in all_best_params and 'error' not in all_best_params.get(ticker, {}):
            print(f"\n({i+1}/{len(tickers_to_process)}) {ticker} sudah dioptimasi sebelumnya. Melewati.")
            continue

        print(f"\n({i+1}/{len(tickers_to_process)}) Memulai optimasi untuk: {ticker}")
        
        try:
            raw_df = pd.read_sql(f"SELECT * FROM '{ticker}'", engine, index_col='Date', parse_dates=['Date'])
            if len(raw_df) < 250:
                print(f"-> Data untuk {ticker} tidak cukup panjang. Melewati.")
                all_best_params[ticker] = {'error': 'data tidak cukup'}
                continue
        except Exception as e:
            print(f"-> Gagal memuat data untuk {ticker}. Error: {e}")
            all_best_params[ticker] = {'error': f'gagal memuat data: {e}'}
            continue

        best_score_for_ticker = -1
        best_params_for_ticker = None
        
        # Loop untuk setiap kombinasi parameter
        for j, params in enumerate(all_param_combinations):
            
            X, y = prepare_features_and_target(raw_df, params)

            if X.empty: continue
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            if len(X_train) == 0 or len(X_test) == 0: continue

            model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_leaf=params['min_samples_leaf'],
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_test)
            score = f1_score(y_test, predictions, pos_label=1, zero_division=0)
            
            if score > best_score_for_ticker:
                best_score_for_ticker = score
                best_params_for_ticker = params
        
        if best_params_for_ticker:
            print(f"-> 'Resep Emas' ditemukan untuk {ticker} dengan skor F1: {best_score_for_ticker:.4f}")
            all_best_params[ticker] = best_params_for_ticker
            saham_yang_dioptimasi_kali_ini.append(ticker)
        else:
            print(f"-> Tidak ditemukan parameter yang valid untuk {ticker}.")
            all_best_params[ticker] = {'error': 'tidak ada parameter valid'}

        # Simpan hasil ke file JSON setiap kali satu saham selesai, untuk keamanan
        with open(output_file, 'w') as f:
            json.dump(all_best_params, f, indent=4)
        print(f"-> Hasil untuk {ticker} disimpan ke '{output_file}'.")


    # --- LAPORAN AKHIR ---
    print("\n\n--- PROSES OPTIMASI SELESAI ---")
    end_time = time.time()
    total_waktu_menit = (end_time - start_time) / 60
    
    print("======================================================")
    print(f"Laporan Akhir:")
    print(f"Total waktu            : {total_waktu_menit:.2f} menit")
    print(f"Total saham diproses   : {len(tickers_to_process)}")
    print(f"File 'resep emas' telah diperbarui di '{output_file}'")
    
    # Meninggalkan jejak
    with open('logs/optimizer_last_run.log', 'w') as f:
        f.write(datetime.now().isoformat())
    print("\nStempel waktu optimasi berhasil dicatat.")
    print("======================================================")

    # --- PANGGIL "KURIR" UNTUK SINKRONISASI OTOMATIS ---
    if saham_yang_dioptimasi_kali_ini: # Hanya sync jika ada resep baru yang ditemukan
        pesan_commit = f"Auto-sync: Optimasi {len(saham_yang_dioptimasi_kali_ini)} resep saham"
        sync_to_github(pesan_commit)