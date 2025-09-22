import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, inspect
import time
import argparse
import os
from datetime import datetime, timedelta
from github_sync import sync_to_github

# --- KONFIGURASI & SETUP ---
db_file_path = "sqlite:///data_saham.db"
engine = create_engine(db_file_path)
start_date = "2020-01-01"

# --- FUNGSI UTAMA UNTUK UPDATE DATA SATU SAHAM ---
def update_stock_data(ticker_symbol):
    """
    Mengunduh dan menyimpan data Harian & Mingguan untuk satu ticker.
    """
    inspector = inspect(engine)
    end_date = datetime.now() + timedelta(days=1)
    
    try:
        # --- PROSES DATA HARIAN (DAILY) ---
        if not inspector.has_table(ticker_symbol):
            data_daily = yf.download(
                ticker_symbol, 
                start=start_date, 
                end=end_date,
                interval="1d", 
                progress=False, 
                timeout=10,
                auto_adjust=False # <-- Tambahan eksplisit
            )
            
            if not data_daily.empty:
                if isinstance(data_daily.columns, pd.MultiIndex):
                    data_daily.columns = data_daily.columns.get_level_values(0)
                data_daily.to_sql(ticker_symbol, engine, if_exists='replace')
        
        # --- PROSES DATA MINGGUAN (WEEKLY) ---
        table_name_weekly = f"{ticker_symbol}_weekly"
        if not inspector.has_table(table_name_weekly):
            data_weekly = yf.download(
                ticker_symbol, 
                start=start_date, 
                end=end_date,
                interval="1wk", 
                progress=False, 
                timeout=10,
                auto_adjust=False # <-- Tambahan eksplisit
            )

            if not data_weekly.empty:
                if isinstance(data_weekly.columns, pd.MultiIndex):
                    data_weekly.columns = data_weekly.columns.get_level_values(0)
                data_weekly.to_sql(table_name_weekly, engine, if_exists='replace')
        
        return True

    except Exception:
        return False

# --- BAGIAN EKSEKUSI UTAMA (DENGAN AUTO-SYNC) ---
if __name__ == "__main__":
    if not os.path.exists('logs'):
        os.makedirs('logs')

    parser = argparse.ArgumentParser(description="Pengunduh Data Saham Skala Penuh.")
    parser.add_argument("--tickers", nargs='+', help="(Opsional) Daftar ticker spesifik yang akan diunduh.")
    args = parser.parse_args()

    tickers_to_process = []

    if args.tickers:
        tickers_to_process = [ticker.upper() for ticker in args.tickers]
        print(f"--- MENJALANKAN DALAM MODE SPESIFIK UNTUK {len(tickers_to_process)} SAHAM ---")
    else:
        try:
            df_all_stocks = pd.read_csv('semua_saham_bei.csv')
            tickers_to_process = df_all_stocks['ticker'].tolist()
            print(f"--- MENJALANKAN DALAM MODE SKALA PENUH UNTUK {len(tickers_to_process)} SAHAM ---")
        except FileNotFoundError:
            print("Error: File 'semua_saham_bei.csv' tidak ditemukan.")
            exit()
    
    total_saham = len(tickers_to_process)
    sukses_count = 0
    gagal_count = 0
    gagal_list = []
    start_time = time.time()
    
    for i, ticker in enumerate(tickers_to_process):
        print(f"Memproses {i+1}/{total_saham}: {ticker}", end='\r')
        sukses = update_stock_data(ticker)
        if sukses:
            sukses_count += 1
        else:
            gagal_count += 1
            gagal_list.append(ticker)
        time.sleep(0.5)

    end_time = time.time()
    total_waktu_menit = (end_time - start_time) / 60
    
    print("\n\n" + "="*54)
    print("--- PROSES PENGUNDUHAN DATA SELESAI ---")
    print(f"Laporan Akhir:")
    print(f"Waktu                  : {total_waktu_menit:.2f} menit")
    print(f"Total saham diproses   : {total_saham}")
    print(f"Berhasil diunduh       : {sukses_count}")
    print(f"Gagal diunduh          : {gagal_count}")
    
    if gagal_count > 0:
        print("\nDaftar saham yang gagal:")
        print(", ".join(gagal_list))
    
    with open('logs/get_data_last_run.log', 'w') as f:
        f.write(datetime.now().isoformat())
    print("\nStempel waktu update berhasil dicatat.")
    print("="*54)

    if sukses_count > 0:
        sync_to_github(f"Auto-sync: Update {sukses_count} data saham")