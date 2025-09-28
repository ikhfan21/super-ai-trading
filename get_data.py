import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
import time
import argparse
import os
from datetime import datetime, timedelta
from github_sync import sync_to_github

# --- KONFIGURASI & SETUP ---
db_file_path = "sqlite:///data_saham.db"
engine = create_engine(db_file_path)
start_date = "2020-01-01"

# --- FUNGSI UTAMA UNTUK UPDATE DATA SATU SAHAM (DIPERBAIKI TOTAL) ---
def update_stock_data(ticker_symbol):
    """
    Mengunduh dan MENIMPA data Harian & Mingguan untuk satu ticker
    untuk memastikan data selalu yang terbaru.
    """
    end_date = datetime.now() + timedelta(days=1)
    
    try:
        # --- PROSES DATA HARIAN (DAILY) ---
        data_daily = yf.download(
            ticker_symbol, start=start_date, end=end_date,
            interval="1d", progress=False, timeout=10, auto_adjust=False
        )
        
        if data_daily.empty:
            # Jika tidak ada data sama sekali, anggap gagal
            return False 
        
        # Selalu timpa data yang ada
        if isinstance(data_daily.columns, pd.MultiIndex):
            data_daily.columns = data_daily.columns.get_level_values(0)
        data_daily.to_sql(ticker_symbol, engine, if_exists='replace', index=True)
        
        # --- PROSES DATA MINGGUAN (WEEKLY) ---
        data_weekly = yf.download(
            ticker_symbol, start=start_date, end=end_date,
            interval="1wk", progress=False, timeout=10, auto_adjust=False
        )

        if not data_weekly.empty:
            table_name_weekly = f"{ticker_symbol}_weekly"
            if isinstance(data_weekly.columns, pd.MultiIndex):
                data_weekly.columns = data_weekly.columns.get_level_values(0)
            data_weekly.to_sql(table_name_weekly, engine, if_exists='replace', index=True)
        
        return True

    except Exception:
        # Jika ada error apapun saat download, anggap gagal
        return False
    # --- BAGIAN EKSEKUSSI UTAMA (VERSI PUSAT KONTROL) ---
if __name__ == "__main__":
    # Buat folder 'logs' jika belum ada
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Siapkan argumen parser
    parser = argparse.ArgumentParser(description="Pengunduh Data Saham Skala Penuh.")
    parser.add_argument(
        "--tickers", 
        nargs='+', 
        help="(Opsional) Daftar ticker spesifik yang akan diunduh (contoh: BBCA.JK TLKM.JK)"
    )
    args = parser.parse_args()

    tickers_to_process = []

    # Tentukan mode kerja: Spesifik atau Semua
    if args.tickers:
        # MODE SPESIFIK: Hanya proses ticker yang diberikan
        tickers_to_process = [ticker.upper() for ticker in args.tickers]
        print(f"--- MENJALANKAN DALAM MODE SPESIFIK UNTUK {len(tickers_to_process)} SAHAM ---")
    else:
        # MODE SEMUA: Proses semua saham dari file CSV
        try:
            df_all_stocks = pd.read_csv('semua_saham_bei.csv')
            tickers_to_process = df_all_stocks['ticker'].tolist()
            print(f"--- MENJALANKAN DALAM MODE SKALA PENUH UNTUK {len(tickers_to_process)} SAHAM ---")
        except FileNotFoundError:
            print("Error: File 'semua_saham_bei.csv' tidak ditemukan. Jalankan script 'update_master_list.py' terlebih dahulu.")
            exit()
    
    # Inisialisasi penghitung untuk laporan
    total_saham = len(tickers_to_process)
    sukses_count = 0
    gagal_count = 0
    gagal_list = []
    start_time = time.time()
    
    # Loop utama
    for i, ticker in enumerate(tickers_to_process):
        # Tampilkan progres di baris yang sama agar tidak spam terminal
        print(f"Memproses {i+1}/{total_saham}: {ticker}", end='\r')
        
        sukses = update_stock_data(ticker)
        
        if sukses:
            sukses_count += 1
        else:
            gagal_count += 1
            gagal_list.append(ticker)
        
        # Jeda singkat untuk menghormati server yfinance
        time.sleep(0.5)

    # LAPORAN AKHIR
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
    
    # MENINGGALKAN JEJAK: Tulis stempel waktu ke file log
    with open('logs/get_data_last_run.log', 'w') as f:
        f.write(datetime.now().isoformat())
    print("\nStempel waktu update berhasil dicatat.")
    print("="*54)
    
    # Panggil "kurir" untuk sinkronisasi otomatis
    if sukses_count > 0:
        sync_to_github(f"Auto-sync: Update {sukses_count} data saham")