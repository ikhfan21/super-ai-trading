import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

def jalankan_backtesting(ticker_to_analyze, modal_awal=100_000_000):
    """
    Fungsi utama untuk menjalankan simulasi backtesting untuk satu saham.
    """
    # --- 1. MEMUAT & MEMPERSIAPKAN DATA (Sama seperti di analysis.py) ---
    print(f"\n--- Memulai Backtesting untuk {ticker_to_analyze} ---")
    db_file_path = "sqlite:///data_saham.db"
    engine = create_engine(db_file_path)
    
    try:
        df = pd.read_sql(f"SELECT * FROM '{ticker_to_analyze}'", engine, index_col='Date', parse_dates=['Date'])
    except Exception as e:
        print(f"Gagal memuat data untuk {ticker_to_analyze}. Error: {e}")
        return

    # Hitung Indikator & Sinyal
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['Posisi'] = np.where(df['SMA50'] > df['SMA200'], 1, 0)
    df['Sinyal'] = df['Posisi'].diff()

    # --- 2. INISIALISASI PORTOFOLIO & SIMULASI ---
    kas = modal_awal
    jumlah_saham = 0
    nilai_portfolio_harian = []
    
    for i in range(len(df)):
        harga_hari_ini = df['Close'].iloc[i]
        
        # Logika Beli
        if df['Sinyal'].iloc[i] == 1 and kas > 0:
            jumlah_saham_dibeli = kas / harga_hari_ini
            jumlah_saham += jumlah_saham_dibeli
            kas = 0
            print(f"{df.index[i].date()}: BUY @ {harga_hari_ini:.2f} | Total Saham: {jumlah_saham:.2f}")

        # Logika Jual
        elif df['Sinyal'].iloc[i] == -1 and jumlah_saham > 0:
            kas += jumlah_saham * harga_hari_ini
            print(f"{df.index[i].date()}: SELL @ {harga_hari_ini:.2f} | Kas Sekarang: Rp {kas:,.0f}")
            jumlah_saham = 0
            
        # Hitung nilai portfolio setiap hari
        nilai_portfolio_harian.append(kas + (jumlah_saham * harga_hari_ini))
        
    df['Nilai_Portfolio'] = nilai_portfolio_harian
    
    # --- 3. MENGHITUNG & MENAMPILKAN HASIL AKHIR ---
    nilai_akhir = df['Nilai_Portfolio'].iloc[-1]
    total_return_pct = ((nilai_akhir - modal_awal) / modal_awal) * 100
    
    # Hitung return "Buy and Hold" sebagai pembanding
    buy_and_hold_return_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
    
    print("\n--- HASIL AKHIR BACKTESTING ---")
    print(f"Modal Awal          : Rp {modal_awal:,.0f}")
    print(f"Nilai Akhir Portfolio : Rp {nilai_akhir:,.0f}")
    print(f"Total Return Strategi : {total_return_pct:.2f}%")
    print(f"Total Return Buy & Hold : {buy_and_hold_return_pct:.2f}%")

    # --- 4. VISUALISASI KINERJA (EQUITY CURVE) ---
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Nilai_Portfolio'], label='Kinerja Strategi Golden Cross', color='blue')
    # Plot kinerja Buy and Hold
    plt.plot(df.index, (df['Close'] / df['Close'].iloc[0]) * modal_awal, label='Kinerja Buy & Hold', color='gray', linestyle='--')
    
    plt.title(f'Kinerja Strategi vs. Buy & Hold pada {ticker_to_analyze}', fontsize=16)
    plt.xlabel('Tanggal')
    plt.ylabel('Nilai Portfolio (IDR)')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- BAGIAN EKSEKUSI UTAMA ---
if __name__ == "__main__":
    jalankan_backtesting('BBCA.JK')