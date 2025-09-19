# Impor library yang kita butuhkan
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np # Impor numpy

# --- KONFIGURASI & SETUP ---
db_file_path = "sqlite:///data_saham.db"
engine = create_engine(db_file_path)
ticker_to_analyze = 'BBCA.JK'

# --- MEMUAT DATA DARI DATABASE ---
print(f"Memuat data untuk {ticker_to_analyze} dari database...")
try:
    df = pd.read_sql(f"SELECT * FROM '{ticker_to_analyze}'", engine, index_col='Date', parse_dates=['Date'])
    print("Data berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat data. Error: {e}")
    exit()

# --- MENGHITUNG INDIKATOR TEKNIKAL ---
print("Menghitung Simple Moving Averages (SMA)...")
df['SMA50'] = df['Close'].rolling(window=50).mean()
df['SMA200'] = df['Close'].rolling(window=200).mean()

# --------------------------------------------------------------------
# LANGKAH BARU: MENDETEKSI SINYAL TRADING (GOLDEN & DEATH CROSS)
# --------------------------------------------------------------------
print("Mendeteksi sinyal Golden Cross & Death Cross...")

# 1. Buat kolom 'Posisi' -> 1 jika SMA50 di atas SMA200, 0 jika di bawah
df['Posisi'] = np.where(df['SMA50'] > df['SMA200'], 1, 0)

# 2. Buat kolom 'Sinyal' -> Hitung perbedaan 'Posisi' dengan hari sebelumnya (.diff())
#    Jika berubah dari 0 ke 1, hasilnya 1 (Golden Cross / Sinyal Beli)
#    Jika berubah dari 1 ke 0, hasilnya -1 (Death Cross / Sinyal Jual)
df['Sinyal'] = df['Posisi'].diff()

# 3. Saring dan tampilkan hari-hari di mana sinyal terjadi
sinyal_beli = df[df['Sinyal'] == 1]
sinyal_jual = df[df['Sinyal'] == -1]

print("\n--- SINYAL BELI (GOLDEN CROSS) TERDETEKSI PADA TANGGAL ---")
print(sinyal_beli.index.strftime('%Y-%m-%d')) # Tampilkan hanya tanggalnya

print("\n--- SINYAL JUAL (DEATH CROSS) TERDETEKSI PADA TANGGAL ---")
print(sinyal_jual.index.strftime('%Y-%m-%d')) # Tampilkan hanya tanggalnya

# --- VISUALISASI DATA DENGAN PENANDA SINYAL ---
print("\nMembuat grafik dengan penanda sinyal...")

plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], label='Harga Penutupan', color='skyblue', linewidth=1.5, alpha=0.5)
plt.plot(df.index, df['SMA50'], label='SMA 50 Hari', color='orange', linewidth=2)
plt.plot(df.index, df['SMA200'], label='SMA 200 Hari', color='red', linewidth=2)

# Tambahkan penanda sinyal Beli (panah hijau ke atas)
plt.scatter(sinyal_beli.index, df.loc[sinyal_beli.index]['SMA50'], label='Sinyal Beli (Golden Cross)', marker='^', color='green', s=150, zorder=10)
# Tambahkan penanda sinyal Jual (panah merah ke bawah)
plt.scatter(sinyal_jual.index, df.loc[sinyal_jual.index]['SMA50'], label='Sinyal Jual (Death Cross)', marker='v', color='red', s=150, zorder=10)

plt.title(f'Analisis Sinyal Golden/Death Cross pada {ticker_to_analyze}', fontsize=16)
plt.xlabel('Tanggal', fontsize=12)
plt.ylabel('Harga Penutupan (IDR)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()