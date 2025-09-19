import streamlit as st
import os
import subprocess
import sys
from datetime import datetime
from sqlalchemy import create_engine, inspect
import pandas as pd

st.set_page_config(layout="wide")

# --- FUNGSI-FUNGSI BANTU ---

def get_last_run_time(log_file):
    """Membaca stempel waktu dari file log."""
    log_path = os.path.join('logs', log_file)
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            try:
                timestamp = datetime.fromisoformat(f.read().strip())
                return timestamp.strftime('%d %b %Y, %H:%M:%S')
            except:
                return "Format tidak valid"
    return "Belum pernah dijalankan"

def run_script(command, output_placeholder):
    """Menjalankan script eksternal dan menampilkan outputnya secara real-time."""
    output_text = ""
    try:
        process = subprocess.Popen([sys.executable] + command, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.STDOUT, 
                                   text=True, 
                                   encoding='utf-8', 
                                   bufsize=1,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        
        for line in iter(process.stdout.readline, ''):
            output_text += line
            output_placeholder.code(output_text, language='powershell')
        
        process.stdout.close()
        process.wait()
        return process.returncode
    except FileNotFoundError:
        output_placeholder.code(f"Error: Perintah '{sys.executable}' tidak ditemukan.")
        return -1
    except Exception as e:
        output_placeholder.code(f"Terjadi error saat menjalankan subprocess: {e}")
        return -1

# !! PERBAIKAN DI SINI: Tambahkan garis bawah pada argumen 'engine' !!
@st.cache_data
def get_available_stocks_for_control_panel(_engine):
    """Fungsi khusus untuk Pusat Kontrol agar tidak konflik cache."""
    inspector = inspect(_engine)
    stock_names = [name for name in inspector.get_table_names() if not name.endswith(('_weekly', '_sentiment')) and 'news' not in name and 'broker' not in name]
    return sorted(stock_names)

# --- JUDUL APLIKASI ---
st.title("⚙️ Pusat Kontrol Sistem AI")
st.write("Halaman ini digunakan untuk memonitor dan menjalankan proses backend secara manual.")

# --- PANEL MONITORING ---
st.header("Panel Monitoring Status", divider='rainbow')
st.write("Menampilkan waktu terakhir setiap tugas otomatis berhasil diselesaikan.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Update Data Harga Terakhir", get_last_run_time('get_data_last_run.log'))
with col2:
    st.metric("Update Sentimen Terakhir", get_last_run_time('news_scraper_last_run.log'))
with col3:
    st.metric("Pelatihan Ulang AI Terakhir", get_last_run_time('trainer_last_run.log'))

st.info("Catatan: Waktu di atas hanya diperbarui jika script yang relevan berjalan sampai selesai tanpa error.")
# --- PANEL EKSEKUSI MANUAL ---
st.header("Panel Eksekusi Manual", divider='rainbow')

# --- Sub-Bagian: Perawatan Sistem (DIPERBARUI DENGAN FITUR UPLOAD) ---
st.subheader("Perawatan Sistem", divider='rainbow')
st.write("Gunakan fitur ini untuk memperbarui daftar induk saham BEI di sistem.")

uploaded_file = st.file_uploader(
    "Unggah file 'semua_saham_bei.csv' baru di sini:",
    type=['csv']
)

if uploaded_file is not None:
    try:
        # Baca file yang diunggah
        df_new_tickers = pd.read_csv(uploaded_file)
        
        # Validasi sederhana: pastikan ada kolom 'ticker'
        if 'ticker' in df_new_tickers.columns:
            # Timpa file lama dengan yang baru
            df_new_tickers.to_csv('semua_saham_bei.csv', index=False)
            st.success(f"Berhasil! Daftar induk saham telah diperbarui dengan {len(df_new_tickers)} ticker.")
            st.info("Harap muat ulang (refresh) halaman ini untuk melihat daftar saham terbaru di widget.")
        else:
            st.error("Format file CSV tidak valid. Pastikan ada kolom bernama 'ticker'.")

    except Exception as e:
        st.error(f"Gagal memproses file yang diunggah. Error: {e}")


st.write("---")

# --- Sub-Bagian: Pengumpulan Data Harian ---
st.subheader("Pengumpulan Data Harian", divider='rainbow')

# Inisialisasi engine dan daftar saham sekali saja
db_file_path = "data_saham.db"
engine = create_engine(f"sqlite:///{db_file_path}")
# Panggil fungsi yang sudah diperbaiki
stock_list = get_available_stocks_for_control_panel(engine)

data_tickers = st.multiselect(
    "Pilih saham spesifik untuk diunduh (kosongkan untuk semua dari 'semua_saham_bei.csv'):",
    options=stock_list,
    key="data_selection"
)

col_data1, col_data2 = st.columns(2)
with col_data1:
    if st.button("Unduh Data Harga Pilihan", use_container_width=True):
        command = ['get_data.py']
        if data_tickers:
            command.append("--tickers")
            command.extend(data_tickers)
        
        with st.spinner(f"Menjalankan `get_data.py` untuk {len(data_tickers) or 'semua'} saham..."):
            with st.expander("Lihat Output Terminal", expanded=True):
                terminal_output = st.empty()
                return_code = run_script(command, terminal_output)
                if return_code == 0:
                    st.success("Proses unduh data harga selesai!")
                else:
                    st.error("Terjadi error saat menjalankan script.")

with col_data2:
    if st.button("Unduh Data Sentimen Pilihan", use_container_width=True):
        command = ['news_scraper.py']
        if data_tickers:
            command.append("--tickers")
            command.extend(data_tickers)
            
        with st.spinner(f"Menjalankan `news_scraper.py` untuk {len(data_tickers) or 'watchlist default'} saham..."):
            with st.expander("Lihat Output Terminal", expanded=True):
                terminal_output = st.empty()
                return_code = run_script(command, terminal_output)
                if return_code == 0:
                    st.success("Proses unduh data sentimen selesai!")
                else:
                    st.error("Terjadi error saat menjalankan script.")
                    # --- Sub-Bagian: Pelatihan & Analisis ---
st.write("---")
st.subheader("Pelatihan, Optimasi, & Backtesting", divider='rainbow')

analysis_tickers = st.multiselect(
    "Pilih saham spesifik untuk dilatih/dianalisis (kosongkan untuk semua model yang ada):",
    options=stock_list,
    key="analysis_selection"
)

col_analysis1, col_analysis2 = st.columns(2)
with col_analysis1:
    if st.button("4. Latih Ulang Model AI Pilihan", use_container_width=True):
        command = ['trainer.py']
        if analysis_tickers:
            command.append("--tickers")
            command.extend(analysis_tickers)
            
        with st.spinner(f"Menjalankan `trainer.py` untuk {len(analysis_tickers) or 'semua'} saham..."):
            with st.expander("Lihat Output Terminal", expanded=True):
                terminal_output = st.empty()
                return_code = run_script(command, terminal_output)
                if return_code == 0:
                    st.success("Proses pelatihan ulang model selesai!")
                else:
                    st.error("Terjadi error saat menjalankan script.")

with col_analysis2:
    if st.button("5. Jalankan Backtest Pilihan", use_container_width=True):
        command = ['ai_backtester.py']
        if analysis_tickers:
            command.append("--tickers")
            command.extend(analysis_tickers)

        with st.spinner(f"Menjalankan `ai_backtester.py` untuk {len(analysis_tickers) or 'semua'} saham..."):
            with st.expander("Lihat Output Terminal", expanded=True):
                terminal_output = st.empty()
                return_code = run_script(command, terminal_output)
                if return_code == 0:
                    st.success("Proses backtesting selesai! Laporan CSV telah diperbarui.")
                else:
                    st.error("Terjadi error saat menjalankan script.")

# Expander khusus untuk Optimizer karena ini proses yang sangat berat
with st.expander("⚠️ 6. Jalankan Optimasi Parameter (Proses Sangat Lama)"):
    st.warning("PERINGATAN: Proses ini akan memakan waktu sangat lama (berjam-jam per saham) dan akan membebani CPU Anda secara maksimal.")
    
    optimizer_tickers = st.multiselect(
        "Pilih saham yang akan dioptimasi:",
        options=stock_list,
        key="optimizer_selection"
    )

    if st.button("JALANKAN OPTIMASI SEKARANG", type="primary"):
        if optimizer_tickers:
            command = ['optimizer.py', '--tickers'] + optimizer_tickers
            with st.spinner(f"Memulai proses optimasi untuk {len(optimizer_tickers)} saham. Ini akan berjalan di latar belakang..."):
                with st.expander("Lihat Output Terminal", expanded=True):
                    terminal_output = st.empty()
                    return_code = run_script(command, terminal_output)
                    if return_code == 0:
                        st.success("Proses optimasi selesai! File 'optimal_params.json' telah diperbarui.")
                    else:
                        st.error("Terjadi error saat menjalankan script.")
        else:
            st.error("Silakan pilih setidaknya satu saham untuk dioptimasi.")