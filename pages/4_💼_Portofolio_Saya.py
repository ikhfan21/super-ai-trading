import streamlit as st
import pandas as pd
from datetime import datetime
# Impor semua fungsi dari backend portofolio kita yang sudah final
from portfolio_manager import (
    get_all_positions, 
    add_position, 
    update_position, 
    delete_position, 
    get_recommendation_for_position,
    load_data
)

st.set_page_config(layout="wide")

# --- FUNGSI BANTU KHUSUS UNTUK HALAMAN INI ---
@st.cache_data(ttl=900) # Cache harga selama 15 menit
def get_latest_price(ticker):
    """Mengambil harga penutupan terakhir untuk satu saham."""
    df = load_data(ticker, 'daily')
    if not df.empty:
        return df['Close'].iloc[-1]
    return 0

# --- JUDUL APLIKASI ---
st.title("💼 Portofolio Saya & Asisten AI")
st.write("Catat posisi trading Anda dan dapatkan rekomendasi aksi personal dari Super AI.")

# --- BAGIAN 1: FORMULIR UNTUK MENAMBAH POSISI BARU (TANPA TANGGAL) ---
st.header("Tambah Posisi Baru", divider='rainbow')
# Tombol untuk membersihkan cache
if st.sidebar.button("🔄 Refresh Data Cache", use_container_width=True, key='refresh_portfolio'):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Cache berhasil dibersihkan! Data akan dimuat ulang.")
    st.rerun()

with st.form("add_position_form", clear_on_submit=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker_input = st.text_input("Kode Saham (Contoh: BBCA)")
    with col2:
        buy_price_input = st.number_input("Harga Beli Rata-rata per Lembar", min_value=0, step=1)
    with col3:
        lots_input = st.number_input("Jumlah Lot", min_value=1, step=1)
    
    submitted = st.form_submit_button("Simpan Posisi Baru", use_container_width=True, type="primary")

    if submitted:
        if ticker_input and buy_price_input > 0 and lots_input > 0:
            add_position(
                ticker=ticker_input,
                buy_price=buy_price_input,
                lots=lots_input
            )
            st.success(f"Posisi {ticker_input.upper()} berhasil ditambahkan!")
            st.cache_data.clear() # Hapus cache agar data portofolio di-load ulang
            st.rerun()
        else:
            st.warning("Harap isi semua kolom dengan benar.")
            # --- BAGIAN 2: MENAMPILKAN SEMUA POSISI AKTIF ---
st.write("---")
st.header("Posisi Aktif Saya", divider='rainbow')

all_my_positions = get_all_positions()

if all_my_positions.empty:
    st.info("Anda belum memiliki posisi aktif. Silakan tambahkan posisi baru menggunakan formulir di atas.")
else:
    for index, position in all_my_positions.iterrows():
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1.5])
            
            with col1:
                st.subheader(f"{position['ticker']}")
                st.write(f"Harga Beli Rata-rata:")
                st.subheader(f"Rp {position['buy_price']:,.0f}")

            latest_price = get_latest_price(position['ticker'])
            buy_value = position['buy_price'] * position['lots'] * 100
            current_value = latest_price * position['lots'] * 100 if latest_price > 0 else 0
            profit_loss = current_value - buy_value
            profit_loss_pct = (profit_loss / buy_value) * 100 if buy_value > 0 else 0

            with col2:
                st.metric("Nilai Beli", f"Rp {buy_value:,.0f}")
                st.metric("Jumlah Lot", f"{position['lots']} Lot")

            with col3:
                st.metric("Harga Saat Ini", f"Rp {latest_price:,.0f}")
                st.metric(
                    label="Potensi Profit/Loss", 
                    value=f"Rp {profit_loss:,.0f}", 
                    delta=f"{profit_loss_pct:.2f}%"
                )
            
            with col4:
                if st.button("Dapatkan Analisis AI", key=f"analyze_{position['id']}", use_container_width=True, type="primary"):
                    st.session_state[f"show_analysis_{position['id']}"] = not st.session_state.get(f"show_analysis_{position['id']}", False)
                
                action_cols = st.columns(2)
                with action_cols[0]:
                    if st.button("✏️", key=f"edit_{position['id']}", use_container_width=True, help="Edit Posisi"):
                        st.session_state[f"edit_mode_{position['id']}"] = not st.session_state.get(f"edit_mode_{position['id']}", False)
                with action_cols[1]:
                    if st.button("🗑️", key=f"delete_{position['id']}", use_container_width=True, help="Hapus Posisi"):
                        delete_position(position['id'])
                        st.cache_data.clear()
                        st.rerun()

            # --- BAGIAN 3: MENAMPILKAN HASIL ANALISIS & FORM EDIT ---
            if st.session_state.get(f"show_analysis_{position['id']}", False):
                with st.spinner(f"AI sedang menganalisis posisi {position['ticker']}..."):
                    rekomendasi = get_recommendation_for_position(position)
                
                if rekomendasi.get("error"):
                    st.error(f"Gagal mendapatkan rekomendasi: {rekomendasi['error']}")
                else:
                    with st.container(border=True):
                        st.subheader(f"⚡ Rekomendasi Aksi untuk {position['ticker']}")
                        st.metric("Rekomendasi AI", rekomendasi['rekomendasi_aksi'])
                        st.info(f"**Narasi AI:** {rekomendasi['narasi']}")
                        
                        rec_col1, rec_col2, rec_col3 = st.columns(3)
                        rec_col1.metric("Harga Terakhir", f"Rp {rekomendasi['harga_terakhir']:,.0f}")
                        rec_col2.metric("Prediksi Stop Loss AI", f"Rp {rekomendasi['stop_loss_price']:,.0f}")
                        rec_col3.metric("Prediksi Take Profit AI", f"Rp {rekomendasi['take_profit_price']:,.0f}")
                        
                        if rekomendasi['rekomendasi_tambahan'] != "-":
                            st.success(f"**Saran Tambahan:** {rekomendasi['rekomendasi_tambahan']}")

            if st.session_state.get(f"edit_mode_{position['id']}", False):
                with st.form(f"edit_form_{position['id']}"):
                    st.subheader(f"Edit Posisi {position['ticker']}")
                    
                    edit_col1, edit_col2, edit_col3 = st.columns(3)
                    with edit_col1:
                        new_ticker = st.text_input("Kode Saham", value=position['ticker'], key=f"edit_ticker_{position['id']}")
                    with edit_col2:
                        new_buy_price = st.number_input("Harga Beli", value=position['buy_price'], key=f"edit_price_{position['id']}")
                    with edit_col3:
                        new_lots = st.number_input("Jumlah Lot", value=position['lots'], key=f"edit_lots_{position['id']}")
                    
                    save_button, cancel_button = st.columns([1, 4])
                    with save_button:
                        if st.form_submit_button("Simpan", use_container_width=True, type="primary"):
                            update_position(
                                position_id=position['id'], ticker=new_ticker, buy_price=new_buy_price, lots=new_lots
                            )
                            st.session_state[f"edit_mode_{position['id']}"] = False
                            st.cache_data.clear()
                            st.rerun()
                    with cancel_button:
                        if st.form_submit_button("Batal", use_container_width=True):
                            st.session_state[f"edit_mode_{position['id']}"] = False
                            st.rerun()