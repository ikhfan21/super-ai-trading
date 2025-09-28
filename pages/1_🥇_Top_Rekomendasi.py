import streamlit as st
from sqlalchemy import create_engine
from screener import run_screener, get_available_stocks
import pandas as pd

st.set_page_config(layout="wide")

def display_recommendations_page():
    st.title("🥇 Top Rekomendasi Saham")
    st.write("Pusat komando untuk menemukan saham-saham unggulan berdasarkan analisis AI dan Systematic.")

    db_file_path = "data_saham.db"
    stock_list = get_available_stocks(db_file_path)

    if not stock_list:
        st.warning("Database saham kosong. Harap jalankan `get_data.py` terlebih dahulu.")
        return
    
    # Tombol untuk membersihkan cache
    if st.sidebar.button("🔄 Refresh Data Cache", use_container_width=True, key='refresh_recs'):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache berhasil dibersihkan! Data akan dimuat ulang.")
        st.rerun()

    st.sidebar.header("Parameter Screener")
    atr_multiplier = st.sidebar.slider("Multiplier ATR (Jangka Pendek)", 1.0, 4.0, 2.0, 0.1, key='recs_atr')
    risk_reward_ratio = st.sidebar.slider("Rasio R/R (Jangka Pendek)", 1.0, 4.0, 1.5, 0.1, key='recs_rrr')
    rrr_long_term = st.sidebar.slider("Rasio R/R (Jangka Panjang)", 1.0, 5.0, 2.0, 0.5, key='recs_rrr_long')

    if 'run_analysis' not in st.session_state:
        st.session_state['run_analysis'] = False

    if st.button("Jalankan Analisis & Cari Saham Unggulan", use_container_width=True):
        st.session_state['run_analysis'] = True

    if st.session_state['run_analysis']:
        progress_bar = st.progress(0, text="Analisis Dimulai...")
        def update_progress(ticker, percentage):
            progress_bar.progress(percentage, text=f"Menganalisis: {ticker}")
        
        with st.spinner("Mesin AI sedang menganalisis seluruh saham... Harap tunggu."):
            df_short, df_long = run_screener(stock_list, db_file_path, atr_multiplier, risk_reward_ratio, rrr_long_term, update_progress)
        
        st.session_state['df_short'] = df_short
        st.session_state['df_long'] = df_long
        st.success("Analisis selesai!")
        progress_bar.empty()
        st.session_state['run_analysis'] = False

    st.write("---")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚀 Rekomendasi Jangka Pendek (AI-Driven)")
        if 'df_short' in st.session_state and not st.session_state['df_short'].empty:
            df_short_display = st.session_state['df_short']
            for index, row in df_short_display.iterrows():
                with st.container(border=True):
                    c1, c2 = st.columns([1,3])
                    with c1:
                        if st.button(f"🔍 Analisis {row['Saham']}", key=f"short_{row['Saham']}", use_container_width=True):
                            st.session_state['selected_ticker_from_recs'] = row['Saham']
                            st.switch_page("pages/2_🔍_Analisis_Detail.py")
                        st.metric("Skor", f"{row['Skor Rekomendasi']:.2f}")
                    with c2:
                        st.metric("Harga Beli", f"Rp {row['Harga Beli']:,.0f}")
                        plan_cols = st.columns(2)
                        plan_cols[0].metric("Stop Loss", f"Rp {row['Harga Stop Loss']:,.0f}")
                        plan_cols[1].metric("Take Profit", f"Rp {row['Harga Take Profit']:,.0f}")

        else:
            st.info("Tidak ada rekomendasi jangka pendek yang memenuhi kriteria saat ini.")

    with col2:
        st.subheader("⏳ Rekomendasi Jangka Panjang (Systematic)")
        if 'df_long' in st.session_state and not st.session_state['df_long'].empty:
            df_long_display = st.session_state['df_long']
            for index, row in df_long_display.iterrows():
                 with st.container(border=True):
                    c1, c2 = st.columns([1,3])
                    with c1:
                        if st.button(f"🔍 Analisis {row['Saham']}", key=f"long_{row['Saham']}", use_container_width=True):
                            st.session_state['selected_ticker_from_recs'] = row['Saham']
                            st.switch_page("pages/2_🔍_Analisis_Detail.py")
                        st.metric("RSI Mingguan", f"{row['RSI Mingguan']:.2f}")

                    with c2:
                        st.metric("Harga Beli", f"Rp {row['Harga Beli']:,.0f}")
                        plan_cols = st.columns(2)
                        plan_cols[0].metric("Stop Loss", f"Rp {row['Harga Stop Loss']:,.0f}")
                        plan_cols[1].metric("Take Profit", f"Rp {row['Harga Take Profit']:,.0f}")
        else:
            st.info("Tidak ada rekomendasi jangka panjang yang memenuhi kriteria saat ini.")

# Panggil fungsi utama jika file ini dijalankan
if __name__ == "__main__":
    display_recommendations_page()