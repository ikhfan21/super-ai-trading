# 🏠_Beranda.py
import streamlit as st

st.set_page_config(layout="wide")

st.title("Selamat Datang di Super AI Trading Dashboard Anda!")
st.write("---")
st.header("Gunakan Panel Navigasi di Sebelah Kiri untuk Memulai")
st.sidebar.success("Pilih halaman di atas.")

st.write(
    """
    Ini adalah pusat komando Anda untuk menganalisis pasar saham Indonesia menggunakan
    kecerdasan buatan yang telah kita bangun dan latih bersama.

    **Fitur Utama:**
    - **Top Rekomendasi:** Menampilkan 10 saham terbaik untuk trading jangka pendek dan jangka panjang.
    - **Analisis Detail:** Membedah setiap saham secara mendalam dengan puluhan indikator,
      analisis sentimen, rencana trading, dan narasi dari AI.

    Selamat menganalisis!
    """
)