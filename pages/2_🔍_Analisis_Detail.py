import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, inspect
import plotly.graph_objects as go
import joblib
import pandas_ta as ta
import numpy as np
import json

# --- KONFIGURASI & SETUP ---
st.set_page_config(layout="wide") 
db_file_path = "data_saham.db"
engine = create_engine(f"sqlite:///{db_file_path}")

# --- FUNGSI-FUNGSI BANTU ---
@st.cache_data
def load_data(ticker, timeframe='daily'):
    table_name = f"{ticker}_weekly" if timeframe == 'weekly' else ticker
    try:
        df = pd.read_sql(f"SELECT * FROM '{table_name}'", engine, index_col='Date', parse_dates=['Date'])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_sentiment_data(ticker):
    try:
        df = pd.read_sql(f"SELECT * FROM news_sentiment WHERE ticker='{ticker}'", engine, parse_dates=['date'])
        df = df.sort_values(by='date', ascending=False).drop_duplicates(subset=['headline'])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def get_available_stocks(_engine):
    inspector = inspect(_engine)
    stock_names = [name for name in inspector.get_table_names() if not name.endswith(('_weekly', '_sentiment')) and 'news' not in name and 'broker' not in name]
    return sorted(stock_names)

@st.cache_resource
def load_ai_model(ticker):
    model_filename = f'models/{ticker}_model.joblib'
    try:
        model = joblib.load(model_filename)
        return model
    except FileNotFoundError:
        return None

@st.cache_data
def load_optimal_params():
    try:
        with open('optimal_params.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def interpretasi_adx(row):
    adx = row.get('ADX_14', 0)
    dmp = row.get('DMP_14', 0)
    dmn = row.get('DMN_14', 0)
    if adx > 25 and dmp > dmn: return "Uptrend Kuat"
    elif adx > 25 and dmn > dmp: return "Downtrend Kuat"
    elif adx < 20: return "Sideways / Tren Lemah"
    else: return "Tren Netral"

def cari_pola_candlestick(row):
    pola_bullish, pola_bearish = [], []
    for col in row.index:
        if col.startswith('CDL_') and row[col] != 0:
            (pola_bullish if row[col] > 0 else pola_bearish).append(col.replace('CDL_', '').replace('_', ' '))
    return pola_bullish, pola_bearish

def generate_narrative(data, ticker):
    narasi = []
    sinyal = data['Prediksi_Sinyal']
    harga = data['Close']
    sma_weekly = data.get('SMA_20_weekly', 0)
    status_tren = interpretasi_adx(data)
    pola_bullish, pola_bearish = cari_pola_candlestick(data)
    sentimen = data.get('sentiment_sum', 0)
    resistance_1 = data.get('r1', 0)

    if sinyal == 1:
        narasi.append(f"Berdasarkan analisis komprehensif, model AI memberikan rekomendasi **BELI / TAHAN POSISI** untuk **{ticker}** hari ini.")
        if harga > sma_weekly: narasi.append("Arah jangka panjang juga positif, karena harga saat ini berada di atas harga rata-rata 20 mingguan.")
        if status_tren == "Uptrend Kuat": narasi.append("Rekomendasi ini diperkuat oleh kekuatan tren harian yang sedang dalam fase Naik Kuat.")
        if pola_bullish: narasi.append(f"Momentum jangka pendek juga menguat, ditandai dengan munculnya pola lilin (candlestick) bullish: **{', '.join(pola_bullish)}**.")
        if sentimen > 0: narasi.append("Sentimen pasar yang tercermin dari berita terbaru juga cenderung positif.")
        if harga < sma_weekly: narasi.append("**PERHATIAN:** Sinyal beli ini bersifat spekulatif karena berlawanan dengan tren utama mingguan yang masih menunjukkan pelemahan.")
        if pola_bearish: narasi.append(f"Perlu diwaspadai munculnya pola candlestick bearish: **{', '.join(pola_bearish)}**, yang bisa menjadi sinyal pelemahan jangka pendek.")
        if harga > resistance_1 > 0: narasi.append(f"**PERHATIAN:** Harga saat ini sudah menembus level Resistance 1 (Rp {resistance_1:,.0f}), yang bisa berarti harga sudah relatif tinggi dalam jangka pendek.")
    else:
        narasi.append(f"Berdasarkan analisis komprehensif, model AI merekomendasikan **JUAL / TUNGGU** untuk **{ticker}** hari ini.")
        if status_tren == "Downtrend Kuat": narasi.append("Rekomendasi ini sejalan dengan tren harian yang sedang dalam fase Turun Kuat.")
        if harga < sma_weekly: narasi.append("Arah jangka panjang juga negatif, dimana harga berada di bawah harga rata-rata 20 mingguan.")
        if pola_bearish: narasi.append(f"Pelemahan jangka pendek juga terlihat dari munculnya pola candlestick bearish: **{', '.join(pola_bearish)}**.")
        if sentimen < 0: narasi.append("Sentimen dari berita terbaru juga cenderung negatif, yang dapat menambah tekanan jual.")

    return " ".join(narasi)

# --- JUDUL APLIKASI ---
st.title("🔍 Analisis Detail Saham")

# --- SIDEBAR ---
st.sidebar.header("Panel Kontrol")
available_stocks = get_available_stocks(engine)
# Tombol untuk membersihkan cache
if st.sidebar.button("🔄 Refresh Data Cache", use_container_width=True, key='refresh_detail'):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Cache berhasil dibersihkan! Data akan dimuat ulang.")
    st.rerun()

if not available_stocks:
    st.sidebar.error("Database saham kosong.")
else:
    # --- LOGIKA NAVIGASI DARI HALAMAN REKOMENDASI ---
    default_ticker_index = 0
    if 'selected_ticker_from_recs' in st.session_state:
        try:
            default_ticker_index = available_stocks.index(st.session_state['selected_ticker_from_recs'])
            del st.session_state['selected_ticker_from_recs']
        except ValueError:
            default_ticker_index = 0

    selected_ticker = st.sidebar.selectbox(
        "Pilih Saham untuk Dianalisis:",
        available_stocks,
        index=default_ticker_index
    )
    atr_multiplier = st.sidebar.slider("Multiplier ATR untuk Cut Loss", 1.0, 4.0, 2.0, 0.1, key='detail_atr')
    risk_reward_ratio = st.sidebar.slider("Rasio Risk/Reward untuk Take Profit", 1.0, 4.0, 1.5, 0.1, key='detail_rrr')
    
    all_optimal_params = load_optimal_params()

    # --- UTAMA APLIKASI ---
    df_daily = load_data(selected_ticker, timeframe='daily')
    df_weekly = load_data(selected_ticker, timeframe='weekly')
    df_sentiment_raw = load_sentiment_data(selected_ticker)
    
    if len(df_daily) > 1 and len(df_weekly) > 1:
        model = load_ai_model(selected_ticker)
        if model is not None:
            
            params = all_optimal_params.get(selected_ticker, {'rsi_length': 14, 'bbands_length': 20, 'n_estimators': 100, 'max_depth': 20, 'min_samples_leaf': 1})
            with st.sidebar.expander("Parameter Optimal Digunakan", expanded=False):
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    st.metric("RSI Length", params.get('rsi_length', 'N/A'))
                    st.metric("Estimators", params.get('n_estimators', 'N/A'))
                    st.metric("Min Samples Leaf", params.get('min_samples_leaf', 'N/A'))
                with p_col2:
                    st.metric("BBands Length", params.get('bbands_length', 'N/A'))
                    st.metric("Max Depth", str(params.get('max_depth', 'N/A')))

            if not df_sentiment_raw.empty:
                sentiment_daily = df_sentiment_raw.groupby(df_sentiment_raw['date'].dt.date).agg(sentiment_sum=('sentiment', 'sum')).reset_index()
                sentiment_daily['date'] = pd.to_datetime(sentiment_daily['date'])
                sentiment_daily.set_index('date', inplace=True)
            else:
                sentiment_daily = pd.DataFrame()

            df_weekly['SMA_20_weekly'] = df_weekly.ta.sma(length=20)
            df_weekly['RSI_14_weekly'] = df_weekly.ta.rsi(length=14)
            df_weekly_features = df_weekly[['SMA_20_weekly', 'RSI_14_weekly']]
            df_merged = pd.merge_asof(df_daily, df_weekly_features, left_index=True, right_index=True)
            df = df_merged.merge(sentiment_daily, left_index=True, right_index=True, how='left')

            df.ta.rsi(length=params.get('rsi_length', 14), append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            df.ta.bbands(length=params.get('bbands_length', 20), append=True)
            df.ta.atr(length=14, append=True)
            df.ta.obv(append=True)
            df.ta.adx(length=14, append=True)
            df.ta.cdl_pattern(name="all", append=True)
            df.ta.pivots(append=True)
            pivot_levels_raw = ['PIVOTS_TRAD_D_P','PIVOTS_TRAD_D_S1','PIVOTS_TRAD_D_R1']
            pivot_levels_simple = ['p','s1','r1']
            rename_dict = {old: new for old, new in zip(pivot_levels_raw, pivot_levels_simple)}
            df.rename(columns=rename_dict, inplace=True)
            df.fillna(0, inplace=True)
            
            required_features = model.feature_names_in_
            for col in required_features:
                if col not in df.columns: df[col] = 0
            X = df[required_features]
            df['Prediksi_Sinyal'] = model.predict(X)
            
            data_hari_terakhir = df.iloc[-1]
            data_hari_kemarin = df.iloc[-2]
            sinyal_terakhir = data_hari_terakhir['Prediksi_Sinyal']
            harga_terakhir = data_hari_terakhir['Close']
            
            st.header(f"Status Terkini: {selected_ticker}", divider='rainbow')
            perubahan_harian = harga_terakhir - data_hari_kemarin['Close']
            atr_rupiah_terakhir = data_hari_terakhir['ATRr_14']

            col1, col2, col3 = st.columns(3)
            col1.metric(label="Harga Terakhir", value=f"Rp {harga_terakhir:,.0f}", delta=f"Rp {perubahan_harian:,.0f} (1D)")
            col2.metric(label="Sinyal AI (Harian)", value="BELI / TAHAN" if sinyal_terakhir == 1 else "JUAL / TUNGGU")
            col3.metric(label="Volatilitas Harian (ATR)", value=f"Rp {atr_rupiah_terakhir:,.0f}")

            st.header("Analisis Konteks Pasar", divider='rainbow')
            
            col_weekly, col_adx = st.columns(2)
            with col_weekly:
                st.subheader("Tren Jangka Panjang (Mingguan)")
                weekly_trend_status = "Diatas SMA 20 Mingguan" if harga_terakhir > data_hari_terakhir['SMA_20_weekly'] else "Dibawah SMA 20 Mingguan"
                st.metric(label="Status Tren Mingguan", value=weekly_trend_status)
                st.metric(label="RSI Mingguan", value=f"{data_hari_terakhir['RSI_14_weekly']:.2f}")
            with col_adx:
                st.subheader("Kekuatan Tren (Harian)")
                status_tren = interpretasi_adx(data_hari_terakhir)
                st.metric(label="Status Tren Harian", value=status_tren)
            
            st.write("---") 

            col_cdl, col_senti = st.columns(2)
            with col_cdl:
                st.subheader("Pola Candlestick (Harian)")
                pola_bullish, pola_bearish = cari_pola_candlestick(data_hari_terakhir)
                if pola_bullish: st.success(f"Bullish: {', '.join(pola_bullish)}")
                if pola_bearish: st.warning(f"Bearish: {', '.join(pola_bearish)}")
                if not pola_bullish and not pola_bearish: st.info("Tidak ada pola signifikan.")
            
            with col_senti:
                st.subheader("Sentimen Berita (Hari Ini)")
                sentiment_sum_today = data_hari_terakhir.get('sentiment_sum', 0)
                news_count_today = int(data_hari_terakhir.get('news_count', 0))
                
                sentiment_text = "Netral"
                if sentiment_sum_today > 0: sentiment_text = "Positif"
                elif sentiment_sum_today < 0: sentiment_text = "Negatif"
                st.metric(label="Mood Pasar Saat Ini", value=sentiment_text)
                st.metric(label="Jumlah Berita Teranalisis", value=news_count_today)

            st.header("Narasi & Interpretasi AI", divider='rainbow')
            narasi_final = generate_narrative(data_hari_terakhir, selected_ticker)
            st.info(narasi_final)

            if not df_sentiment_raw.empty:
                st.subheader("Dasar Analisis Sentimen (Berita Terkini)")
                berita_terkini = df_sentiment_raw[df_sentiment_raw['date'].dt.date == data_hari_terakhir.name.date()]
                if not berita_terkini.empty:
                    for index, row in berita_terkini.head(5).iterrows():
                        if row['sentiment'] == 1: st.markdown(f"🟢 {row['headline']}")
                        elif row['sentiment'] == -1: st.markdown(f"🔴 {row['headline']}")
                        else: st.markdown(f"⚪ {row['headline']}")
                else:
                    st.write("Tidak ada berita yang ditemukan untuk hari ini.")
            
            st.header("Level Kunci Pivot Points Hari Ini", divider='rainbow')
            col_pp, col_s1, col_r1 = st.columns(3)
            col_pp.metric("Pivot Point (PP)", f"Rp {data_hari_terakhir.get('p', 0):,.0f}")
            col_s1.metric("Support 1 (S1)", f"Rp {data_hari_terakhir.get('s1', 0):,.0f}")
            col_r1.metric("Resistance 1 (R1)", f"Rp {data_hari_terakhir.get('r1', 0):,.0f}")

            st.header("Rencana Aksi Trading (Harian)", divider='rainbow')
            
            rencana_trading_valid = False
            if sinyal_terakhir == 1:
                st.subheader("Rencana Aksi Saat Ini (Posisi Beli / Tahan)")
                harga_beli_saran = harga_terakhir
                risiko_per_saham = atr_multiplier * atr_rupiah_terakhir
                harga_cut_loss = harga_beli_saran - risiko_per_saham
                harga_take_profit = harga_beli_saran + (risiko_per_saham * risk_reward_ratio)
                rencana_trading_valid = True

                plan_col1, plan_col2, plan_col3 = st.columns(3)
                plan_col1.metric("Harga Beli Disarankan", f"Rp {harga_beli_saran:,.0f}")
                plan_col2.metric("Level Cut Loss", f"Rp {harga_cut_loss:,.0f}")
                plan_col3.metric("Level Take Profit", f"Rp {harga_take_profit:,.0f}")
            else:
                st.subheader("Observasi Setup Beli Berikutnya")
                harga_pemicu_beli = data_hari_terakhir.get(f'BBM_{params.get("bbands_length", 20)}_2.0_2.0', 0)
                risiko_per_saham = atr_multiplier * atr_rupiah_terakhir
                harga_cut_loss = harga_pemicu_beli - risiko_per_saham
                harga_take_profit = harga_pemicu_beli + (risiko_per_saham * risk_reward_ratio)
                rencana_trading_valid = True

                plan_col1, plan_col2, plan_col3 = st.columns(3)
                plan_col1.metric("Potensi Harga Pemicu Beli (BBM)", f"Rp {harga_pemicu_beli:,.0f}")
                plan_col2.metric("Estimasi Cut Loss", f"Rp {harga_cut_loss:,.0f}")
                plan_col3.metric("Estimasi Take Profit", f"Rp {harga_take_profit:,.0f}")
                st.info("Ini adalah skenario jika harga berhasil menembus ke atas level pemicu.")

            st.header("Grafik Interaktif (Harian)", divider='rainbow')
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Harga Penutupan', line=dict(color='skyblue')))
            
            bbu_col = f'BBU_{params.get("bbands_length", 20)}_2.0_2.0'
            bbl_col = f'BBL_{params.get("bbands_length", 20)}_2.0_2.0'
            if bbu_col in df.columns and bbl_col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[bbu_col], mode='lines', name='Bollinger Atas', line=dict(width=0.5, color='gray')))
                fig.add_trace(go.Scatter(x=df.index, y=df[bbl_col], mode='lines', name='Bollinger Bawah', line=dict(width=0.5, color='gray'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))
            
            buy_periods = df[df['Prediksi_Sinyal'] == 1]
            fig.add_trace(go.Scatter(x=buy_periods.index, y=buy_periods['Close'], mode='markers', name='AI: Beli/Tahan', marker=dict(color='lime', size=5, symbol='circle-open')))
            
            if rencana_trading_valid:
                entry_price = harga_beli_saran if sinyal_terakhir == 1 else harga_pemicu_beli
                if entry_price > 0:
                    fig.add_hline(y=entry_price, line_dash="dot", line_color="blue", annotation_text="LEVEL BELI")
                    fig.add_hline(y=harga_cut_loss, line_dash="dot", line_color="red", annotation_text="CUT LOSS")
                    fig.add_hline(y=harga_take_profit, line_dash="dot", line_color="green", annotation_text="TAKE PROFIT")

            if 'p' in data_hari_terakhir and data_hari_terakhir['p'] > 0:
                fig.add_hline(y=data_hari_terakhir['p'], line_dash="dash", line_color="cyan", annotation_text="Pivot Point (PP)")
                fig.add_hline(y=data_hari_terakhir['s1'], line_dash="dash", line_color="lightgreen", annotation_text="Support 1 (S1)")
                fig.add_hline(y=data_hari_terakhir['r1'], line_dash="dash", line_color="lightcoral", annotation_text="Resistance 1 (R1)")
            
            fig.update_layout(title=f'Analisis Interaktif untuk {selected_ticker}', xaxis_title='Tanggal', yaxis_title='Harga (IDR)', legend_title='Legenda', height=600)
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.error(f"Data untuk {selected_ticker} tidak dapat dimuat.")