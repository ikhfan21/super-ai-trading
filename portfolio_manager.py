import pandas as pd
from sqlalchemy import create_engine, inspect, text as sqlalchemy_text
import numpy as np
import pandas_ta as ta
import joblib
import json
from datetime import datetime

# --- KONFIGURASI ---
DB_FILE_PATH = "data_saham.db"
TABLE_NAME = "my_portfolio"

# --- FUNGSI-FUNGSI DATABASE (FINAL TANPA TANGGAL & CERDAS) ---
def get_engine():
    """Membuat dan mengembalikan koneksi engine ke database."""
    return create_engine(f"sqlite:///{DB_FILE_PATH}")

def create_portfolio_table():
    """Membuat tabel portofolio jika belum ada (tanpa kolom tanggal)."""
    engine = get_engine()
    inspector = inspect(engine)
    if not inspector.has_table(TABLE_NAME):
        print(f"Membuat tabel '{TABLE_NAME}'...")
        with engine.connect() as conn:
            query = sqlalchemy_text(f"""
            CREATE TABLE {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker VARCHAR(10) NOT NULL,
                buy_price REAL NOT NULL,
                lots INTEGER NOT NULL
            );
            """)
            conn.execute(query)
            conn.commit()
        print("Tabel berhasil dibuat.")

def get_all_positions():
    """Mengambil semua posisi dari tabel portofolio."""
    engine = get_engine()
    try:
        df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)
        return df
    except Exception:
        create_portfolio_table()
        return pd.DataFrame(columns=['id', 'ticker', 'buy_price', 'lots'])

def add_position(ticker, buy_price, lots):
    """Menambahkan posisi baru ke portofolio (dengan validasi .JK)."""
    engine = get_engine()
    ticker = ticker.upper().strip()
    if not ticker.endswith('.JK'):
        ticker += '.JK'
    df = pd.DataFrame([{'ticker': ticker, 'buy_price': buy_price, 'lots': lots}])
    df.to_sql(TABLE_NAME, engine, if_exists='append', index=False)
    print(f"Posisi baru untuk {ticker} berhasil ditambahkan.")
    return True

def update_position(position_id, ticker, buy_price, lots):
    """Memperbarui posisi yang sudah ada (dengan validasi .JK)."""
    engine = get_engine()
    ticker = ticker.upper().strip()
    if not ticker.endswith('.JK'):
        ticker += '.JK'
    with engine.connect() as conn:
        query = sqlalchemy_text(f"""
        UPDATE {TABLE_NAME}
        SET ticker = :ticker, buy_price = :buy_price, lots = :lots
        WHERE id = :id
        """)
        conn.execute(query, {"ticker": ticker, "buy_price": buy_price, "lots": lots, "id": position_id})
        conn.commit()
    print(f"Posisi ID {position_id} berhasil diperbarui.")
    return True

def delete_position(position_id):
    """Menghapus posisi dari portofolio berdasarkan ID-nya."""
    engine = get_engine()
    with engine.connect() as conn:
        query = sqlalchemy_text(f"DELETE FROM {TABLE_NAME} WHERE id = :id")
        conn.execute(query, {"id": position_id})
        conn.commit()
    print(f"Posisi ID {position_id} berhasil dihapus.")
    return True

# --- FUNGSI-FUNGSI BANTU ANALISIS ---
def load_data(ticker, timeframe='daily'):
    engine = get_engine()
    table_name = f"{ticker}_weekly" if timeframe == 'weekly' else ticker
    try:
        df = pd.read_sql(f"SELECT * FROM '{table_name}'", engine, index_col='Date', parse_dates=['Date'])
        return df
    except Exception:
        return pd.DataFrame()

# Fungsi baru untuk memuat 3 model AI
def load_ai_models(ticker):
    try:
        model_arah = joblib.load(f'models/{ticker}_arah_model.joblib')
        model_sl = joblib.load(f'models/{ticker}_sl_model.joblib')
        model_tp = joblib.load(f'models/{ticker}_tp_model.joblib')
        return model_arah, model_sl, model_tp
    except FileNotFoundError:
        return None, None, None

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

# --- FUNGSI BARU: MESIN REKOMENDASI POSISI (VERSI REGRESSOR) ---
def get_recommendation_for_position(position_data):
    """
    Menganalisis posisi trading dan memberikan rekomendasi aksi
    menggunakan tiga model AI: Arah, Stop Loss (SL), dan Take Profit (TP).
    """
    ticker = position_data['ticker']
    buy_price = position_data['buy_price']
    
    # 1. Muat semua data dan TIGA model AI
    df_daily = load_data(ticker, 'daily')
    df_weekly = load_data(ticker, 'weekly')
    model_arah, model_sl, model_tp = load_ai_models(ticker)
    all_optimal_params = load_optimal_params()

    # Validasi data dan model
    if df_daily.empty or df_weekly.empty or not all([model_arah, model_sl, model_tp]):
        return {"error": "Data pasar atau salah satu model AI (Arah, SL, TP) tidak ditemukan."}

    # 2. Lakukan rekayasa fitur yang sama persis seperti di trainer.py
    params = all_optimal_params.get(ticker, {'rsi_length': 14, 'bbands_length': 20})
    
    df_weekly['SMA_20_weekly'] = df_weekly.ta.sma(length=20)
    df_weekly['RSI_14_weekly'] = df_weekly.ta.rsi(length=14)
    df_weekly_features = df_weekly[['SMA_20_weekly', 'RSI_14_weekly']]
    df = pd.merge_asof(df_daily, df_weekly_features, left_index=True, right_index=True)
    
    df.ta.rsi(length=params.get('rsi_length', 14), append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=params.get('bbands_length', 20), append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)
    df.ta.adx(length=14, append=True)
    df.ta.cdl_pattern(name="all", append=True)
    df.ta.pivots(append=True)
    pivot_levels_raw = ['PIVOTS_TRAD_D_P','PIVOTS_TRAD_D_S1','PIVOTS_TRAD_D_R1','PIVOTS_TRAD_D_S2','PIVOTS_TRAD_D_R2']
    pivot_levels_simple = ['p','s1','r1','s2','r2']
    rename_dict = {old: new for old, new in zip(pivot_levels_raw, pivot_levels_simple)}
    df.rename(columns=rename_dict, inplace=True)
    for level in pivot_levels_simple:
        if level in df.columns: df[f'Jarak_ke_{level.upper()}'] = (df['Close'] - df[level]) / df['Close']
    for level in ['p', 's1', 'r1']:
        if level in df.columns: df[f'Posisi_vs_{level.upper()}'] = np.where(df['Close'] > df[level], 1, 0)
    try:
        df_sentiment = pd.read_sql(f"SELECT * FROM news_sentiment WHERE ticker='{ticker}'", get_engine(), parse_dates=['date'])
        if not df_sentiment.empty:
            sentiment_daily = df_sentiment.groupby(df_sentiment['date'].dt.date).agg(sentiment_sum=('sentiment', 'sum')).reset_index()
            sentiment_daily['date'] = pd.to_datetime(sentiment_daily['date'])
            sentiment_daily.set_index('date', inplace=True)
            df = df.merge(sentiment_daily, left_index=True, right_index=True, how='left')
    except:
        df['sentiment_sum'] = 0
    df.fillna(0, inplace=True)

    # 3. Buat prediksi dengan ketiga model
    # Siapkan data X (fitur) untuk hari terakhir saja
    X_last_day = df.drop(columns=[col for col in df.columns if col in pivot_levels_simple]).iloc[[-1]]
    
    # Pastikan semua fitur yang dibutuhkan model ada
    for col in model_arah.feature_names_in_:
        if col not in X_last_day.columns:
            X_last_day[col] = 0
    X_last_day = X_last_day[model_arah.feature_names_in_] # Urutkan kolom

    prediksi_arah = model_arah.predict(X_last_day)[0]
    prediksi_sl = model_sl.predict(X_last_day)[0]
    prediksi_tp = model_tp.predict(X_last_day)[0]

    # 4. Bangun rekomendasi berdasarkan data hari terakhir
    last_day = df.iloc[-1]
    harga_terakhir = last_day['Close']

    # Logika Rekomendasi Aksi
    rekomendasi_aksi = "TAHAN POSISI (HOLD)" if prediksi_arah == 1 else "JUAL (SELL / EXIT)"

    # Logika Stop Loss & Take Profit (sepenuhnya dari AI)
    # Sebagai pengaman, pastikan SL di bawah harga saat ini dan TP di atas
    stop_loss_price = min(prediksi_sl, harga_terakhir * 0.98) # Ambil yang lebih rendah, maksimal 2% dari harga
    take_profit_price = max(prediksi_tp, harga_terakhir * 1.02) # Ambil yang lebih tinggi, minimal 2% dari harga

    # Logika Rekomendasi Tambahan (Beli lagi / Averaging Up)
    rekomendasi_tambahan = "-"
    if prediksi_arah == 1 and harga_terakhir > buy_price and interpretasi_adx(last_day) == "Uptrend Kuat":
        rekomendasi_tambahan = "Tren sedang sangat kuat. Pertimbangkan untuk menambah posisi (Averaging Up)."
        
    # Bangun Narasi
    narasi = f"Model AI Arah memberikan sinyal **{rekomendasi_aksi}**. "
    if prediksi_arah == 1:
        narasi += f"Model AI Peramal Harga memprediksi rentang pergerakan harga dalam 5 hari ke depan adalah antara **Rp {prediksi_sl:,.0f} (terendah)** hingga **Rp {prediksi_tp:,.0f} (tertinggi)**. "
        narasi += "Disarankan untuk menahan posisi selama harga tidak jatuh di bawah level Stop Loss yang diprediksi."
    else:
        narasi += "Model mendeteksi potensi pelemahan atau akhir dari tren naik. Disarankan untuk mengamankan keuntungan atau membatasi kerugian. "
    
    return {
        "rekomendasi_aksi": rekomendasi_aksi,
        "stop_loss_price": stop_loss_price,
        "take_profit_price": take_profit_price,
        "rekomendasi_tambahan": rekomendasi_tambahan,
        "harga_terakhir": harga_terakhir,
        "narasi": narasi,
        "error": None
    }

# --- BAGIAN EKSEKUSI UTAMA ---
if __name__ == "__main__":
    print("Mengecek dan mempersiapkan database portofolio...")
    create_portfolio_table()
    print("\nDatabase siap. Berikut adalah isi portofolio saat ini:")
    print(get_all_positions())

