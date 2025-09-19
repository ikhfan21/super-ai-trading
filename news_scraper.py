import requests
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy import create_engine, text as sqlalchemy_text
import time
from datetime import datetime, timedelta
import urllib.parse
import numpy as np # <-- PERBAIKAN DI SINI

# --- KONFIGURASI ---
db_file_path = "sqlite:///data_saham.db"
engine = create_engine(db_file_path)

# --- KAMUS SENTIMEN MINI ---
KAMUS_POSITIF = [
    'laba', 'naik', 'untung', 'akuisisi', 'sukses', 'optimis', 'bullish',
    'ekspansi', 'pertumbuhan', 'meningkat', 'positif', 'inovasi', 'efisiensi',
    'prospek cerah', 'rekor', 'tertinggi', 'menguat', 'surplus', 'right issue', 'buyback'
]
KAMUS_NEGATIF = [
    'rugi', 'turun', 'anjlok', 'boncos', 'pesimis', 'bearish', 'lesu',
    'koreksi', 'penurunan', 'melemah', 'negatif', 'skandal', 'gagal',
    'krisis', 'utang', 'defisit', 'masalah', 'risiko', 'jatuh', 'gugatan'
]

def analisis_sentimen_sederhana(teks):
    teks = teks.lower()
    skor = 0
    for kata in KAMUS_POSITIF:
        if kata in teks: skor += 1
    for kata in KAMUS_NEGATIF:
        if kata in teks: skor -= 1
    if skor > 0: return 1
    elif skor < 0: return -1
    else: return 0

def scrape_and_analyze_news(ticker_symbol):
    stock_name = ticker_symbol.replace('.JK', '')
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=14)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    search_query = f"saham {stock_name} after:{start_date_str} before:{end_date_str}"
    
    encoded_query = urllib.parse.quote(search_query)
    url = f"https://news.google.com/search?q={encoded_query}&hl=id&gl=ID&ceid=ID:id"
    
    headers = { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36" }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article')
        news_list = []
        
        for article in articles:
            title_tag = article.find('a', class_='JtKRv')
            time_tag = article.find('time', class_='hvbAAd')
            
            if title_tag and time_tag and time_tag.has_attr('datetime'):
                title = title_tag.text
                article_date = pd.to_datetime(time_tag['datetime']).date()
                sentiment_score = analisis_sentimen_sederhana(title)
                
                news_list.append({
                    'date': article_date.isoformat(),
                    'ticker': ticker_symbol, 
                    'headline': title, 
                    'sentiment': sentiment_score
                })

        if not news_list:
            return 0

        df_news = pd.DataFrame(news_list)
        df_news.to_sql('news_sentiment', engine, if_exists='append', index=False)
        return len(df_news)

    except requests.exceptions.RequestException:
        return 0
    except Exception as e:
        print(f"-> [{ticker_symbol}] GAGAL (error tak terduga): {e}")
        return 0

# --- BAGIAN EKSEKUSI UTAMA ---
if __name__ == "__main__":
    
    print("\n--- MULAI PROSES SCRAPING & ANALISIS SENTIMEN ---")
    print("Mode: Skala Penuh | Rentang Waktu: 14 Hari Terakhir")
    
    try:
        with engine.connect() as conn:
            conn.execute(sqlalchemy_text("DROP TABLE IF EXISTS news_sentiment"))
            conn.commit()
        print("Tabel sentimen lama berhasil dihapus.")
    except Exception as e:
        print(f"Tidak bisa menghapus tabel lama (mungkin belum ada): {e}")

    try:
        df_all_stocks = pd.read_csv('semua_saham_bei.csv')
        tickers_to_process = df_all_stocks['ticker'].tolist()
        print(f"Akan memproses sentimen untuk {len(tickers_to_process)} saham. Ini akan sangat lama...")
    except FileNotFoundError:
        print("Error: File 'semua_saham_bei.csv' tidak ditemukan. Proses dibatalkan.")
        exit()
    
    start_time = time.time()
    total_berita = 0
    
    for i, ticker in enumerate(tickers_to_process):
        print(f"Memproses sentimen: ({i+1}/{len(tickers_to_process)}) {ticker}", end='\r')
        
        jumlah_berita_ditemukan = scrape_and_analyze_news(ticker)
        total_berita += jumlah_berita_ditemukan
        
        time.sleep(np.random.uniform(2, 4))
        
    end_time = time.time()
    total_waktu_menit = (end_time - start_time) / 60
    
    print("\n\n" + "="*54)
    print(f"--- SEMUA PROSES SENTIMEN SELESAI ---")
    print(f"Total Waktu              : {total_waktu_menit:.2f} menit")
    print(f"Total Saham Diproses     : {len(tickers_to_process)}")
    print(f"Total Judul Berita Ditemukan: {total_berita}")
    print("="*54)