# pages/pelabelan_sentimen.py
import streamlit as st
import pandas as pd
import os
import sys
import ast

# Tambahkan direktori induk untuk mengimpor modul dari utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import data_loader, text_preprocessing

def sentiment_analysis_lexicon_indonesia(text_list, lexicon_positive_dict, lexicon_negative_dict):
    """
    Melakukan analisis sentimen menggunakan metode leksikon.
    text_list diharapkan sudah dalam bentuk list of tokens (kata).
    """
    score = 0
    if not text_list: # Handle jika list kosong
        return 0, 'Netral'

    for word in text_list:
        score += lexicon_positive_dict.get(word, 0)
        score += lexicon_negative_dict.get(word, 0)
    
    if score > 0:
        sentiment = 'Positif'
    elif score < 0:
        sentiment = 'Negatif'
    else:
        sentiment = 'Netral'
    return score, sentiment

def show_page():
    st.header("Pelabelan Sentimen")
    
    # Unggah data yang sudah diproses
    uploaded_preprocessed = st.file_uploader("Unggah file CSV yang sudah diproses (harus mengandung kolom 'Text Stemming')", type="csv")
    
    df2 = None
    if uploaded_preprocessed is not None:
        df2 = pd.read_csv(uploaded_preprocessed)
        if 'Text Stemming' not in df2.columns:
            st.error("File harus mengandung kolom 'Text Stemming'. Pastikan Anda mengunggah data yang benar.")
            st.stop()
        
        st.write("### Data yang Diunggah")
        st.write(df2.head())
        
        # Tombol untuk memulai analisis sentimen
        if st.button("Mulai Pelabelan Sentimen", key="start_sentiment"):
            st.session_state.run_sentiment = True
    else:
        # Coba ambil dari session state jika belum diunggah ulang
        if st.session_state.preprocessed_data is not None:
            df2 = st.session_state.preprocessed_data.copy()
            if 'Text Stemming' not in df2.columns:
                 st.error("Session state data tidak mengandung 'Text Stemming'. Harap unggah ulang atau pastikan preprocessing berhasil.")
                 st.stop()
            st.write("### Data dari Tahap Preprocessing Sebelumnya")
            st.write(df2[['Komentar', 'Text Stemming']].head())
            if st.button("Mulai Pelabelan Sentimen", key="start_sentiment_from_session"):
                st.session_state.run_sentiment = True
        else:
            st.info("Harap unggah file CSV yang sudah diproses atau selesaikan preprocessing di tab sebelumnya.")
            st.stop()

    if st.session_state.get('run_sentiment', False) and df2 is not None:
        # Membersihkan data sebelum pelabelan
        st.write("### Pembersihan Data Sebelum Pelabelan")
        
        initial_count = len(df2)
        
        # Konversi string representasi list ke list Python aktual
        df2['Text Stemming'] = df2['Text Stemming'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith('[') else x)

        # Hapus baris dengan nilai NaN atau list kosong
        df2 = df2.dropna(subset=['Text Stemming'])
        df2 = df2[df2['Text Stemming'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        
        final_count = len(df2)
        
        st.write(f"Jumlah data awal: {initial_count}")
        st.write(f"Total data bersih yang akan diproses: {final_count} (Dihapus: {initial_count - final_count} baris)")
        st.success("Data berhasil dibersihkan dan siap dilabeli.")
        st.write("Contoh data setelah dibersihkan:")
        st.write(df2.head(5))

        # Muat leksikon sentimen
        lexicon_positive = data_loader.load_lexicon("positive")
        lexicon_negative = data_loader.load_lexicon("negative")

        if not lexicon_positive or not lexicon_negative:
            st.error("File leksikon sentimen tidak ditemukan atau terjadi kesalahan. Pastikan 'kamus_positive.xlsx' dan 'kamus_negative.xlsx' ada di direktori 'dataset'.")
            st.session_state.run_sentiment = False # Reset flag agar tidak terus berjalan
            st.stop()
        else:
            st.success("Leksikon sentimen berhasil dimuat.")
        
        with st.spinner('Melakukan analisis sentimen...'):
            results = df2['Text Stemming'].apply(lambda x: sentiment_analysis_lexicon_indonesia(x, lexicon_positive, lexicon_negative))
            
            # Pisahkan skor dan sentimen
            scores = [r[0] for r in results]
            sentiments = [r[1] for r in results]
            
            df2['Skor Polaritas'] = scores
            df2['Sentimen'] = sentiments
            
            # Simpan hasil di session state
            st.session_state.sentiment_results = df2
            
            # Tampilkan hasil
            st.write("### Hasil Analisis Sentimen")
            st.write(df2)
            
            # Distribusi sentimen
            st.write("### Distribusi Sentimen")
            sentiment_counts = df2['Sentimen'].value_counts()
            st.bar_chart(sentiment_counts)
            
            # Contoh hasil
            st.write("### Contoh Hasil")
            cols = st.columns(3)
            with cols[0]:
                st.write("**Contoh Positif**")
                pos_examples = df2[df2['Sentimen'] == 'Positif'].head(3)
                st.write(pos_examples[['Text Stemming', 'Skor Polaritas']])
            with cols[1]:
                st.write("**Contoh Negatif**")
                neg_examples = df2[df2['Sentimen'] == 'Negatif'].head(3)
                st.write(neg_examples[['Text Stemming', 'Skor Polaritas']])
            with cols[2]:
                st.write("**Contoh Netral**")
                neu_examples = df2[df2['Sentimen'] == 'Netral'].head(3)
                st.write(neu_examples[['Text Stemming', 'Skor Polaritas']])
            
            # Tombol unduh
            st.download_button(
                label="Unduh Hasil Sentimen",
                data=df2.to_csv(index=False).encode('utf-8'),
                file_name='hasil_sentimen.csv',
                mime='text/csv'
            )
        st.session_state.run_sentiment = False # Reset flag setelah selesai