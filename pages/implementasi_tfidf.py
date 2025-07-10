# pages/implementasi_tfidf.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import sys
import os

# Tambahkan direktori induk untuk mengimpor modul dari utils (jika diperlukan)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def show_page():
    st.header("Implementasi TF-IDF")
    
    # Cek apakah data sudah dipisahkan
    # Pengecekan awal untuk memastikan X_train dan X_test ada dan tidak None
    if 'X_train' not in st.session_state or st.session_state.X_train is None or \
       'X_test' not in st.session_state or st.session_state.X_test is None:
        st.warning("Harap lakukan pembagian data di tab 'Splitting Data' terlebih dahulu.")
        st.info("Silakan ke tab 'Splitting Data' untuk memisahkan data Anda.")
        st.stop() # Menghentikan eksekusi lebih lanjut jika data belum siap
    else:
        st.success("Data training dan testing tersedia.")
        st.write("Contoh Data Training (Text):")
        
        # Pengecekan tambahan sebelum memanggil .head() untuk memastikan X_train adalah Series/DataFrame
        if isinstance(st.session_state.X_train, (pd.Series, pd.DataFrame)):
            st.write(st.session_state.X_train.head())
        else:
            st.error("Kesalahan: Data X_train tidak dalam format yang diharapkan (Pandas Series/DataFrame).")
            st.info("Mohon kembali ke tab 'Splitting Data' dan pastikan data berhasil diproses.")
            return # Menghentikan fungsi jika terjadi masalah format data
            
        if st.button("Implementasi Data dengan TF-IDF"):
            with st.spinner('Sedang melakukan implementasi TF-IDF...'):
                
                # Pengecekan kembali untuk keamanan di dalam blok button
                if st.session_state.X_train is None or st.session_state.X_test is None:
                    st.error("Data training atau testing tidak ditemukan. Mohon ulangi langkah 'Splitting Data'.")
                    return # Keluar dari logika tombol jika data hilang
                
                # Buat TF-IDF Vectorizer
                tfidf_vectorizer = TfidfVectorizer()
                
                # Fit dan transform data training
                X_train_tfidf = tfidf_vectorizer.fit_transform(st.session_state.X_train)
                
                # Transform data testing
                X_test_tfidf = tfidf_vectorizer.transform(st.session_state.X_test)
                
                # Simpan hasil di session state
                st.session_state.X_train_tfidf = X_train_tfidf
                st.session_state.X_test_tfidf = X_test_tfidf
                st.session_state.tfidf_vectorizer = tfidf_vectorizer
                
                st.success("Implementasi TF-IDF berhasil!")
                
                # Tampilkan informasi
                st.subheader("Hasil Implementasi")
                
                st.write(f"**Dimensi Data Training (setelah TF-IDF):** {X_train_tfidf.shape}")
                st.write(f"**Dimensi Data Testing (setelah TF-IDF):** {X_test_tfidf.shape}")
                st.write(f"**Jumlah Fitur (kata unik):** {len(tfidf_vectorizer.get_feature_names_out())}")
                
                # Tampilkan beberapa fitur penting (50 fitur pertama)
                st.subheader("Contoh Fitur yang Dihasilkan")
                st.write(tfidf_vectorizer.get_feature_names_out()[:50])
                
                # Visualisasi fitur penting
                st.subheader("Visualisasi Fitur Penting (Top 20 TF-IDF Scores)")
                
                # Ambil 20 fitur dengan bobot tertinggi
                # Menggunakan .A untuk mengkonversi sparse matrix ke dense array
                sum_tfidf = X_train_tfidf.sum(axis=0).A1 # Mengkonversi ke 1D array
                feature_names = tfidf_vectorizer.get_feature_names_out()
                
                # Buat list of tuples (term, score)
                tfidf_scores = [(feature_names[idx], score) for idx, score in enumerate(sum_tfidf)]
                
                # Urutkan berdasarkan skor dan ambil 20 teratas
                tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:20]
                
                df_tfidf = pd.DataFrame(tfidf_scores, columns=['Term', 'Skor TF-IDF'])
                st.bar_chart(df_tfidf.set_index('Term'))
                
                # Tombol untuk mengekspor model TF-IDF
                st.download_button(
                    label="Unduh Model TF-IDF",
                    data=pickle.dumps(tfidf_vectorizer),
                    file_name='tfidf_vectorizer.pkl',
                    mime='application/octet-stream'
                )