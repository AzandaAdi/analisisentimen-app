# pages/splitting_data.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

# Tambahkan direktori induk untuk mengimpor modul dari utils (jika diperlukan)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def show_page():
    st.header("Splitting Data Train dan Data Testing untuk Pelatihan Model")
    
    if st.session_state.sentiment_results is not None:
        df_inset = st.session_state.sentiment_results
        
        st.write("### Data yang Akan Displit")
        st.write(df_inset.head())
        
        # Definisikan X dan y
        # Pastikan 'Text Stemming' adalah string atau di-join menjadi string untuk TF-IDF
        df_inset['Text Stemming Str'] = df_inset['Text Stemming'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else str(x))
        X = df_inset["Text Stemming Str"]
        y = df_inset["Sentimen"]
        
        # Slider untuk menentukan ukuran data test
        test_size = st.slider(
            "Persentase Data Testing",
            min_value=0.1,
            max_value=0.5,
            value=0.2, # Ubah nilai default ke 0.2 sesuai praktik umum
            step=0.05,
            help="Pilih persentase data yang akan digunakan untuk testing"
        )
        
        random_state = st.number_input(
            "Random State",
            min_value=0,
            value=32,
            help="Nilai untuk mengontrol pengacakan data untuk reproduktibilitas"
        )
        
        if st.button("Split Data"):
            if len(X) == 0:
                st.error("Data untuk splitting kosong. Harap pastikan ada data setelah pelabelan sentimen.")
                return

            # Lakukan pemisahan data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size,
                stratify=y, # Penting untuk menjaga proporsi kelas sentimen
                random_state=random_state
            )
            
            # Simpan hasil pemisahan di session state
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            # Tampilkan hasil
            st.success("Data berhasil displit!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Jumlah Data Train", value=len(X_train))
                st.write("Contoh data train:")
                st.write(pd.DataFrame({'Text': X_train, 'Sentimen': y_train}).head())
                
            with col2:
                st.metric("Jumlah Data Testing", value=len(X_test))
                st.write("Contoh data testing:")
                st.write(pd.DataFrame({'Text': X_test, 'Sentimen': y_test}).head())
            
            # Tombol untuk mengekspor data
            st.download_button(
                label="Unduh Data Train (X_train & y_train)",
                data=pd.DataFrame({'Text': X_train, 'Sentimen': y_train}).to_csv(index=False).encode('utf-8'),
                file_name='data_train.csv',
                mime='text/csv'
            )
            
            st.download_button(
                label="Unduh Data Testing (X_test & y_test)",
                data=pd.DataFrame({'Text': X_test, 'Sentimen': y_test}).to_csv(index=False).encode('utf-8'),
                file_name='data_testing.csv',
                mime='text/csv'
            )
    else:
        st.warning("Harap selesaikan proses pelabelan sentimen di tab sebelumnya terlebih dahulu.")
        st.info("Silakan ke tab 'Pelabelan Sentimen' untuk memproses data Anda.")