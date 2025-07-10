# app.py
import streamlit as st
import os
import sys

# Tambahkan direktori proyek ke PYTHONPATH
sys.path.append(os.path.dirname(__file__))

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Analisis Sentimen Komentar YouTube MPL Indonesia", layout="wide")

st.title("Analisis Sentimen Komentar YouTube MPL Indonesia")

# Inisialisasi session state (jika belum ada)
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'sentiment_results' not in st.session_state:
    st.session_state.sentiment_results = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'X_train_tfidf' not in st.session_state:
    st.session_state.X_train_tfidf = None
if 'X_test_tfidf' not in st.session_state:
    st.session_state.X_test_tfidf = None
if 'tfidf_vectorizer' not in st.session_state:
    st.session_state.tfidf_vectorizer = None


# Import halaman-halaman
from pages import preprocessing, pelabelan_sentimen, splitting_data, implementasi_tfidf, klasifikasi_naive_bayes

# Buat tab untuk navigasi antar halaman
tab_names = ["Preprocessing", "Pelabelan Sentimen", "Splitting Data", "Implementasi TF-IDF", "Klasifikasi Naive Bayes"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_names)

# Render konten untuk setiap tab
with tab1:
    preprocessing.show_page()

with tab2:
    pelabelan_sentimen.show_page()

with tab3:
    splitting_data.show_page()

with tab4:
    implementasi_tfidf.show_page()

with tab5:
    klasifikasi_naive_bayes.show_page()