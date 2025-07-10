# pages/klasifikasi_naive_bayes.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import sys
import os

# Tambahkan direktori induk untuk mengimpor modul dari utils (jika diperlukan)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def show_page():
    st.header("Klasifikasi dengan Naive Bayes")
    
    # Cek apakah data TF-IDF sudah tersedia di session state
    if 'X_train_tfidf' not in st.session_state or 'X_test_tfidf' not in st.session_state:
        st.warning("Harap lakukan transformasi TF-IDF di tab 'Implementasi TF-IDF' terlebih dahulu.")
        st.stop()
    
    if st.button("Latih Model Naive Bayes"):
        with st.spinner('Melatih model...'):
            # Ambil data dari session state
            X_train_tfidf = st.session_state.X_train_tfidf
            X_test_tfidf = st.session_state.X_test_tfidf
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test
            
            # Latih model Multinomial Naive Bayes
            # Parameter alpha dapat disesuaikan untuk smoothing
            model = MultinomialNB(alpha=0.3) 
            model.fit(X_train_tfidf, y_train)
            
            # Lakukan prediksi pada data testing
            predictions = model.predict(X_test_tfidf)
            
            # Hitung metrik evaluasi
            cm = confusion_matrix(y_test, predictions, labels=['Negatif', 'Netral', 'Positif'])
            cr = classification_report(y_test, predictions, output_dict=True, labels=['Negatif', 'Netral', 'Positif'])
            accuracy = accuracy_score(y_test, predictions)
            
            # Tampilkan hasil evaluasi model
            st.subheader("Hasil Evaluasi Model")
            
            # 1. Confusion Matrix
            st.markdown("**Confusion Matrix:**")
            cm_df = pd.DataFrame(cm, 
                               index=['Aktual Negatif', 'Aktual Netral', 'Aktual Positif'],
                               columns=['Prediksi Negatif', 'Prediksi Netral', 'Prediksi Positif'])
            st.dataframe(cm_df.style.set_properties(**{'text-align': 'center'}))
            
            # 2. Classification Report
            st.markdown("**Classification Report:**")
            
            # Buat DataFrame dari classification report
            class_report_df = pd.DataFrame(cr).transpose()
            
            # Ubah format untuk menampilkan akurasi dan rata-rata dengan benar
            st.dataframe(class_report_df.style.format("{:.2f}", subset=['precision', 'recall', 'f1-score']))
            
            # 3. Akurasi akhir
            st.markdown(f"**Akurasi Model:** {accuracy:.4f}")
            
            # Visualisasi Confusion Matrix
            st.subheader("Visualisasi Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title('Confusion Matrix')
            plt.ylabel('Aktual')
            plt.xlabel('Prediksi')
            st.pyplot(fig)
            plt.close()
            
            # Tambahan: Perbandingan Aktual vs Prediksi
            st.subheader("Perbandingan Aktual vs Prediksi")
            
            # Buat dataframe untuk perbandingan
            comparison_df = pd.DataFrame({
                'Text': st.session_state.X_test,
                'Actual': st.session_state.y_test,
                'Predicted': predictions
            })
            
            # Tampilkan 5 contoh pertama
            st.dataframe(comparison_df.head(5))
            
            # Visualisasi distribusi aktual vs prediksi
            st.subheader("Distribusi Aktual vs Prediksi")
            
            # Gabungkan data untuk visualisasi
            plot_df = comparison_df.melt(id_vars=['Text'], 
                                        value_vars=['Actual', 'Predicted'],
                                        var_name='Type', 
                                        value_name='Sentiment')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=plot_df, x='Sentiment', hue='Type', palette=['#1f77b4', '#ff7f0e'], order=['Negatif', 'Netral', 'Positif'])
            plt.title('Perbandingan Distribusi Sentimen Aktual dan Prediksi')
            plt.xlabel('Sentimen')
            plt.ylabel('Jumlah')
            plt.legend(title='Tipe')
            st.pyplot(fig)
            plt.close()
            
            # Tombol unduh model
            st.download_button(
                label="Unduh Model (naive_bayes_model.pkl)",
                data=pickle.dumps(model),
                file_name='naive_bayes_model.pkl',
                mime='application/octet-stream'
            )
            
            # Tombol unduh hasil prediksi
            st.download_button(
                label="Unduh Hasil Prediksi (hasil_prediksi.csv)",
                data=comparison_df.to_csv(index=False).encode('utf-8'),
                file_name='hasil_prediksi.csv',
                mime='text/csv'
            )