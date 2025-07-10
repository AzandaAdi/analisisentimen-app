# pages/preprocessing.py
import streamlit as st
import pandas as pd
import os
import sys

# Tambahkan direktori induk untuk mengimpor modul dari utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import text_preprocessing, data_loader

def show_page():
    st.header("Preprocessing")
    
    uploaded_file = st.file_uploader("Pilih file CSV untuk preprocessing", type="csv", key="preprocess_uploader")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Data Asli")
        st.write(df.head(5))
        
        if st.button("Mulai Preprocessing", key="start_preprocess"):
            st.subheader("Tahapan Preprocessing")
            
            # 1. Case Folding
            st.write("### 1. Case Folding")
            df['Text Case Folding'] = df['Komentar'].apply(text_preprocessing.case_folding)
            st.write(df['Text Case Folding'].head(5))
            
            # Cleansing
            st.write("### 2. Pembersihan Teks (Cleansing)")
            df['Text Case Folding'] = df['Text Case Folding'].apply(text_preprocessing.remove_tweet_special)
            df['Text Case Folding'] = df['Text Case Folding'].apply(text_preprocessing.remove_number)
            df['Text Case Folding'] = df['Text Case Folding'].apply(text_preprocessing.remove_punctuation)
            df['Text Case Folding'] = df['Text Case Folding'].apply(text_preprocessing.remove_whitespace_LT)
            df['Text Case Folding'] = df['Text Case Folding'].apply(text_preprocessing.remove_whitespace_multiple)
            df['Text Case Folding'] = df['Text Case Folding'].apply(text_preprocessing.remove_singl_char)
            st.write(df['Text Case Folding'].head(5))

            # 3. Tokenizing
            st.write("### 3. Tokenizing")
            df['Text Tokenizing'] = df['Text Case Folding'].apply(text_preprocessing.tokenize_text)
            st.write(df['Text Tokenizing'].head(5))
            
            # 4. Normalisasi
            st.write("### 4. Normalisasi")
            normalizad_word_dict = data_loader.load_dictionary()
            if normalizad_word_dict:
                df['Text Normalization'] = df['Text Tokenizing'].apply(lambda x: text_preprocessing.normalize_terms(x, normalizad_word_dict))
                st.write(df['Text Normalization'].head(5))
            else:
                st.warning("File kamus tidak ditemukan atau terjadi kesalahan. Melewati tahap normalisasi.")
                df['Text Normalization'] = df['Text Tokenizing']
            
            # 5. Stopword Removal
            st.write("### 5. Penghapusan Stopword")
            list_stopwords = set(data_loader.load_stopwords())
            
            # Tambahkan stopword tambahan yang spesifik untuk domain ini
            list_stopwords_additional = [
                "yg", "dg", "rt", "dgn", "ny", 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang', 'gak', 'ga', 'krn', 'nya', 'nih', 'sih',
                'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 'jd', 'jgn', 'sdh', 'aja', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                '&amp', 'yah', 'sdgkan', 'sdg', 'emg', 'sm', 'pls', 'mlu', 'ken', 'allah', 'brb', 'btw', 'b/c', 'cod', 'cmiiw', 'fyi',
                'gg', 'ggwp', 'idk', 'ikr', 'lol', 'ootd', 'lmao', 'oot', 'pap', 'otw', 'tfl', 'vc', 'ygy', 'href', 'evos', 'rrq', 'wkwkw', 
                'awokwkkwowow', 'wkwkwk', 'wkwkwkwkwkwkwkwkwkwkkwkw', 'hahahahaha', 'kwowkwok', 'wkwowkwowkwow', 'kwokwokwokwokwow',
                'wkkw', 'wkwk', 'akwokaow', 'awokawok', 'hahah', 'kwwkjwkw', 'kwwkwkkw', 'awokawokawok', 'hahahaha', 'wkwkkw', 'aowkwowkwowks',
                'wkwkwkwkwk', 'hahhaa', 'wkwkwkwkkw', 'awowkowwkokk', 'wkwkwkwkwkwkw', 'awokwok', 'wkjwkw', 'wkwkkwkwk', 'weekwwowke',
                'wkkakwkwkwkw', 'awowkowk', 'wkwkwkwk', 'wkwkwl', 'awowkoaowkw', 'hahaha', 'awokaowk', 'awaokawaokawaok', 'hahahaa', 'kwkwkw',
                'wkwkwkwkwkwkwk', 'kwkwkwkw', 'mpl', 'season', 'msc', 'prime', 'onic', 'sonic', 'liquid', 'tlid', 'kongdom', 'grand', 'final',
                'aura', 'mpl', 'kingdom', 'khondom', 'kairi', 'sze', 'ling', 'ppq', 'abang', 'angela', 'favian', 'kondom', 'pepeqi', 'skayar',
                'skylar', 'poke', 'yawi', 'id', 'ph', 'mseries', 'rinz', 'aboy', 'wkwkwkw', 'ml', 'nxl', 'series', 'mobile', 'legends', 'indonesia',
                'sutsujin', 'kb', 'ixia', 'fnatic', 'galaxy', 'kingdoms', 'chou', 'kezchute', 'cavalery', 'bang', 'cavalry', 'blacklist', 'elelki',
                'aurafire', 'inter', 'malay', 'ylid', 'idok', 'fams', 'djsbjdbdjionicccccccccccccc', 'acil', 'aowkawok', 'alexis', 'balexis', 'fyp',
                'xixixixixi', 'wkwkkwwkwkwkw', 'aowkwkkw', 'awokwkwk', 'awokowkowkowk', 'awokwokowkowkwok', 'aowkow', 'wkwkwkwkw', 'wokwokwok',
                'awowkwk', 'wkwkwkwkbrsyarat', 'wkowkowkobrkasian', 'wkwkwkkwkwkwkwkwk', 'toy', 'adi', 'kiboy', 'rine', 'albert', 'widy', 'yehiskiel', 'geek',
                'btr', 'baloy', 'eman', 'lukerrq', 'luke', 'dyren', 'chiko', 'joy', 'nnael', 'yeb', 'lutpi', 'sanz', 'navi', 'savero', 'hoshi', 'granger',
                'ekwkwk', 'yss', 'nana', 'oniccccccc', 'lutpiii', 'arsenal', 'kiboi', 'brusko', 'esl', 'kongdm', 'toyy', 'ff', 'maniac', 'gold', 'lane',
                'sanzz', 'keeeeee', 'poookee', 'gblok', 'lukas', 'hayabusa', 'konoha', 'real', 'madrid', 'awokawoak', 'kaisze', 'hayadia', 'rank', 'epic',
                'viva', 'toyo', 'fam', 'onik', 'lemon', 'rrqdyren', 'onicsport', 'szeeeee', 'awwww', 'sanzzz', 'oniiiiiiiiic', 'jiraya', 'tobirama', 'madara',
                'katsuyu', 'ler', 'filipina', 'kongdon', 'baxia', 'yve', 'baxilit', 'sanzzzz', 'kindom', 'oniccc', 'kingdoom', 'khecut', 'saaaaaaaanzz', 'jungler',
                'mid', 'exp', 'roam', 'ewc', 'wkwkk', 'mvp', 'michell', 'kongdong', 'tiktok', 'assasin', 'esport', 'tank', 'ass', 'xp', 'keszchut', 'haya', 'jung',
                'diren', 'homebois', 'malaysia', 'selangor', 'red', 'giants', 'immortal', 'recall', 'skill', 'heal', 'minions', 'hero', 'draft', 'pick', 'paveus', 'hazell',
                'level', 'ngetroll', 'grangernya', 'pharsa', 'match', 'ban', 'skaylar', 'gatot', 'familia', 'alterego', 'lord', 'defend', 'comeback', 'echo',
                'dyrrennnn', 'ppk', 'claude', 'tigreal', 'kigdom', 'pepeq', 'dyren', 'lupi', 'wkwkwkkwkww', 'wkwkkk', 'wkwkwkkkw', 'skyler', 'riyadh', 'lerrr',
                'ytta', 'swaylow', 'yebb', 'epos', 'huzle', 'pulung', 'qontol', 'memeqi', 'toyloll', 'kzcute', 'srg', 'roamer', 'sanzzzzzzzzz', 'wkwkwwk', 'grenjerrr',
                'cihuyyy', 'riyad', 'yuzhong', 'helcurt', 'jawhead', 'benedetta', 'fighter', 'brawl', 'dyrennn', 'bened', 'pemsdog', 'om', 'wawa', 'game', 'donkey', 'coach'
            ]
            list_stopwords.update(list_stopwords_additional)
            
            df['Text Filtering'] = df['Text Normalization'].apply(lambda x: text_preprocessing.remove_stopwords(x, list_stopwords))
            st.write(df['Text Filtering'].head(5))
            
            # 6. Stemming
            st.write("### 6. Stemming")
            with st.spinner('Sedang melakukan stemming... Ini mungkin memakan waktu lama.'):
                df['Text Stemming'] = df['Text Filtering'].swifter.apply(text_preprocessing.stem_tokens)
            st.write(df['Text Stemming'].head(5))
            
            # Simpan data yang sudah diproses di session state
            st.session_state.preprocessed_data = df
            
            st.success("Preprocessing Selesai!")
            
            st.download_button(
                label="Unduh Data Hasil Preprocessing",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='data_hasil_preprocessing.csv',
                mime='text/csv'
            )
    else:
        st.info("Silakan unggah file CSV untuk memulai preprocessing")