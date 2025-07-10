# utils/data_loader.py
import pandas as pd
import os

# Konfigurasi path ke direktori dataset
DATASET_DIR = "dataset"
DICTIONARY_FILE = os.path.join(DATASET_DIR, "kamuskatabaku.xlsx")
STOPWORDS_FILE = os.path.join(DATASET_DIR, "stopwordbahasa.txt")
POSITIVE_LEXICON_FILE = os.path.join(DATASET_DIR, "kamus_positive.xlsx")
NEGATIVE_LEXICON_FILE = os.path.join(DATASET_DIR, "kamus_negative.xlsx")

def load_dictionary():
    """Memuat kamus kata baku dari file Excel."""
    try:
        if os.path.exists(DICTIONARY_FILE):
            df = pd.read_excel(DICTIONARY_FILE)
            word_dict = {}
            for index, row in df.iterrows():
                if row[0] not in word_dict:
                    word_dict[row[0]] = row[1]
            return word_dict
        else:
            return None
    except Exception as e:
        print(f"Error saat memuat kamus kata baku: {e}")
        return None

def load_stopwords():
    """Memuat daftar stopword dari file teks."""
    try:
        if os.path.exists(STOPWORDS_FILE):
            with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
                stopwords = f.read().splitlines()
            return stopwords
        else:
            return []
    except Exception as e:
        print(f"Error saat memuat stopword: {e}")
        return []

def load_lexicon(lexicon_type="positive"):
    """Memuat leksikon sentimen (positif atau negatif) dari file Excel."""
    if lexicon_type == "positive":
        file_path = POSITIVE_LEXICON_FILE
    elif lexicon_type == "negative":
        file_path = NEGATIVE_LEXICON_FILE
    else:
        raise ValueError("Tipe leksikon tidak valid. Gunakan 'positive' atau 'negative'.")

    try:
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            lexicon_dict = {}
            for index, row in df.iterrows():
                if row[0] not in lexicon_dict:
                    lexicon_dict[row[0]] = row[1]
            return lexicon_dict
        else:
            return None
    except Exception as e:
        print(f"Error saat memuat leksikon {lexicon_type}: {e}")
        return None