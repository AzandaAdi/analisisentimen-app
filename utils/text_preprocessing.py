# utils/text_preprocessing.py
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter # Pastikan swifter terinstal: pip install swifter
import ast # Untuk ast.literal_eval

# Tambahkan path NLTK lokal (relative ke file ini)
nltk_data_path = os.path.join(os.path.dirname(__file__), '..', 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Pastikan sumber daya NLTK sudah diunduh
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Inisialisasi stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Cache untuk stemming agar lebih efisien
_stemming_cache = {}

def remove_tweet_special(text):
    """Menghapus karakter khusus dari teks."""
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    text = text.encode('ascii', 'replace').decode('ascii')
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    return text.replace("http://", " ").replace("https://", " ")

def remove_number(text):
    """Menghapus angka dari teks."""
    return re.sub(r"\d+", "", text)

def remove_punctuation(text):
    """Menghapus tanda baca dari teks."""
    return text.translate(str.maketrans("","",string.punctuation))

def remove_whitespace_LT(text):
    """Menghapus spasi di awal dan akhir teks."""
    return text.strip()

def remove_whitespace_multiple(text):
    """Menghapus spasi berlebih dari teks."""
    return re.sub('\s+',' ',text)

def remove_singl_char(text):
    """Menghapus karakter tunggal dari teks."""
    return re.sub(r"\b[a-zA-Z]\b", "", text)

def case_folding(text):
    """Melakukan case folding (mengubah teks menjadi huruf kecil)."""
    return text.lower()

def tokenize_text(text):
    """Melakukan tokenisasi teks."""
    return word_tokenize(text)

def normalize_terms(tokens, normalized_word_dict):
    """Melakukan normalisasi kata berdasarkan kamus."""
    if normalized_word_dict:
        return [normalized_word_dict.get(term, term) for term in tokens]
    return tokens

def remove_stopwords(tokens, list_stopwords):
    """Menghapus stopword dari daftar token."""
    return [word for word in tokens if word not in list_stopwords]

def stem_tokens(tokens):
    """Melakukan stemming pada daftar token."""
    stemmed_tokens = []
    for term in tokens:
        if term not in _stemming_cache:
            _stemming_cache[term] = stemmer.stem(term)
        stemmed_tokens.append(_stemming_cache[term])
    return stemmed_tokens

def clean_and_convert_list_string(text_col):
    """
    Membersihkan kolom yang mungkin berisi string representasi list atau data kosong.
    Mengkonversi string representasi list menjadi list Python aktual.
    """
    cleaned_data = []
    for item in text_col:
        if isinstance(item, str):
            item = item.strip()
            if item.startswith('[') and item.endswith(']'):
                try:
                    evaled_item = ast.literal_eval(item)
                    if isinstance(evaled_item, list) and evaled_item: # Pastikan list tidak kosong
                        cleaned_data.append(evaled_item)
                    else:
                        cleaned_data.append([]) # Tambahkan list kosong jika dievaluasi sebagai list kosong
                except (ValueError, SyntaxError):
                    cleaned_data.append([item]) # Jika bukan list, perlakukan sebagai satu token
            elif item: # Jika bukan string list dan tidak kosong
                cleaned_data.append([item]) # Perlakukan sebagai satu token
            else:
                cleaned_data.append([]) # Jika string kosong
        elif isinstance(item, list) and item: # Jika sudah list dan tidak kosong
            cleaned_data.append(item)
        else:
            cleaned_data.append([]) # Jika None atau list kosong
    return cleaned_data
