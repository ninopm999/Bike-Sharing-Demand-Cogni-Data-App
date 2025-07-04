import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
import pickle
import datetime
import os
from scipy.stats.mstats import winsorize

# ===================================================================================
# Definisi Fungsi Pra-pemrosesan (Sama seperti di Notebook Colab)
# ===================================================================================
def winsorize_series_robust(df_or_series, column_name=None, limits=(0.01, 0.01)):
    if isinstance(df_or_series, pd.DataFrame):
        if column_name is None or column_name not in df_or_series.columns:
            return df_or_series
        series_to_winsorize = df_or_series[column_name].copy()
    elif isinstance(df_or_series, pd.Series):
        series_to_winsorize = df_or_series.copy()
        column_name = df_or_series.name
    else:
        raise ValueError("Input harus DataFrame atau Series Pandas.")

    winsorized_array = winsorize(series_to_winsorize, limits=limits)

    if isinstance(df_or_series, pd.DataFrame):
        df_out = df_or_series.copy()
        df_out[column_name] = winsorized_array
        return df_out
    else:
        return pd.Series(winsorized_array, name=column_name, index=df_or_series.index)

def preprocess_initial_features(input_df):
    df = input_df.copy()
    if 'datetime' in df.columns:
        df['hour_val'] = df['datetime'].dt.hour
        df['month_val'] = df['datetime'].dt.month
        df['weekday_val'] = df['datetime'].dt.weekday
        df['day'] = df['datetime'].dt.day
        df['year_cat'] = df['datetime'].dt.year.astype(str)
        df['dayofyear'] = df['datetime'].dt.dayofyear
    if 'atemp' in df.columns:
        df = df.drop('atemp', axis=1, errors='ignore')
    return df

def create_cyclical_features(input_df):
    df = input_df.copy()
    if 'hour_val' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_val']/24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_val']/24.0)
    if 'month_val' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month_val']/12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['month_val']/12.0)
    if 'weekday_val' in df.columns:
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday_val']/7.0)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday_val']/7.0)
    return df

# ===================================================================================
# Konfigurasi Halaman Streamlit
# ===================================================================================
st.set_page_config(
    page_title="Prediksi Sewa Sepeda COGNIDATA",
    page_icon="🚲",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'mailto:fauzanzahid720@gmail.com',
        'Report a bug': "mailto:fauzanzahid720@gmail.com",
        'About': "### Aplikasi Prediksi Permintaan Sepeda\nTim COGNIDATA\nPowered by XGBoost & Scikit-learn."
    }
)

# ===================================================================================
# Muat Model
# ===================================================================================
@st.cache_resource
def load_pickled_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"File model '{model_path}' tidak ditemukan di path yang diharapkan. Pastikan file ada di direktori yang sama dengan aplikasi.")
        return None
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"File model '{model_path}' tidak ditemukan. Pastikan file ada di direktori yang sama dengan aplikasi.")
        return None
    except pickle.UnpicklingError as e:
        st.error(f"Terjadi kesalahan saat unpickling model: {e}. File model mungkin rusak atau tidak kompatibel.")
        return None
    except ModuleNotFoundError as e:
        st.error(f"Terjadi kesalahan saat memuat model (ModuleNotFoundError): {e}. Pastikan semua library yang dibutuhkan model ada di requirements.txt.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan umum saat memuat model: {e}")
        return None

MODEL_FILENAME = 'XGBoost_SKLearn_Pipeline_Final.pkl'
pipeline_model = load_pickled_model(MODEL_FILENAME)

# ===================================================================================
# HTML Template Placeholder
# ===================================================================================
PRIMARY_BG_COLOR = "#003366"
PRIMARY_TEXT_COLOR = "#FFFFFF"
ACCENT_COLOR = "#FFD700"
HTML_BANNER = """..."""
HTML_FOOTER = """..."""

# ===================================================================================
# Fungsi Utama Aplikasi
# ===================================================================================
def main():
    stc.html(HTML_BANNER, height=170)
    menu_options = {
        "🏠 Beranda": show_homepage,
        "⚙️ Aplikasi Prediksi": run_prediction_app,
        "📖 Info Model": show_model_info_page
    }
    # ✅ FIX: Tambahkan label agar tidak kosong
    choice = st.sidebar.radio("Navigasi", list(menu_options.keys()), label_visibility="collapsed")

    if pipeline_model is None and choice == "⚙️ Aplikasi Prediksi":
        st.error("MODEL PREDIKSI GAGAL DIMUAT. Halaman prediksi tidak dapat ditampilkan.")
        st.markdown("Silakan periksa file model dan log, atau hubungi administrator.")
    else:
        menu_options[choice]()

    stc.html(HTML_FOOTER, height=70)

# ===================================================================================
# Placeholder fungsi halaman
# ===================================================================================
def show_homepage():
    st.markdown("## Selamat Datang di Dasbor Prediksi Permintaan Sepeda!")
    st.image("https://img.freepik.com/free-photo/row-parked-rental-bikes_53876-63261.jpg", 
             caption="Inovasi Transportasi Perkotaan dengan Berbagi Sepeda", use_container_width=True)
    # (Tambahkan konten lainnya sesuai kebutuhan)

def run_prediction_app():
    st.markdown("## ⚙️ Masukkan Parameter untuk Prediksi")
    st.info("Form input dan prediksi model ditampilkan di sini.")
    # (Placeholder fungsi prediksi)

def show_model_info_page():
    st.markdown("## 📖 Info Model")
    st.write("Model ini menggunakan XGBoost dan pipeline Scikit-learn untuk memprediksi permintaan sepeda.")

# ===================================================================================
# Jalankan Aplikasi
# ===================================================================================
if __name__ == '__main__':
    main()
