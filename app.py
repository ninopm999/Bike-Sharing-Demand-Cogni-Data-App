import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
import pickle
import datetime
import os
from scipy.stats.mstats import winsorize

# ===================================================================================
# Definisi Fungsi Pra-pemrosesan
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
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

MODEL_FILENAME = 'XGBoost_SKLearn_Pipeline_Final.pkl'
pipeline_model = load_pickled_model(MODEL_FILENAME)

# ===================================================================================
# Fungsi Utama Aplikasi
# ===================================================================================
def main():
    st.title("🚲 Prediksi Permintaan Sewa Sepeda")
    st.markdown("Selamat datang di aplikasi prediksi permintaan sepeda berbasis machine learning!")

    menu_options = {
        "🏠 Beranda": show_homepage,
        "⚙️ Aplikasi Prediksi": run_prediction_app,
        "📖 Info Model": show_model_info_page
    }
    choice = st.sidebar.radio("Pilih Halaman", list(menu_options.keys()), label_visibility="collapsed")

    if pipeline_model is None and choice == "⚙️ Aplikasi Prediksi":
        st.error("MODEL PREDIKSI GAGAL DIMUAT. Halaman prediksi tidak dapat ditampilkan.")
    else:
        menu_options[choice]()

# ===================================================================================
# Fungsi Halaman
# ===================================================================================
def show_homepage():
    st.header("🏠 Beranda")
    st.image("https://img.freepik.com/free-photo/row-parked-rental-bikes_53876-63261.jpg",
             caption="Inovasi Transportasi Perkotaan dengan Berbagi Sepeda",
             use_container_width=True)
    st.write("Aplikasi ini menggunakan model machine learning untuk memprediksi permintaan penyewaan sepeda berdasarkan waktu dan kondisi cuaca.")

def run_prediction_app():
    st.header("⚙️ Aplikasi Prediksi")
    with st.form(key='prediction_form'):
        date_input = st.date_input("Tanggal", datetime.date(2012, 12, 19))
        hour_input = st.slider("Jam (0–23)", 0, 23, 12)
        temp_input = st.number_input("Suhu (Celsius)", min_value=-10.0, max_value=50.0, value=25.0)
        humidity_input = st.slider("Kelembaban (%)", 0, 100, 60)
        windspeed_input = st.slider("Kecepatan Angin", 0.0, 1.0, 0.25)
        season_input = st.selectbox("Musim", [1, 2, 3, 4])
        weather_input = st.selectbox("Cuaca", [1, 2, 3, 4])
        submit_button = st.form_submit_button(label='Prediksi')

    if submit_button:
        dt_combined = datetime.datetime.combine(date_input, datetime.time(hour_input))
        df = pd.DataFrame({
            'datetime': [dt_combined],
            'season': [season_input],
            'holiday': [0],
            'workingday': [1],
            'weather': [weather_input],
            'temp': [temp_input],
            'humidity': [humidity_input],
            'windspeed': [windspeed_input]
        })

        df = preprocess_initial_features(df)
        df = create_cyclical_features(df)

        try:
            prediction = pipeline_model.predict(df)
            st.success(f"🎯 Estimasi Jumlah Peminjam Sepeda: {int(prediction[0]):,} orang")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")

def show_model_info_page():
    st.header("📖 Info Model")
    st.write("Model ini menggunakan algoritma XGBoost dan pipeline Scikit-learn untuk memproses data waktu dan cuaca menjadi prediksi jumlah peminjam sepeda.")

# ===================================================================================
# Jalankan Aplikasi
# ===================================================================================
if __name__ == '__main__':
    main()
