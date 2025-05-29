import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
import pickle
import datetime
import os
from scipy.stats.mstats import winsorize # Pastikan scipy ada di requirements.txt

# ===================================================================================
# Definisi Fungsi Pra-pemrosesan (Sama seperti di Notebook Colab)
# ===================================================================================
def winsorize_series_robust(df_or_series, column_name=None, limits=(0.01, 0.01)):
    if isinstance(df_or_series, pd.DataFrame):
        if column_name is None or column_name not in df_or_series.columns:
            # Jika kolom tidak ada, kembalikan DataFrame asli tanpa error
            # print(f"Peringatan: Kolom '{column_name}' untuk winsorizing tidak ditemukan di DataFrame.")
            return df_or_series
        series_to_winsorize = df_or_series[column_name].copy()
    elif isinstance(df_or_series, pd.Series):
        series_to_winsorize = df_or_series.copy()
        column_name = df_or_series.name # Ambil nama kolom dari series
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
        df['weekday_val'] = df['datetime'].dt.weekday # Senin=0, Minggu=6
        df['day'] = df['datetime'].dt.day
        df['year_cat'] = df['datetime'].dt.year.astype(str) 
        df['dayofyear'] = df['datetime'].dt.dayofyear
        # Kolom 'datetime' asli akan di-drop nanti setelah semua fitur turunan dibuat,
        # sebelum dimasukkan ke ColumnTransformer jika CT tidak mengharapkannya.
    if 'atemp' in df.columns:
        df = df.drop('atemp', axis=1, errors='ignore')
    # Kolom 'casual' dan 'registered' tidak ada di input dari Streamlit
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
    """Memuat model dari file pickle."""
    if not os.path.exists(model_path):
        st.error(f"File model '{model_path}' tidak ditemukan di path yang diharapkan. Pastikan file ada di direktori yang sama dengan aplikasi.")
        return None
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print(f"Model berhasil dimuat dari: {model_path}")
        return model
    except FileNotFoundError: # Seharusnya sudah ditangani oleh os.path.exists
        st.error(f"File model '{model_path}' tidak ditemukan. Pastikan file ada di direktori yang sama dengan aplikasi.")
        return None
    except pickle.UnpicklingError as e:
        st.error(f"Terjadi kesalahan saat unpickling model: {e}. File model mungkin rusak atau tidak kompatibel.")
        return None
    except ModuleNotFoundError as e:
        st.error(f"Terjadi kesalahan saat memuat model (ModuleNotFoundError): {e}. Pastikan semua library yang dibutuhkan model ada di requirements.txt.")
        st.error("Jika Anda baru saja menghapus PyCaret dari requirements, pastikan model .pkl Anda tidak lagi memiliki dependensi padanya.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan umum saat memuat model: {e}")
        return None

MODEL_FILENAME = 'XGBoost_SKLearn_Pipeline_Final.pkl' # Sesuai nama file dari notebook
pipeline_model = load_pickled_model(MODEL_FILENAME)

# ===================================================================================
# HTML Templates (Tidak ada perubahan)
# ===================================================================================
PRIMARY_BG_COLOR = "#003366"
PRIMARY_TEXT_COLOR = "#FFFFFF"
ACCENT_COLOR = "#FFD700"
HTML_BANNER = f"""...""" # (Isi sama seperti sebelumnya)
HTML_FOOTER = f"""...""" # (Isi sama seperti sebelumnya)

# ===================================================================================
# Fungsi Utama Aplikasi (Tidak ada perubahan signifikan)
# ===================================================================================
def main():
    stc.html(HTML_BANNER, height=170)
    menu_options = {
        "🏠 Beranda": show_homepage,
        "⚙️ Aplikasi Prediksi": run_prediction_app,
        "📖 Info Model": show_model_info_page
    }
    st.sidebar.title("Navigasi Aplikasi")
    choice = st.sidebar.radio("", list(menu_options.keys()), label_visibility="collapsed")

    if pipeline_model is None and choice == "⚙️ Aplikasi Prediksi":
        st.error("MODEL PREDIKSI GAGAL DIMUAT. Halaman prediksi tidak dapat ditampilkan.")
        st.markdown("Silakan periksa file model dan log, atau hubungi administrator.")
    else:
        menu_options[choice]()
    
    stc.html(HTML_FOOTER, height=70)

# ===================================================================================
# Halaman Beranda (Tidak ada perubahan)
# ===================================================================================
def show_homepage():
    # ... (Isi sama seperti sebelumnya) ...
    st.markdown("## Selamat Datang di Dasbor Prediksi Permintaan Sepeda!")
    st.markdown("""
    Aplikasi ini adalah alat bantu cerdas untuk memprediksi jumlah total sepeda yang kemungkinan akan disewa dalam satu jam tertentu. 
    Dengan memanfaatkan data historis dan model machine learning canggih, kami bertujuan untuk memberikan estimasi yang dapat diandalkan 
    untuk membantu Anda dalam perencanaan dan operasional bisnis berbagi sepeda.

    ---
    #### Mengapa Prediksi Ini Penting?
    - Optimalisasi Stok Sepeda
    - Efisiensi Operasional dan Penjadwalan Perawatan
    - Peningkatan Kepuasan Pelanggan dengan Ketersediaan Sepeda
    - Dasar Strategi Pemasaran dan Promosi

    ---
    #### Cara Kerja Aplikasi:
    1.  Pilih "**⚙️ Aplikasi Prediksi**" dari menu navigasi di sebelah kiri.
    2.  Masukkan detail parameter waktu, kondisi cuaca, dan lingkungan pada formulir yang disediakan.
    3.  Klik tombol "**Prediksi Sekarang**" untuk melihat estimasi jumlah sewa.
    
    Jelajahi juga halaman "**📖 Info Model**" untuk memahami lebih dalam tentang teknologi di balik prediksi ini.

    ---
    #### Sumber Data:
    Dataset yang digunakan dalam pengembangan model ini berasal dari kompetisi Kaggle:
    [Bike Sharing Demand - Kaggle](https://www.kaggle.com/competitions/bike-sharing-demand/data)
    """)
    
    st.image("https://img.freepik.com/free-photo/row-parked-rental-bikes_53876-63261.jpg", 
             caption="Inovasi Transportasi Perkotaan dengan Berbagi Sepeda", use_column_width=True)


# ===================================================================================
# Halaman Aplikasi Prediksi (PERBAIKAN UTAMA DI SINI)
# ===================================================================================
def run_prediction_app():
    st.markdown("## ⚙️ Masukkan Parameter untuk Prediksi")
    
    if pipeline_model is None:
        # Pesan error sudah ditangani di fungsi main atau saat load_pickled_model
        return

    st.markdown("#### 📅 Informasi Waktu")
    col_date, col_time = st.columns([1, 1]) 
    with col_date:
        input_date = st.date_input("Tanggal Prediksi", datetime.date.today() + datetime.timedelta(days=1), 
                                   min_value=datetime.date.today(),
                                   help="Pilih tanggal untuk prediksi.")
    with col_time:
        input_time = st.time_input("Waktu Prediksi", datetime.time(10, 0), 
                                   help="Pilih waktu (jam & menit) untuk prediksi.", step=datetime.timedelta(hours=1)) # Step 1 jam
    dt_object = datetime.datetime.combine(input_date, input_time)
    
    is_working_day_auto = 1 if dt_object.weekday() < 5 else 0 
    workingday_display_text = "Hari Kerja" if is_working_day_auto == 1 else "Akhir Pekan/Libur"
    st.info(f"Prediksi untuk: **{dt_object.strftime('%A, %d %B %Y, pukul %H:%M')}** ({workingday_display_text})")
    
    st.markdown("---")

    st.markdown("#### 📋 Kondisi & Lingkungan")
    col_kondisi1, col_kondisi2, col_lingkungan = st.columns([2, 2, 2.5]) 

    with col_kondisi1: 
        st.markdown("##### Musim & Liburan")
        season_options = {1: "Musim Semi", 2: "Musim Panas", 3: "Musim Gugur", 4: "Musim Dingin"}
        current_month = dt_object.month
        if current_month in [3, 4, 5]: default_season = 1
        elif current_month in [6, 7, 8]: default_season = 2
        elif current_month in [9, 10, 11]: default_season = 3
        else: default_season = 4
        
        season = st.selectbox("Musim", options=list(season_options.keys()), 
                              format_func=lambda x: f"{season_options[x]} (Kode: {x})", 
                              index=list(season_options.keys()).index(default_season),
                              key="season_select")
        
        holiday = st.radio("Hari Libur Nasional?", (0, 1), format_func=lambda x: "Ya" if x == 1 else "Tidak", 
                           index=0, horizontal=True, key="holiday_radio")

    with col_kondisi2: 
        st.markdown("##### Status Hari & Cuaca")
        workingday = st.radio("Hari Kerja Aktual?", (0, 1), 
                              format_func=lambda x: "Ya" if x == 1 else "Tidak", 
                              index=is_working_day_auto, horizontal=True, key="workingday_radio",
                              help=f"Terdeteksi otomatis sebagai '{workingday_display_text}', Anda bisa mengubahnya jika perlu.")
        
        weather_options = {1: "Cerah/Sedikit Berawan", 2: "Kabut/Berawan Sebagian", 3: "Hujan/Salju Ringan", 4: "Cuaca Ekstrem"}
        weather = st.selectbox("Kondisi Cuaca", options=list(weather_options.keys()), 
                               format_func=lambda x: f"{weather_options[x]} (Kode: {x})", 
                               index=0, key="weather_select")

    with col_lingkungan: 
        st.markdown("##### Parameter Lingkungan")
        temp = st.number_input("Suhu (°C)", min_value=-20.0, max_value=50.0, value=25.0, step=0.5, format="%.1f", key="temp_input")
        humidity = st.slider("Kelembapan (%)", min_value=0, max_value=100, value=60, step=1, key="humidity_slider")
        windspeed = st.number_input("Kecepatan Angin (km/jam)", min_value=0.0, max_value=80.0, value=10.0, step=0.1, format="%.1f", key="windspeed_input")

    st.markdown("---")
    
    if st.button("Prediksi Jumlah Sewa Sekarang!", use_container_width=True, type="primary", key="predict_button_main"):
        # 1. Buat DataFrame awal dari input pengguna
        input_data_dict = {
            'datetime': [dt_object], 
            'season': [season], 
            'holiday': [holiday],
            'workingday': [workingday], 
            'weather': [weather], 
            'temp': [temp],
            'humidity': [humidity], 
            'windspeed': [windspeed],
            'atemp': [temp] # Tambahkan atemp, nanti akan di-drop oleh preprocess_initial_features
        }
        input_df_raw = pd.DataFrame(input_data_dict)

        # 2. Terapkan pra-pemrosesan dan rekayasa fitur seperti di notebook
        #    Ini akan membuat kolom seperti 'hour_val', 'month_val', 'year_cat', 'hour_sin', 'windspeed' (mungkin sudah di-winsorize), dll.
        #    dan menghapus 'datetime' asli serta 'atemp'.
        try:
            df_p1 = preprocess_initial_features(input_df_raw.copy())
            df_p2 = create_cyclical_features(df_p1)
            
            # Terapkan winsorizing manual, pastikan kolom ada
            input_df_engineered = df_p2.copy()
            if 'humidity' in input_df_engineered.columns:
                input_df_engineered = winsorize_series_robust(input_df_engineered, column_name='humidity', limits=(0.01, 0.01))
            if 'windspeed' in input_df_engineered.columns:
                input_df_engineered = winsorize_series_robust(input_df_engineered, column_name='windspeed', limits=(0.05, 0.05))

            # 3. Drop kolom 'datetime' jika masih ada, karena pipeline dilatih tanpa itu sebagai input langsung ke CT
            if 'datetime' in input_df_engineered.columns:
                input_df_engineered = input_df_engineered.drop('datetime', axis=1)

            # 4. Pastikan urutan kolom dan nama kolom sesuai dengan yang diharapkan pipeline
            #    (saat X_train_engineered dimasukkan ke .fit() di notebook)
            #    ColumnTransformer di dalam pipeline akan memilih kolom berdasarkan nama.
            
            # Ambil daftar kolom yang diharapkan dari fitur yang digunakan untuk melatih pipeline di notebook
            # Ini harus sama dengan X_train_engineered.columns.tolist() di notebook SEBELUM .fit() pipeline
            # Urutan mungkin penting jika ada langkah selain ColumnTransformer di awal pipeline Anda.
            # Untuk ColumnTransformer, yang penting nama kolomnya ada.
            
            # Kolom yang diharapkan oleh ColumnTransformer (preprocessor_ct) di pipeline Anda:
            # (Ini harus diambil dari definisi `numeric_features_for_scaling` dan 
            # `categorical_features_for_ohe` di notebook Anda setelah semua rekayasa fitur manual)
            expected_cols_for_ct = [
                'temp', 'humidity', 'windspeed', 'day', 'dayofyear',
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                'weekday_sin', 'weekday_cos', 'season', 'holiday', 
                'workingday', 'weather', 'hour_val', 'month_val', 
                'weekday_val', 'year_cat'
            ]
            
            # Pastikan semua kolom yang diharapkan ada, jika tidak, bisa menyebabkan error saat predict
            # Atau, jika pipeline Anda cukup robust (misal, ColumnTransformer dengan remainder='drop'),
            # ia mungkin hanya akan menggunakan kolom yang ditemukannya.
            # Untuk keamanan, kita bisa membuat DataFrame final dengan kolom yang diharapkan saja.
            
            missing_cols = set(expected_cols_for_ct) - set(input_df_engineered.columns)
            if missing_cols:
                st.error(f"Rekayasa fitur di app.py tidak menghasilkan semua kolom yang diharapkan. Kolom yang hilang: {missing_cols}")
                st.dataframe(input_df_engineered) # Tampilkan untuk debug
                return

            # Pilih hanya kolom yang diharapkan oleh ColumnTransformer (jika CT Anda strict)
            # input_features_for_pipeline = input_df_engineered[expected_cols_for_ct]
            # Jika ColumnTransformer Anda menggunakan `remainder='drop'`, ia akan menangani kolom ekstra.
            # Yang penting adalah kolom yang akan ditransformasi oleh CT ada.
            input_features_for_pipeline = input_df_engineered 

            st.markdown("#### Hasil Prediksi")
            prediction_log = pipeline_model.predict(input_features_for_pipeline)
            predicted_count_original = np.expm1(prediction_log[0])
            predicted_count_final = max(0, int(round(predicted_count_original)))
            
            st.metric(label="Estimasi Jumlah Sewa Sepeda", value=f"{predicted_count_final} unit")

            if predicted_count_final < 50:
                st.info("Saran: Permintaan diprediksi rendah.")
            elif predicted_count_final < 250:
                st.success("Saran: Permintaan diprediksi sedang.")
            else:
                st.warning("Saran: Permintaan diprediksi tinggi.")

        except KeyError as e:
            st.error(f"Gagal membuat prediksi (KeyError): Kolom '{e}' tidak ditemukan setelah rekayasa fitur.")
            st.error("Ini biasanya berarti ada ketidaksesuaian antara fitur yang dibuat di app.py dan yang diharapkan model.")
            st.write("DataFrame setelah rekayasa fitur (sebelum prediksi):")
            st.dataframe(input_df_engineered)
        except Exception as e:
            st.error(f"Gagal membuat prediksi (Error Umum): {e}")
            st.write("DataFrame setelah rekayasa fitur (sebelum prediksi):")
            st.dataframe(input_df_engineered)
            
#====================================================================================#
# Halaman Informasi Model (DENGAN PERBAIKAN TAMPILAN PIPELINE)
#====================================================================================#
def show_model_info_page():
    st.markdown("## 📖 Informasi Detail Model Prediksi")
    st.markdown(f"""
    Model prediktif yang menjadi tulang punggung aplikasi ini adalah **XGBoost Regressor** yang dipaketkan dalam pipeline Scikit-learn.
    Pipeline ini dikembangkan dengan inspirasi dari alur kerja PyCaret, namun untuk deployment, pipeline finalnya disimpan dan digunakan secara mandiri dengan Scikit-learn untuk dependensi yang lebih ramping.

    #### Arsitektur & Pra-pemrosesan (dalam Pipeline):
    Model yang Anda gunakan (`{MODEL_FILENAME}`) adalah **keseluruhan pipeline pra-pemrosesan Scikit-learn dan model XGBoost** yang telah di-*fit* pada data historis. Proses yang ditangani oleh pipeline ini kemungkinan mencakup:
    - **Ekstraksi Fitur Waktu**: Dari kolom `datetime` (misalnya jam, hari, bulan, tahun, hari dalam seminggu, hari dalam setahun).
    - **Rekayasa Fitur Siklikal**: Transformasi sin/cos untuk fitur waktu periodik (jam, bulan, hari dalam seminggu) untuk menangkap sifat siklusnya.
    - **Penanganan Pencilan (Winsorizing)**: Untuk fitur seperti `humidity` dan `windspeed` jika diterapkan.
    - **Scaling Fitur Numerik**: Menggunakan `StandardScaler` atau metode serupa.
    - **Encoding Fitur Kategorikal**: Menggunakan `OneHotEncoder` untuk fitur seperti `season`, `weather`, `holiday`, `workingday`, dan fitur waktu kategorikal (`hour_val`, `month_val`, `weekday_val`, `year_cat`).
    - **Transformasi Target**: Variabel target (`count`) di-log-transformasi (`log1p`) sebelum pelatihan untuk menormalkan distribusinya. Prediksi dari model juga dalam skala log dan kemudian di-inverse-transform (`expm1`) kembali ke skala jumlah sewa asli di aplikasi ini.

    #### Sumber Data Acuan:
    Model ini dikembangkan berdasarkan konsep dan data dari kompetisi Kaggle:
    [Bike Sharing Demand - Kaggle](https://www.kaggle.com/competitions/bike-sharing-demand/data)

    #### Performa Model (Contoh dari Sesi Pelatihan Awal):
    *Metrik di bawah ini adalah contoh dari sesi pelatihan dan bisa bervariasi tergantung pada set validasi yang digunakan.*
    - **MAPE (Mean Absolute Percentage Error) pada Skala Asli**: Sekitar **21.82%**
    - **RMSLE (Root Mean Squared Logarithmic Error) pada Skala Asli**: Sekitar **0.2691**
    - **R² (R-squared) pada Skala Asli**: Sekitar **0.9612**
    
    *Performa pada data baru dapat bervariasi.*
    """)
    
    if pipeline_model is not None:
        st.markdown("#### Detail Pipeline dan Parameter Estimator Inti (XGBoost):")
        
        # Tampilkan langkah-langkah pipeline
        st.write("**Struktur Langkah-langkah Pipeline:**")
        if hasattr(pipeline_model, 'steps'):
            for i, (step_name, step_estimator) in enumerate(pipeline_model.steps):
                st.text(f"Langkah {i+1}: {step_name}")
                # Untuk ColumnTransformer, kita bisa coba tampilkan transformernya
                if hasattr(step_estimator, 'transformers') and step_estimator.transformers:
                    st.text("  Transformers di dalam ColumnTransformer:")
                    for t_name, t_obj, t_cols in step_estimator.transformers_:
                        st.text(f"    - {t_name}: {type(t_obj).__name__} pada kolom {t_cols[:3]}...") # Tampilkan beberapa kolom awal
                else:
                    st.text(f"  Estimator: {type(step_estimator).__name__}")
        else:
            st.text(f"Objek model tunggal: {type(pipeline_model).__name__}")

        # Tampilkan parameter model akhir (XGBoost)
        try:
            actual_model_estimator = None
            # Coba akses model akhir dari pipeline
            if hasattr(pipeline_model, 'steps'): # Jika pipeline_model adalah objek Pipeline
                # Asumsi model regresi adalah langkah terakhir
                final_step_estimator = pipeline_model.steps[-1][1]
                
                # Jika model dibungkus oleh TransformedTargetRegressor
                if hasattr(final_step_estimator, 'regressor_') and hasattr(final_step_estimator.regressor_, 'get_params'):
                     actual_model_estimator = final_step_estimator.regressor_ # Akses regressor yang sudah di-fit
                elif hasattr(final_step_estimator, 'regressor') and hasattr(final_step_estimator.regressor, 'get_params'):
                     actual_model_estimator = final_step_estimator.regressor # Untuk TransformedTargetRegressor sebelum fit
                elif hasattr(final_step_estimator, 'get_params'): # Jika langkah terakhir adalah model itu sendiri
                    actual_model_estimator = final_step_estimator
            # Jika pipeline_model BUKAN Pipeline, tapi mungkin TTR atau model itu sendiri
            elif hasattr(pipeline_model, 'regressor_') and hasattr(pipeline_model.regressor_, 'get_params'):
                 actual_model_estimator = pipeline_model.regressor_
            elif hasattr(pipeline_model, 'regressor') and hasattr(pipeline_model.regressor, 'get_params'):
                 actual_model_estimator = pipeline_model.regressor
            elif hasattr(pipeline_model, 'get_params'):
                actual_model_estimator = pipeline_model
            
            if actual_model_estimator and hasattr(actual_model_estimator, 'get_params'):
                st.markdown("**Parameter Model XGBoost (Estimator Inti):**")
                # Tampilkan parameter dengan cara yang lebih aman
                params_to_show = {k: str(v) for k, v in actual_model_estimator.get_params(deep=False).items()}
                st.json(params_to_show, expanded=False)
            else:
                st.warning("Tidak dapat mengekstrak parameter model XGBoost secara detail dari pipeline.")
        except Exception as e:
            st.warning(f"Terjadi kesalahan saat mencoba menampilkan parameter model: {e}")
    else:
        st.warning("Objek pipeline model tidak tersedia.")
    
    st.info("Untuk detail teknis lebih lanjut mengenai proses pelatihan dan validasi, silakan merujuk pada dokumentasi pengembangan internal Tim COGNIDATA.")

#====================================================================================#
# Menjalankan Aplikasi
#====================================================================================#
if __name__ == "__main__":
    if pipeline_model is None:
        # Pesan error sudah cukup jelas di atas saat load_pickled_model atau di main()
        pass
    main()
