# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import subprocess
import sys
import os

# Fungsi untuk menginstal dependensi dari requirements.txt
def install_requirements():
    if not os.path.isfile("requirements.txt"):
        print("File requirements.txt tidak ditemukan!")
        return

    # Menginstal dependensi
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Panggil fungsi untuk menginstal
install_requirements()


import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Student Performance Analyst",
    layout="centered"
)

# Judul Aplikasi
st.title("Student Performance Analyst")
st.markdown("""
    <p style='font-size:18px; color:gray; text-align: justify;'>
        Muhammad Irvan Arfirza. 
        Github: <a href='https://github.com/firzzairvn' target='_blank' style='color:gray;'>https://github.com/firzzairvn</a>
    </p>
""", unsafe_allow_html=True)


# Memuat Dataset
uploaded_file = st.file_uploader("Pilih file CSV", type="csv")


if uploaded_file is not None:
    # Baca file CSV menggunakan Pandas
    df = pd.read_csv(uploaded_file)

    # Tampilkan Dataframe sebagai tabel di Streamlit
    st.write("Dataframe:")
    st.dataframe(df)


    # Bagian Data Understanding
    st.header("Data Understanding")
    st.markdown("""
    <p style='font-size:18px; color:gray; text-align: justify;'>Pada bagian ini, kita akan melihat deskripsi dari data yang kita miliki, termasuk data yang hilang, distribusi data kategorikal, 
                dan statistik deskriptif. Memahami data adalah langkah awal yang sangat penting sebelum melangkah ke tahap analisis dan modeling.
                </p>
    """, unsafe_allow_html=True)


    st.subheader("1. Jumlah Nilai yang Hilang Sebelum Preparation")
    st.code('df.isnull().sum().sort_values(ascending=False)', language='python')
    df_bfrPrep = df.isnull().sum().sort_values(ascending=False)
    st.write(df_bfrPrep)
    with st.expander("Penjelasan Lengkap Kode"):
        st.write("""
Fungsi `df.dropna().isnull().sum().sort_values(ascending=False)` digunakan untuk menghitung 
jumlah nilai yang hilang (NaN) dalam DataFrame setelah menghapus baris-baris yang memiliki 
nilai kosong. Pertama, `dropna()` akan menghapus semua baris yang mengandung nilai NaN, 
kemudian `isnull().sum()` menghitung jumlah nilai yang hilang di setiap kolom, dan 
`sort_values(ascending=False)` mengurutkan hasilnya dari yang terbanyak ke yang tersedikit. 
Ini membantu untuk memahami data mana yang memiliki masalah dengan nilai yang hilang setelah 
proses pembersihan.
""")


    st.subheader("2. Jumlah Nilai yang Hilang Setelah Preparation")
    st.code('df.dropna().isnull().sum().sort_values(ascending=False)', language='python')
    df_aftPrep = df.dropna().isnull().sum().sort_values(ascending=False)
    st.write(df_aftPrep)
    with st.expander("Penjelasan Lengkap Kode"):
        st.write("Data tersebut merupakan data yang sudah dilakukan pembersihan pada bagian Data Preparation dengan melakukan drop terhadap data yang null")

    st.subheader("3. Jumlah Duplikasi Data")
    st.code('df.duplicated().sum()', language='python')
    st.write("Jumlah duplikasi data:", df.duplicated().sum())
    with st.expander("Penjelasan Lengkap Kode"):
        st.write("""
Fungsi `df.duplicated().sum()` digunakan untuk menghitung jumlah baris duplikat dalam DataFrame. 
Setiap baris dianggap duplikat jika semua nilai di baris tersebut identik dengan baris lainnya. 
Mengetahui jumlah duplikasi sangat penting untuk memastikan integritas data dan menghindari 
analisis yang tidak akurat akibat adanya data yang sama dalam dataset.
""")


    st.subheader("4. Statistik Deskriptif")
    st.code('df.describe()', language='python')
    st.write(df.describe())
    with st.expander("Penjelasan Lengkap Kode"):
        st.write("""
Fungsi `df.describe()` memberikan ringkasan statistik deskriptif untuk kolom numerik dalam DataFrame. 
Ini mencakup metrik seperti jumlah (count), rata-rata (mean), deviasi standar (std), nilai minimum (min), 
nilai maksimum (max), dan persentil (25%, 50%, 75%). Ringkasan ini berguna untuk memahami distribusi data 
dan karakteristik statistik dari fitur numerik dalam dataset.
""")


    st.subheader("5. Visualisasi Data Kategorikal")
    st.markdown("Visualisasi distribusi data kategorikal yang ada di dalam dataset.")
    for col in df:
        if df[col].dtype == 'O':
            fig, ax = plt.subplots(figsize=(5,3))
            sns.countplot(x=col, data=df, palette='viridis', ax=ax)
            ax.set_title(f'Distribusi Kategori {col}')
            st.pyplot(fig)
    with st.expander("Penjelasan Lengkap Kode"):
        st.write("""
Kode ini membuat visualisasi count plot untuk setiap kolom kategorikal dalam dataset, 
di mana setiap kolom dengan tipe data objek (string) akan dianalisis. 
Dengan ukuran grafik yang diatur menjadi 5x3 inci, fungsi `sns.countplot` 
menghitung dan menampilkan frekuensi setiap kategori dalam kolom, menggunakan palet warna 'viridis'. 
Judul grafik menunjukkan kategori yang sedang divisualisasikan dengan format 'Distribusi Kategori {nama_kolom}'.
""")

    st.subheader("6. Scatterplot Nilai Siswa")
    st.code("""
sns.scatterplot(data=df['Exam_Score'], color='brown')
plt.ylabel('Nilai Siswa')
plt.xlabel('Index Siswa')
plt.title('Persebaran nilai siswa')
fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.show()
""", language='python')
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.scatterplot(data=df, x=df.index, y='Exam_Score', color='brown', ax=ax)
    ax.set_ylabel('Nilai Siswa')
    ax.set_xlabel('Index Siswa')
    ax.set_title('Persebaran nilai siswa')
    st.pyplot(fig)
    with st.expander("Penjelasan Lengkap Kode"):
        st.write("""
Kode ini menghasilkan visualisasi scatter plot untuk menunjukkan persebaran nilai siswa 
dari kolom `Exam_Score` dalam dataset. Grafik ini memplot nilai siswa pada sumbu y 
dan indeks siswa pada sumbu x. Ukuran grafik diatur menjadi 5x3 inci, dengan 
warna titik yang digunakan adalah cokelat. Sumbu y diberi label 'Nilai Siswa' 
dan sumbu x diberi label 'Index Siswa', sementara judul grafik menjelaskan isi grafik 
dengan mencantumkan 'Persebaran nilai siswa'.
""")


    st.subheader("7. Histogram Nilai Siswa")
    st.code("""
plt.figure(figsize=(8, 5))
sns.distplot(df['Exam_Score'], color='r', bins=100, hist_kws={'alpha': 0.4})
plt.title('Histogram Nilai Siswa')
plt.xticks(rotation=90)
plt.xlabel('Nilai Siswa')
plt.ylabel('Nilai Siswa')
plt.tight_layout()
plt.show()
""", language='python')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Exam_Score'], color='r', bins=100, kde=True, ax=ax)
    ax.set_title('Histogram Nilai Siswa')
    ax.set_xlabel('Nilai Siswa')
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)
    with st.expander("Penjelasan Lengkap Kode"):
        st.write("""
Kode ini digunakan untuk membuat histogram dari nilai ujian siswa yang terdapat dalam kolom `Exam_Score` pada DataFrame `df`. 
Dengan ukuran grafik 8x5 inci, histogram ditampilkan dengan 100 bin dan batang berwarna merah yang memiliki transparansi 40%, 
memungkinkan visualisasi distribusi data yang jelas. Judul grafik diatur menjadi "Histogram Nilai Siswa", 
dengan label sumbu X dan Y juga diberi nama "Nilai Siswa". Rotasi 90 derajat pada label sumbu X memastikan keterbacaan, 
dan `plt.tight_layout()` digunakan untuk mengatur tata letak grafik agar tidak ada elemen yang terpotong. 
Akhirnya, `plt.show()` menampilkan grafik tersebut.
""")

    st.subheader("8. Histogram dari Semua Kolom")
    st.code("df.hist(figsize=(15,15))", language='python')
    fig, ax = plt.subplots(figsize=(15, 15))
    df.hist(ax=ax)
    st.pyplot(fig)
    with st.expander("Penjelasan Lengkap Kode"):
        st.write("""
Fungsi `df.hist(ax=ax)` digunakan untuk membuat histogram dari setiap kolom numerik dalam 
DataFrame `df`. Histogram ini ditampilkan dalam satu figur yang besar dengan ukuran 15x15 
inci, memberikan visualisasi distribusi nilai untuk setiap kolom secara bersamaan. 
Ini sangat berguna untuk memahami sebaran data dan identifikasi pola atau anomali yang 
mungkin ada dalam dataset.
""")


    # Bagian Data Preparation
    st.header("Data Preparation")
    st.markdown("""
    <p style='font-size:18px; color:gray; text-align: justify; >Data Preparation merupakan tahap penting dalam analisis data yang bertujuan untuk 
menyiapkan dataset agar siap digunakan dalam model analisis atau pembelajaran mesin. 
Tahap ini meliputi beberapa langkah, seperti menghapus nilai yang hilang, mengisi 
nilai yang hilang pada kolom kategorikal, mengubah data kategorikal menjadi format 
numerik melalui encoding, dan menggabungkan data kembali ke dalam satu DataFrame. 
Dengan melakukan langkah-langkah ini, kita memastikan bahwa dataset bersih, terstruktur, 
dan siap untuk analisis lebih lanjut atau pemodelan.</p>
    """, unsafe_allow_html=True)

    st.subheader("1. Menghapus Nilai yang Hilang")
    st.code("df = df.dropna()", language='python')
    df_prep = df.dropna()
    st.write("Jumlah data setelah menghapus nilai yang hilang:", df_prep.shape)
    with st.expander("Penjelasan Lengkap Kode"):
        st.write("""Dengan menggunakan kode `df = df.dropna()`, kita menghapus semua baris yang 
memiliki nilai yang hilang (NaN) dalam DataFrame `df`. Langkah ini penting 
dalam tahap persiapan data, karena nilai yang hilang dapat memengaruhi 
proses analisis dan pemodelan. Dengan menghilangkan baris-baris tersebut, 
kita memastikan bahwa data yang digunakan dalam analisis dan pemodelan 
adalah lengkap dan akurat, sehingga menghasilkan hasil yang lebih valid. """)

    st.subheader("2. Menentukan Kolom Numerik dan Kategorikal")
    st.code("""
num_col = [col for col in df.columns if df[col].dtype != 'object']
cat_col = [col for col in df.columns if df[col].dtype == 'object']
cat_col
""", language='python')
    num_col = [col for col in df_prep.columns if df_prep[col].dtype != 'object']
    cat_col = [col for col in df_prep.columns if df_prep[col].dtype == 'object']
    st.write("Kolom numerik:", num_col)
    st.write("Kolom kategorikal:", cat_col)
    with st.expander("Penjelasan Lengkap Kode"):
        st.write("""
kita mendefinisikan dua daftar, yaitu num_col dan cat_col. Daftar num_col berisi nama-nama kolom yang memiliki tipe data numerik (bukan objek), sementara cat_col berisi nama-nama kolom yang memiliki tipe data kategorikal (objek). 
Pemisahan ini penting untuk memudahkan analisis dan pemrosesan data, 
karena kolom numerik dan kategorikal memerlukan teknik yang berbeda dalam analisis dan pemodelan.""")

    st.subheader("3. Mengisi Nilai yang Hilang di Kolom Kategorikal")
    st.code("""
for col in cat_col:
    df[col].fillna(df[col].mode()[0], inplace=True)
""", language='python')
    for col in cat_col:
        df_prep[col].fillna(df_prep[col].mode()[0], inplace=True)
    st.write("Jumlah data setelah mengisi nilai yang hilang:", df_prep.shape)
    with st.expander("Penjelasan Lengkap Kode"):
        st.write("""kode ini mengisi nilai yang hilang pada kolom-kolom dalam daftar cat_col dengan modus (nilai yang paling sering muncul) dari masing-masing kolom. Pengisian nilai yang hilang dengan modus ini berguna untuk mempertahankan integritas
data dan memastikan bahwa analisis selanjutnya tidak terganggu oleh keberadaan nilai yang hilang dalam kolom kategorikal. """)

    st.subheader("4. Mengubah Data Kategorikal Menjadi Dummy")
    st.code("""
data_dummy = pd.get_dummies(df[cat_col], drop_first=True)
data_dummy.shape
""", language='python')
    data_dummy = pd.get_dummies(df_prep[cat_col], drop_first=True)
    st.write("Shape data dummy:", data_dummy.shape)
    with st.expander("Penjelasan Lengkap Kode"):
        st.write("""kode ini mengubah kolom-kolom kategorikal dalam DataFrame df yang terdapat dalam daftar cat_col menjadi variabel dummy menggunakan fungsi pd.get_dummies(). Argumen drop_first=True digunakan untuk menghindari multikolinearitas dengan menghapus salah satu kolom dummy yang dihasilkan.
Output dari data_dummy.shape menunjukkan dimensi dari DataFrame dummy yang dihasilkan, memberikan informasi tentang jumlah baris dan kolom baru yang ditambahkan. """)


    st.subheader("5. Menggabungkan Data")
    st.code("""
df.drop(columns=cat_col, inplace=True)
df = pd.concat([df, data_dummy], axis=1)
df.shape
""", language='python')
    df_prep.drop(columns=cat_col, inplace=True)
    df_prep = pd.concat([df_prep, data_dummy], axis=1)
    st.write("Shape data setelah digabung:", df_prep.shape)
    with st.expander("Penjelasan Lengkap Kode"):
        st.write("""kode ini pertama-tama menghapus kolom-kolom kategorikal dari DataFrame df yang terdapat dalam daftar cat_col dengan menggunakan fungsi drop(). Kemudian, variabel dummy yang dihasilkan sebelumnya dalam data_dummy digabungkan kembali ke dalam DataFrame df menggunakan pd.concat(). Penggabungan ini dilakukan secara horizontal (axis=1) untuk memastikan bahwa struktur DataFrame tetap utuh. 
Output dari df.shape menunjukkan dimensi akhir dari DataFrame setelah penghapusan dan penggabungan kolom, memberikan informasi tentang jumlah baris dan kolom yang baru.  """)
    
    # Tampilkan DataFrame yang telah diproses
    st.subheader("DataFrame Setelah Preparation")
    st.write(df_prep)

    # ---- Modeling ----
    st.header("Modeling Using Linear Regression and Support Vector Regression")
    st.markdown("""
    <p style='font-size:18px; color:gray;text-align: justify; '>Modeling menggunakan Linear Regression dan Support Vector Regression (SVR) adalah langkah 
penting dalam analisis prediktif untuk memahami hubungan antara variabel input dan output. 
Linear Regression berfungsi untuk memodelkan hubungan linier antara fitur dan target, 
sedangkan SVR lebih fleksibel dan dapat menangani data yang tidak terdistribusi secara linier. 
Kedua model ini digunakan untuk memprediksi nilai dari variabel target 'Exam_Score' berdasarkan 
fitur-fitur lainnya dalam dataset. Melalui evaluasi kinerja model seperti Mean Squared Error 
(MSE) dan akurasi, kita dapat menentukan model mana yang lebih efektif dalam melakukan prediksi.</p>
    """, unsafe_allow_html=True)

    # Memisahkan fitur dan target
    st.subheader("1. Pemisahan Fitur dan Target")
    X = df_prep.drop('Exam_Score', axis=1)  # Menggunakan df_prep yang sudah di-preparation
    y = df_prep['Exam_Score']
    st.code("X = df_prep.drop('Exam_Score', axis=1)\ny = df_prep['Exam_Score']", language='python')
    st.write("Shape X:", X.shape)
    st.write("Shape y:", y.shape)
    with st.expander("Penjelasan Lengkap Kode"):
        st.write("""Pada kode ini, kita memisahkan DataFrame yang telah dipersiapkan (`df_prep`) menjadi dua variabel: `X` dan `y`. 
    Variabel `X` berisi semua kolom fitur yang diperlukan untuk model prediksi, dengan menghapus kolom `Exam_Score` 
    yang merupakan nilai target. Sementara itu, variabel `y` menyimpan kolom `Exam_Score` yang akan diprediksi. 
    Proses pemisahan ini penting untuk mempersiapkan data sebelum melatih model machine learning. """)


    # Membagi dataset menjadi training dan testing set
    st.subheader("2. Pembagian Dataset (Train: 80%, Test: 20%)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.code("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)", language='python')
    st.write("Shape X_train:", X_train.shape)
    st.write("Shape X_test:", X_test.shape)
    st.write("Shape y_train:", y_train.shape)
    st.write("Shape y_test:", y_test.shape)
    with st.expander("Penjelasan Lengkap Kode"):
        st.write("""Pada kode ini, kita membagi dataset yang telah dipisahkan menjadi dua bagian: data pelatihan dan data pengujian 
    menggunakan fungsi `train_test_split` dari scikit-learn. Variabel `X_train` dan `y_train` akan digunakan untuk 
    melatih model, sedangkan `X_test` dan `y_test` akan digunakan untuk menguji performa model setelah dilatih. 
    Dengan menggunakan parameter `test_size=0.2`, kita menetapkan bahwa 20% dari total data akan digunakan sebagai 
    data pengujian, sementara 80% sisanya digunakan untuk pelatihan. Parameter `random_state=42` menjamin bahwa 
    pembagian data akan konsisten setiap kali kode dijalankan. """)


    # Standarisasi fitur
    st.subheader("3. Standarisasi Fitur")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    st.code("scaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)", language='python')
    with st.expander("Penjelasan Lengkap Kode"):
        st.write("""Pada bagian kode ini, kita melakukan standarisasi terhadap fitur-fitur dalam dataset menggunakan `StandardScaler` 
    dari scikit-learn. Pertama, objek `scaler` dibuat untuk menginisialisasi standarizer. Kemudian, metode `fit_transform` 
    diterapkan pada `X_train` untuk menghitung rata-rata dan deviasi standar dari fitur, sekaligus melakukan transformasi 
    pada data pelatihan agar memiliki distribusi dengan rata-rata 0 dan deviasi standar 1. Selanjutnya, metode 
    `transform` diterapkan pada `X_test` untuk mengubah data pengujian dengan parameter yang sama yang digunakan untuk 
    data pelatihan, sehingga model dapat menguji data dalam skala yang konsisten. Proses ini penting untuk memastikan 
    bahwa model tidak bias terhadap skala fitur yang berbeda. """)


    # ---- Uji Coba di beberapa model ----
    st.header("Uji Coba Model")
    # Linear Regression
    st.subheader("1. Linear Regression")
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    st.write("**Mean Squared Error:**", mse)
    st.write("**Train Score:**", lin_reg.score(X_train, y_train))
    st.write("**Test Score:**", lin_reg.score(X_test, y_test))

    # Visualisasi Regresi Linear
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(x=y_test, y=y_pred, ax=ax)
    ax.set_title('Linear Regression: Prediksi vs Nilai Sebenarnya')
    ax.set_xlabel('Nilai Sebenarnya')
    ax.set_ylabel('Nilai Prediksi')
    st.pyplot(fig)

    # Support Vector Regression (SVR)
    st.subheader("2. Support Vector Regression (SVR)")
    svr = SVR()
    svr.fit(X_train, y_train)
    svr_y_pred = svr.predict(X_test)

    st.write("**Mean Squared Error:**", mean_squared_error(y_test, svr_y_pred))
    st.write("**Train Score:**", svr.score(X_train, y_train))
    st.write("**Test Score:**", svr.score(X_test, y_test))

    # ---- Evaluasi Model ----
    st.header("Evaluate Model")
    st.markdown("""
    <p style='font-size:18px; color:gray;text-align: justify; '>Evaluasi Model adalah tahap kritis untuk menilai kinerja model yang telah dibangun. 
Dalam proses ini, kita membandingkan akurasi dari berbagai model yang telah diterapkan, 
seperti Linear Regression dan Support Vector Regression (SVR), dengan menggunakan metrik 
seperti Mean Squared Error (MSE) dan skor akurasi pada dataset uji. Dengan mengevaluasi 
model, kita dapat menentukan mana yang paling efektif dalam memprediksi nilai 'Exam_Score' 
dan mengambil keputusan lebih lanjut mengenai pemodelan yang perlu dilakukan, 
apakah perlu melakukan tuning atau memilih model lain yang lebih sesuai.</p>
    """, unsafe_allow_html=True)
    # Perbandingan Model
    acc = []
    names = ['LinearRegression', 'SVR']
    acc.extend([lin_reg.score(X_test, y_test), svr.score(X_test, y_test)])

    fig, ax = plt.subplots(figsize=(5, 3))  # Atur ukuran grafik
    ax.bar(names, acc, color='c')
    ax.set_title('Perbandingan Model', fontsize=16)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Akurasi', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for i, value in enumerate(acc):
        ax.text(i, value + 0.005, f'{value * 100:.2f}%', ha='center', fontsize=12, color='black')

    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    st.pyplot(fig)

    
else:
    st.info('Upload file CSV untuk mulai.')



