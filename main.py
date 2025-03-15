"""
    API LaptopLand adalah aplikasi berbasis
    Flask yang menyediakan layanan rekomendasi laptop
    berdasarkan input spesifikasi dan preferensi pengguna.
    API ini menggunakan database MySQL untuk
    mengambil data laptop dan melakukan
    perhitungan cosine similarity dengan 
    bobot tertentu untuk menghasilkan rekomendasi.
"""

# Inisialisasi dependencies atau packages
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error
import sys

# Inisialisasi aplikasi Flask
app = Flask(__name__)
application = app

def fetch_data_from_mysql():
    """
    Mengambil data dari tabel 'product' di MySQL.
    Mengembalikan DataFrame pandas yang berisi data dari tabel.
    """
    try:
        # Konfigurasi koneksi ke database MySQL
        # connection = mysql.connector.connect(
        #     host="127.0.0.1",
        #     user="refactor_laptopland",
        #     password="LaptopLand123",
        #     database="refactor_laps"
        # )
        
        # ------ develop local ----------
        connection = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="",
            database="db_laptopland"
        )
        
        # Query untuk mengambil semua data dari tabel 'product'
        query = "SELECT * FROM product"
        df = pd.read_sql(query, connection)
        print(df)
    except Error as e:
        # Mengembalikan DataFrame kosong jika terjadi kesalahan
        print("Error while connecting to MySQL", e)
        return pd.DataFrame()  
    finally:
        # Menutup koneksi jika terhubung
        if connection.is_connected():
            connection.close()
    
    return df

# Fungsi untuk normalisasi data menggunakan min-max normalization
def min_max_normalize(series):
    """
    Melakukan normalisasi min-max pada kolom.
    Args:
        series (pd.Series): Kolom data yang akan dinormalisasi.
    Returns:
        pd.Series: Kolom yang telah dinormalisasi.
    """
    return (series - series.min()) / (series.max() - series.min())

def calculate_cosine_similarity(vector1, vector2, weights):
    """
    Menghitung cosine similarity antara dua vektor dengan bobot.
    Args:
        vector1 (np.array): Vektor pertama.
        vector2 (np.array): Vektor kedua.
        weights (np.array): Bobot untuk masing-masing elemen vektor.
    Returns:
        float: Nilai cosine similarity.
    """
    
    # Mengalikan vektor dengan bobot
    weighted_vector1 = vector1 * weights
    weighted_vector2 = vector2 * weights

    # Menghitung dot product dan magnitudo
    dot_product = np.dot(weighted_vector1, weighted_vector2)
    magnitude1 = np.sqrt(np.dot(weighted_vector1, weighted_vector1))
    magnitude2 = np.sqrt(np.dot(weighted_vector2, weighted_vector2))

    # Mengembalikan hasil cosine similarity
    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0

# Fungsi untuk preprocessing data berdasarkan input pengguna
def preprocess_data(df, user_input):
    """
    Melakukan preprocessing data untuk perhitungan rekomendasi.
    Args:
        df (pd.DataFrame): DataFrame dengan data produk.
        user_input (dict): Input spesifikasi pengguna.
    Returns:
        tuple: DataFrame yang diproses dan vektor input pengguna.
    """
    
    # Salin DataFrame untuk menjaga data asli tetap utuh
    df_copy = df.copy()

    for col in ['price', 'ram', 'storage', 'screen_size']:
        if col in user_input:
            # Pastikan input berupa float
            user_input[col] = float(user_input[col])  

            # Ubah kolom menjadi numerik
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            
            if df_copy[col].isnull().any():
                # Jika ada data yang tidak valid
                return jsonify({"error": f"Invalid data found in column: {col}"}), 400

            # Normalisasi kolom
            df_copy[col] = min_max_normalize(df_copy[col])

            # Normalisasi input pengguna
            user_input[col] = (user_input[col] - df[col].min()) / (df[col].max() - df[col].min())

    # Proses tipe penyimpanan
    if user_input['type_storage'] != 'all':
        df_copy['type_storage'] = df_copy['type_storage'].apply(lambda x: 1 if x == user_input['type_storage'] else 0)
    else:
        df_copy['type_storage'] = 1
        
    df_copy['type_storage'] = df_copy['type_storage'].astype(str)
    df_copy['processor'] = df_copy['processor'].astype(str)


    # Proses prosesor
    if user_input['processor'].lower() == 'intel':
        df_copy['processor'] = df_copy['processor'].apply(lambda x: 1 if x.lower() == 'intel' else 0)
        user_input['processor'] = 1  # Vektor input pengguna untuk Intel
    elif user_input['processor'].lower() == 'amd':
        df_copy['processor'] = df_copy['processor'].apply(lambda x: 1 if x.lower() == 'amd' else 0)
        user_input['processor'] = 0  # Vektor input pengguna untuk AMD
    else:
        df_copy['processor'] = 1
        user_input['processor'] = 1

    # Tetapkan nilai default untuk input pengguna
    if user_input['type_storage'].lower() == 'ssd':
        df_copy['type_storage'] = df_copy['type_storage'].apply(lambda x: 1 if x.lower() == 'ssd' else 0)
        user_input['type_storage'] = 1  # Vektor input pengguna untuk SSD
    elif user_input['type_storage'].lower() == 'hdd':
        df_copy['type_storage'] = df_copy['type_storage'].apply(lambda x: 1 if x.lower() == 'hdd' else 0)
        user_input['type_storage'] = 0  # Vektor input pengguna untuk HDD
    else:
        # Jika 'all', tidak ada preferensi
        df_copy['type_storage'] = 1
        user_input['type_storage'] = 1


    return df_copy, user_input

# Endpoint untuk rekomendasi laptop
@app.route('/recommend', methods=['GET'])
def recommend():
    """
    Endpoint untuk memberikan rekomendasi laptop berdasarkan spesifikasi pengguna.
    Parameter Query:
        - price (float): Harga laptop.
        - ram (float): Kapasitas RAM.
        - storage (float): Kapasitas penyimpanan.
        - screen_size (float): Ukuran layar.
        - type_storage (str): Tipe penyimpanan (SSD/HDD/all).
        - processor (str): Tipe prosesor (Intel/AMD/all).
        - count (int): Jumlah rekomendasi yang ingin ditampilkan (default: 5).
    Returns:
        JSON: Rekomendasi laptop dengan nilai similarity.
    """
    try:
        # Mengambil parameter dari query
        user_input = {
            'price': float(request.args.get('price', -1)),  
            'ram': float(request.args.get('ram', -1)),
            'storage': float(request.args.get('storage', -1)),
            'screen_size': float(request.args.get('screen_size', -1)),
            'type_storage': request.args.get('type_storage', 'all'),  
            'processor': request.args.get('processor', 'all')
        }

        # Mengambil jumlah data yang ingin ditampilkan
        count = int(request.args.get('count', 5))  # Default 5 jika parameter count tidak diberikan
        print("User Input:", user_input)
        print("Count:", count)
        
        # Mengambil data dari MySQL
        df = fetch_data_from_mysql()

        # Filter data berdasarkan type_storage dan processor
        if user_input['type_storage'].lower() != 'all':
            df = df[df['type_storage'].str.lower() == user_input['type_storage'].lower()]

        if user_input['processor'].lower() != 'all':
            df = df[df['processor'].str.lower().str.startswith(user_input['processor'].lower())]

        # Jika tidak ada data yang sesuai dengan filter
        if df.empty:
            return jsonify({
                'average_similarity': 0,
                'data': []
            })

        # Preprocessing data (hanya untuk kolom numerik)
        processed_df, user_vector = preprocess_data(df, user_input)

        # Bobot untuk setiap atribut (hanya untuk kolom numerik)
        weights = np.array([0.17, 0.21, 0.2, 0.11])  # Hanya untuk price, ram, storage, screen_size

        # Buat vektor input pengguna (hanya untuk kolom numerik)
        user_vector = np.array([
            user_vector['price'],
            user_vector['ram'],
            user_vector['storage'],
            user_vector['screen_size']
        ])

        # Menghitung cosine similarity
        similarities = []
        for _, row in processed_df.iterrows():
            product_vector = np.array([
                row['price'],
                row['ram'],
                row['storage'],
                row['screen_size']
            ])
            similarity = calculate_cosine_similarity(user_vector, product_vector, weights)
            similarities.append(similarity)

        # Tambahkan kolom similarity ke DataFrame
        processed_df['similarity'] = similarities
        top_laptops = processed_df.nlargest(count, 'similarity')  # Mengambil sejumlah count teratas

        # Konversi hasil ke JSON
        results = df.loc[top_laptops.index].to_dict(orient='records')

        for i, product in enumerate(results):
            product['similarity'] = similarities[i]
            
        results = sorted(results, key=lambda x: x.get('similarity', 0), reverse=True)

        average_similarity = np.mean(similarities)

        return jsonify({
            'average_similarity': average_similarity,
            'data': results
        })
    
    except Exception as e:
        # Tangani error secara umum
        return jsonify({"error": str(e)}), 400

@app.route('/', methods=['GET'])
def index():
    """
    Endpoint default untuk mengecek apakah API berjalan.
    Returns:
        JSON: Pesan selamat datang dengan versi Python.
    """
    response = {
        'message': 'API Laptopland!',
        'version': f'Python {sys.version}'
    }
    return jsonify(response)



# Menjalankan aplikasi jika file ini di-eksekusi
if __name__ == '__main__':
    app.run(debug=True)
