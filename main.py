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
def min_max_normalize(series, user_value):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series), 0.5)
    normalized_series = (series - min_val) / (max_val - min_val)
    return normalized_series, (user_value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
def calculate_cosine_similarity(vector1, vector2, weights):
    weighted_vector1 = vector1 * weights
    weighted_vector2 = vector2 * weights

    dot_product = np.dot(weighted_vector1, weighted_vector2)
    magnitude1 = np.sqrt(np.dot(weighted_vector1, weighted_vector1))
    magnitude2 = np.sqrt(np.dot(weighted_vector2, weighted_vector2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

# Fungsi untuk preprocessing data berdasarkan input pengguna
def preprocess_data(df, user_input):
    df_copy = df.copy()
    user_normalized = {}

    # Simpan harga asli untuk respons
    df_copy['original_price'] = df_copy['price']

    # Normalisasi untuk setiap fitur numerik kecuali harga
    for col in ['ram', 'storage', 'screen_size']:
        if col in user_input:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            df_copy = df_copy.dropna(subset=[col])
            
            if df_copy[col].empty:
                continue
                
            # Normalisasi data produk dan input pengguna
            normalized_col, user_val = min_max_normalize(df_copy[col], user_input[col])
            df_copy[col] = normalized_col
            user_normalized[col] = user_val

    # Normalisasi harga hanya untuk perhitungan similarity
    if 'price' in user_input:
        df_copy['price'] = pd.to_numeric(df_copy['price'], errors='coerce')
        df_copy = df_copy.dropna(subset=['price'])
        
        if not df_copy['price'].empty:
            normalized_price, user_price_norm = min_max_normalize(df_copy['price'], user_input['price'])
            df_copy['price'] = normalized_price
            user_normalized['price'] = user_price_norm

    # Penanganan tipe penyimpanan
    if user_input['type_storage'].lower() in ['ssd', 'hdd']:
        # Filter data berdasarkan type_storage yang dipilih
        df_copy = df_copy[df_copy['type_storage'].str.lower() == user_input['type_storage'].lower()]
        user_normalized['type_storage'] = 1  # Bobot penuh jika sesuai
    else:
        user_normalized['type_storage'] = 0.5  # Bobot netral jika 'all'

    # Penanganan prosesor
    processor_mapping = {'intel': 1, 'amd': 0}
    if user_input['processor'].lower() in processor_mapping:
        df_copy['processor'] = df_copy['processor'].str.lower().str.contains(
            user_input['processor'].lower()
        ).astype(int)
        user_normalized['processor'] = 1
    else:
        user_normalized['processor'] = 0.5
        df_copy['processor'] = 0.5

    return df_copy, user_normalized

def get_dynamic_weights(user_input):
    weights = {
        'price': 0.17,
        'ram': 0.21,
        'storage': 0.20,
        'screen_size': 0.11,
        'processor': 0.16 if user_input['processor'] != 'all' else 0,
        'type_storage': 0.15 if user_input['type_storage'] != 'all' else 0
    }
    return np.array(list(weights.values()))
@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        user_input = {
            'price': float(request.args.get('price', -1)),
            'ram': float(request.args.get('ram', -1)),
            'storage': float(request.args.get('storage', -1)),
            'screen_size': float(request.args.get('screen_size', -1)),
            'type_storage': request.args.get('type_storage', 'all'),
            'processor': request.args.get('processor', 'all')
        }

        df = fetch_data_from_mysql()
        if df.empty:
            return jsonify({'average_similarity': 0, 'data': []})

        processed_df, user_norm = preprocess_data(df, user_input)
        weights = get_dynamic_weights(user_input)
        
        # Membuat vektor input pengguna
        user_vector = np.array([
            user_norm['price'],
            user_norm['ram'],
            user_norm['storage'],
            user_norm['screen_size'],
            user_norm['processor'],
            user_norm['type_storage']
        ])

        similarities = []
        for _, row in processed_df.iterrows():
            product_vector = np.array([
                row['price'],
                row['ram'],
                row['storage'],
                row['screen_size'],
                row['processor'],
                1 if row['type_storage'].lower() == user_input['type_storage'].lower() else 0
            ])
            similarity = calculate_cosine_similarity(user_vector, product_vector, weights)
            similarities.append(similarity)

        processed_df['similarity'] = similarities
        top_laptops = processed_df.nlargest(int(request.args.get('count', 5)), 'similarity')
        
        # Konversi ke respons JSON
        results = []
        for _, row in top_laptops.iterrows():
            laptop_data = row.to_dict()
            # Gunakan harga asli untuk respons
            laptop_data['price'] = row['original_price']
            # Hapus kolom original_price dari respons
            del laptop_data['original_price']
            results.append(laptop_data)
        
        return jsonify({
            'average_similarity': top_laptops['similarity'].mean(),
            'data': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    
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
