from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)
sec = app

def fetch_data_from_mysql():
    try:
        connection = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="root",
            database="laravel_skripsi"
        )
        
        query = "SELECT * FROM product"
        df = pd.read_sql(query, connection)
        print(df)
    except Error as e:
        print("Error while connecting to MySQL", e)
        return pd.DataFrame()  
    finally:
        if connection.is_connected():
            connection.close()
    
    return df

def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())


def preprocess_data(df, user_input):
    df_copy = df.copy()

    for col in ['price', 'ram', 'storage', 'screen_size']:
        if col in user_input:
            user_input[col] = float(user_input[col])  
            
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            
            if df_copy[col].isnull().any():
                return jsonify({"error": f"Invalid data found in column: {col}"}), 400

            
            df_copy[col] = min_max_normalize(df_copy[col])

            
            user_input[col] = (user_input[col] - df[col].min()) / (df[col].max() - df[col].min())

    
    df_copy['type_storage'] = df_copy['type_storage'].apply(lambda x: 1 if x == user_input['type_storage'] else 0)
    df_copy['processor'] = df_copy['processor'].apply(lambda x: 1 if x == user_input['processor'] else 0)

    
    user_input['type_storage'] = 1  
    user_input['processor'] = 1     

    return df_copy, user_input


def calculate_cosine_similarity(vector1, vector2, weights):
    weighted_vector1 = vector1 * weights
    weighted_vector2 = vector2 * weights

    dot_product = np.dot(weighted_vector1, weighted_vector2)
    magnitude1 = np.sqrt(np.dot(weighted_vector1, weighted_vector1))
    magnitude2 = np.sqrt(np.dot(weighted_vector2, weighted_vector2))

    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0

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

        print("User Input:", user_input)
        
        df = fetch_data_from_mysql()
        if user_input['price'] == -1 and user_input['ram'] == -1 and user_input['storage'] == -1 and user_input['screen_size'] == -1 and user_input['type_storage'] == 'all' and user_input['processor'] == 'all':
            results = df.to_dict(orient='records')
            for product in results:
                product['similarity'] = 0  
            return jsonify({
                'average_similarity': 0,  
                'data': results
            })

        
        processed_df, user_vector = preprocess_data(df, user_input)

        weights = np.array([0.3, 0.2, 0.1, 0.1, 0.15, 0.15])

        user_vector = np.array([
            user_vector['price'],
            user_vector['ram'],
            user_vector['storage'],
            user_vector['screen_size'],
            user_vector['type_storage'],
            user_vector['processor']
        ])

        similarities = []
        for _, row in processed_df.iterrows():
            product_vector = np.array([
                row['price'],
                row['ram'],
                row['storage'],
                row['screen_size'],
                row['type_storage'],
                row['processor']
            ])
            similarity = calculate_cosine_similarity(user_vector, product_vector, weights)
            similarities.append(similarity)

        processed_df['similarity'] = similarities
        top_laptops = processed_df.nlargest(5, 'similarity')

        results = df.loc[top_laptops.index].to_dict(orient='records')

        for i, product in enumerate(results):
            product['similarity'] = similarities[i]

        average_similarity = np.mean(similarities)

        return jsonify({
            'average_similarity': average_similarity,
            'data': results
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400



if __name__ == '__main__':
    app.run(debug=True)
