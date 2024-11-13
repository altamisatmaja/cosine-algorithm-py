from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import mysql.connector

app = Flask(__name__)

def fetch_data_from_mysql():
    connection = mysql.connector.connect(
        host="your_host",      
        user="your_user",      
        password="your_password",  
        database="your_database"  
    )

    query = "SELECT * FROM laptops"
    df = pd.read_sql(query, connection)
    connection.close()
    return df


def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())


def preprocess_data(df, user_input):
    
    df_copy = df.copy()

    
    for col in ['price', 'ram', 'storage', 'screen_size']:
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

    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    else:
        return dot_product / (magnitude1 * magnitude2)


@app.route('/recommend', methods=['POST'])
def recommend():
    
    user_input = request.json

    
    df = fetch_data_from_mysql()

    
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
        laptop_vector = np.array([
            row['price'],
            row['ram'],
            row['storage'],
            row['screen_size'],
            row['type_storage'],
            row['processor']
        ])
        similarity = calculate_cosine_similarity(user_vector, laptop_vector, weights)
        similarities.append(similarity)

    
    processed_df['similarity'] = similarities

    
    top_laptops = processed_df.nlargest(5, 'similarity')

    
    results = df.loc[top_laptops.index].to_dict(orient='records')
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
