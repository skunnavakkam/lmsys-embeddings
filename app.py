import sqlite3
import json
from flask import Flask, render_template, jsonify, request
import numpy as np
from sklearn.decomposition import PCA


# Global variable to store PCA object
pca = None


def get_db_connection():
    conn = sqlite3.connect("data.db")
    conn.row_factory = sqlite3.Row
    return conn


def initialize_pca():
    global pca
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT embedding FROM chatbot_arena")
    all_embeddings = cursor.fetchall()
    conn.close()

    all_embeddings_array = np.array(
        [json.loads(row["embedding"]) for row in all_embeddings]
    )
    pca = PCA(n_components=2)
    pca.fit(all_embeddings_array)


initialize_pca()


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_models")
def get_models():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT model_a FROM chatbot_arena UNION SELECT DISTINCT model_b FROM chatbot_arena"
    )
    models = cursor.fetchall()
    conn.close()
    return jsonify([row[0] for row in models])


@app.route("/get_data")
def get_data():
    model_a = request.args.get("model_a")
    model_b = request.args.get("model_b")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM chatbot_arena WHERE (model_a = ? AND model_b = ?) OR (model_a = ? AND model_b = ?)",
        (model_a, model_b, model_b, model_a),
    )
    data = cursor.fetchall()
    conn.close()

    processed_data = []
    for row in data:
        embeddings = json.loads(row["embedding"])
        processed_data.append(
            {
                "model_a": row["model_a"],
                "model_b": row["model_b"],
                "winner": row["winner"],
                "embeddings": embeddings,
            }
        )

    # Apply PCA
    embeddings_array = np.array([d["embeddings"] for d in processed_data])
    pca_result = pca.transform(embeddings_array)

    for i, d in enumerate(processed_data):
        d["pca_x"] = float(pca_result[i, 0])
        d["pca_y"] = float(pca_result[i, 1])

    return jsonify(processed_data)


if __name__ == "__main__":
    app.run(debug=True)
