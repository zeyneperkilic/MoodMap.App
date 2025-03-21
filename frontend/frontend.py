from flask import Flask, render_template, request
import requests

app = Flask(__name__)

import os
API_URL = "http://localhost:8000"

@app.route('/')
def home():
    return render_template('cover.html')

@app.route('/mood-selection')
def mood_selection():
    return render_template('mood_selection.html')

@app.route('/mood-map/<color>')
def mood_map(color):
    color = color.lower()
    color_to_cluster = {
        "black": 0,  # sad
        "yellow": 1,  # happy
        "red": 2,  # energetic
        "green": 3  # calm
    }

    if color not in color_to_cluster:
        return "Invalid color", 400

    cluster_id = color_to_cluster[color]

    response = requests.get(f"{API_URL}/clusters/{cluster_id}")
    songs = response.json().get("songs", []) if response.status_code == 200 else []

    return render_template(
        "mood_map.html",
        mood_name=color.capitalize(),
        mood_color=color,
        cluster_id=cluster_id,
        songs=songs,
        all_songs=songs
    )

@app.route('/recommend', methods=['GET'])
def recommend():
    cluster_id = request.args.get("cluster_id")
    intensity = request.args.get("intensity")

    if not cluster_id or not intensity:
        return "Missing parameters", 400

    response = requests.get(f"{API_URL}/recommend/{cluster_id}?intensity={intensity}")
    songs = response.json().get("songs", []) if response.status_code == 200 else []

    # Cluster renklendirmesi için mapping
    cluster_to_color = { "0": "black", "1": "yellow", "2": "red", "3": "green" }
    mood_color = cluster_to_color.get(str(cluster_id), "gray")

    return render_template(
        "mood_map.html",
        mood_name="Recommended Songs",
        mood_color=mood_color,   # burası önemli
        cluster_id=cluster_id,
        songs=songs,
        all_songs=songs
    )
@app.route('/feedback')
def feedback():
    response = requests.get(f"{API_URL}/feedback")
    feedback_data = response.json() if response.status_code == 200 else {}
    return feedback_data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
