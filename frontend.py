from flask import Flask, render_template, request
import requests

app = Flask(__name__)

API_URL = "http://127.0.0.1:8000"

@app.route('/')
def home():
    return render_template('cover.html')

@app.route('/mood-selection')
def mood_selection():
    return render_template('mood_selection.html')

@app.route('/mood-map/<color>')
def mood_map(color):
    color = color.lower()
    color_to_cluster = {"red": 0, "black": 1, "yellow": 2, "green": 3}

    if color not in color_to_cluster:
        return "Invalid color", 400

    cluster_id = color_to_cluster[color]

    # Sadece ilgili cluster’ın şarkılarını al
    response = requests.get(f"{API_URL}/clusters/{cluster_id}")
    songs = response.json().get("songs", []) if response.status_code == 200 else []

    return render_template(
        "mood_map.html",
        mood_name=color.capitalize(),
        mood_color=color,
        cluster_id=cluster_id,
        songs=songs
    )

@app.route('/recommend', methods=['GET'])
def recommend():
    cluster_id = request.args.get("cluster_id")
    intensity = request.args.get("intensity")

    if not cluster_id or not intensity:
        return "Missing parameters", 400


    response = requests.get(f"{API_URL}/recommend/{cluster_id}?intensity={intensity}")
    songs = response.json().get("songs", []) if response.status_code == 200 else []

    return render_template(
        "mood_map.html",
        mood_name="Recommended Songs",
        mood_color="gray",
        cluster_id=cluster_id,
        songs=songs,
        all_songs=[]
    )

if __name__ == '__main__':
    app.run(debug=True)