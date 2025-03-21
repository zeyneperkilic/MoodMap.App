from fastapi import FastAPI, Request, HTTPException
import pandas as pd
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import uvicorn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fastapi.middleware.cors import CORSMiddleware
import logging
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file_path = "processed_data.csv"
data = pd.read_csv(file_path)

features = ['danceability', 'energy', 'valence', 'tempo']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_scaled)
data['PCA1'] = pca_result[:, 0]
data['PCA2'] = pca_result[:, 1]
data['PCA3'] = pca_result[:, 2]

feedback_file = "feedback.json"

try:
    with open(feedback_file, "r") as f:
        feedback_data = json.load(f)
except:
    feedback_data = {}

logging.basicConfig(filename='recommendation_check.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def log_weighted_check(cluster_id, intensity, songs):
    try:
        ascending = intensity < 5
        scores = []
        for song in songs:
            score = 0.4 * song.get('energy', 0) + 0.3 * song.get('valence', 0) + 0.2 * song.get('danceability', 0) + 0.1 * song.get('tempo', 0)
            scores.append(score)
        sorted_scores = sorted(scores, reverse=not ascending)

        if scores == sorted_scores:
            logging.info(f"Cluster {cluster_id}, intensity={intensity} weighted score correct.")
        else:
            logging.error(f"Cluster {cluster_id}, intensity={intensity} weighted score not correct.")
    except Exception as e:
        logging.error(f"Weighted control error: {str(e)}")

@app.get("/")
def home():
    return {"message": "Welcome to the Mood-Based Music Recommendation API"}

@app.get("/feedback")
def get_feedback():
    return feedback_data

@app.get("/clusters/{cluster_id}")
def get_cluster_songs(cluster_id: int):
    cluster_songs = data[data["Cluster"] == cluster_id]
    if cluster_songs.empty:
        return {"songs": []}
    return {"songs": cluster_songs.to_dict(orient="records")}


@app.get("/recommend/{cluster_id}")
def recommend_songs(cluster_id: int, intensity: int = 5):
    try:
        cluster_songs = data[data["Cluster"] == cluster_id].copy()
        if cluster_songs.empty:
            return {"cluster": cluster_id, "songs": []}

        # Sad cluster için ek filtreler (örneğin cluster 0 = sad)
        if cluster_id == 0:
            cluster_songs = cluster_songs[
                (cluster_songs['energy'] < 0.7) &
                (cluster_songs['valence'] < 0.7) &
                (cluster_songs['tempo'] < 115) &
                (cluster_songs['danceability'] < 0.7)
                ]

        # Intensity'e göre tempo ve energy aralığı filtresi (diğer moodlar için)
        else:
            min_tempo = 60 + (intensity * 8)   # örneğin 5 intensity => min_tempo = 100
            max_tempo = 180 - (intensity * 3)
            min_energy = intensity * 0.1
            max_energy = min(1.0, 0.6 + (intensity * 0.05))

            cluster_songs = cluster_songs[
                (cluster_songs['tempo'] >= min_tempo) &
                (cluster_songs['tempo'] <= max_tempo) &
                (cluster_songs['energy'] >= min_energy) &
                (cluster_songs['energy'] <= max_energy)
            ]

        ascending = intensity < 5
        cluster_songs['weighted_score'] = (
            0.4 * cluster_songs['energy'] +
            0.3 * cluster_songs['valence'] +
            0.2 * cluster_songs['danceability'] +
            0.1 * cluster_songs['tempo']
        )

        cluster_songs = cluster_songs.sort_values(by='weighted_score', ascending=ascending)
        top_songs = cluster_songs.head(50)
        recommendations = top_songs.sample(n=min(10, len(top_songs)))[
            ["uri", "PCA1", "PCA2", "PCA3", "energy", "valence", "tempo", "danceability"]
        ].to_dict(orient="records")

        log_weighted_check(cluster_id, intensity, recommendations)
        return {"cluster": cluster_id, "songs": recommendations}

    except Exception as e:
        print(f"Error in recommend_songs: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/feedback")
async def save_feedback(request: Request):
    data = await request.json()
    song = data.get("song", "Unknown")
    liked = data.get("liked")
    comment = data.get("comment", "")
    cluster_id = data.get("cluster_id")
    intensity = data.get("intensity")
    weighted_score = data.get("weighted_score")

    if song not in feedback_data:
        feedback_data[song] = {
            "likes": 0,
            "dislikes": 0,
            "comments": [],
            "sentiment": 0,
            "cluster_id": cluster_id,
            "intensity": intensity,
            "weighted_score": weighted_score
        }
    else:
        # varsa güncelle
        feedback_data[song]["cluster_id"] = cluster_id
        feedback_data[song]["intensity"] = intensity
        feedback_data[song]["weighted_score"] = weighted_score

    if liked is True:
        feedback_data[song]["likes"] += 1
    elif liked is False:
        feedback_data[song]["dislikes"] += 1

    if comment:
        feedback_data[song]["comments"].append(comment)
        feedback_data[song]["sentiment"] = analyze_sentiment_vader(comment)

    with open(feedback_file, "w") as f:
        json.dump(feedback_data, f, indent=4)

    return {"message": f"Feedback received for {song}"}

def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)['compound']
    return sentiment_score

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)