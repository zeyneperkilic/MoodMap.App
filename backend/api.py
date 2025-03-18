from fastapi import FastAPI, Request, HTTPException
import pandas as pd
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import uvicorn
from textblob import TextBlob
from fastapi.middleware.cors import CORSMiddleware




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Geliştirme için *; production'da domainini yaz
    allow_credentials=True,
    allow_methods=["*"],   # GET, POST, OPTIONS hepsine izin ver
    allow_headers=["*"],   # Tüm headerlara izin ver
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

    return {"songs": cluster_songs[['uri', 'Cluster', 'PCA1', 'PCA2', 'PCA3']].to_dict(orient="records")}

@app.get("/recommend/{cluster_id}")
def recommend_songs(cluster_id: int, intensity: int = 5):
    try:
        cluster_songs = data[data["Cluster"] == cluster_id].copy()
        if cluster_songs.empty:
            return {"cluster": cluster_id, "songs": []}

        if intensity >= 5:
            sorted_songs = cluster_songs.sort_values(by=['energy', 'valence'], ascending=[False, False])
        else:
            sorted_songs = cluster_songs.sort_values(by=['energy', 'valence'], ascending=[True, True])

        liked_songs = [song for song, feedback in feedback_data.items() if feedback.get("likes", 0) > feedback.get("dislikes", 0)]
        disliked_songs = [song for song, feedback in feedback_data.items() if feedback.get("dislikes", 0) > feedback.get("likes", 0)]

        if liked_songs:
            liked_features = data[data["uri"].isin(liked_songs)][features]
            if not liked_features.empty and not cluster_songs.empty:
                similarity_scores = cosine_similarity(liked_features, cluster_songs[features])
                cluster_songs['similarity'] = similarity_scores.mean(axis=0)
                cluster_songs = cluster_songs.sort_values(by="similarity", ascending=False)

        if disliked_songs:
            disliked_features = data[data["uri"].isin(disliked_songs)][features]
            if not disliked_features.empty and not cluster_songs.empty:
                similarity_scores = cosine_similarity(disliked_features, cluster_songs[features])
                cluster_songs['similarity'] = similarity_scores.mean(axis=0)
                cluster_songs = cluster_songs.sort_values(by="similarity", ascending=True)

        recommendations = sorted_songs.head(20)[["uri", "PCA1", "PCA2", "PCA3"]].to_dict(orient="records")
        return {"cluster": cluster_id, "songs": recommendations}

    except Exception as e:
        print(f"Error in recommend_songs: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/feedback")
async def save_feedback(request: Request):

    data = await request.json()
    song = data.get("song", "Unknown")
    liked = data.get("liked", False)
    comment = data.get("comment", "")

    if song not in feedback_data:
        feedback_data[song] = {"likes": 0, "dislikes": 0, "comments": [], "sentiment": 0}

    if liked:
        feedback_data[song]["likes"] += 1
    else:
        feedback_data[song]["dislikes"] += 1

    if comment:
        feedback_data[song]["comments"].append(comment)
        feedback_data[song]["sentiment"] = analyze_sentiment(comment)

    with open(feedback_file, "w") as f:
        json.dump(feedback_data, f, indent=4)

    return {"message": f"Feedback received for {song}"}

def analyze_sentiment(text):

    analysis = TextBlob(text)
    return analysis.sentiment.polarity

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)