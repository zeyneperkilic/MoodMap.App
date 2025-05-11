from fastapi import FastAPI, Request, HTTPException, Depends, Query
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
import os
from spotify_auth import get_auth_url, get_token_info, get_spotify_client, refresh_token_if_expired
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from session import session_manager
import spotipy
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import Optional

PCA_WEIGHTS = {
    "danceability": 0.28,
    "energy":       0.27,
    "valence":      0.31,
    "tempo":        0.14
}

app = FastAPI()

# Frontend dizinini belirt
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
templates_dir = os.path.join(frontend_dir, "templates")

# Statik dosyaları ve template'leri ayarla
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_dir = os.path.dirname(os.path.abspath(__file__))

# 1) Base ve pred CSV yolları
base_csv = os.path.join(current_dir, "processed_data.csv")
pred_csv = os.path.join(current_dir, "processed_data_with_sentiment.csv")

# 2) DataFrame’leri yükle
df_base = pd.read_csv(base_csv)
df_pred = pd.read_csv(pred_csv)

# 3) Sadece uri + pred_sentiment_1…pred_sentiment_10 al
pred_cols     = ["uri"] + [f"pred_sentiment_{i}" for i in range(1, 11)]
df_pred_small = df_pred[pred_cols]

# 4) Merge: böylece base’deki tüm audio-özellikler bozulmadan kalır
data = df_base.merge(df_pred_small, on="uri", how="left")

# Özellikleri ölçekle ve kümele
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

# Feedback dosyası (full path kullanmak da opsiyonel)
feedback_file = os.path.join(current_dir, "feedback_data.json")
try:
    with open(feedback_file, "r") as f:
        feedback_data = json.load(f)
except FileNotFoundError:
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

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("cover.html", {"request": request})

@app.get("/mood_selection", response_class=HTMLResponse)
async def mood_selection(request: Request):
    return templates.TemplateResponse("mood_selection.html", {"request": request})

@app.get("/mood_map", response_class=HTMLResponse)
async def mood_map(request: Request):
    return templates.TemplateResponse("mood_map.html", {"request": request})

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
        # 1) pull all tracks for this cluster
        cluster_all = data[data["Cluster"] == cluster_id].copy()
        if cluster_all.empty:
            return {"cluster": cluster_id, "songs": []}

        # 2) apply your per‑cluster filtering
        if cluster_id == 0:
            # Sad
            filtered = cluster_all[
                (cluster_all.energy < 0.7) &
                (cluster_all.valence < 0.7) &
                (cluster_all.tempo < 115) &
                (cluster_all.danceability < 0.7)
            ]
        elif cluster_id == 3:
            # Calm
            max_tempo  = 120 - (intensity * 4)
            max_energy = 1.0   - (intensity * 0.05)
            filtered = cluster_all[
                (cluster_all.tempo <= max_tempo) &
                (cluster_all.energy <= max_energy)
            ]
        else:
            # Happy (1) & Energetic (2)
            min_tempo  = 60  + (intensity * 8)
            max_tempo  = 180 - (intensity * 3)
            min_energy = intensity * 0.1
            max_energy = min(1.0, 0.6 + (intensity * 0.05))
            filtered = cluster_all[
                (cluster_all.tempo  >= min_tempo) &
                (cluster_all.tempo  <= max_tempo) &
                (cluster_all.energy >= min_energy) &
                (cluster_all.energy <= max_energy)
            ]

        # 3) fallback to full cluster if empty OR high‑intensity calm/energetic
        if (filtered.empty
            or (cluster_id == 3 and intensity >= 8)
            or (cluster_id == 2 and intensity >= 10)
        ):
            fb = cluster_all.copy()
            fb["weighted_score"] = (
                PCA_WEIGHTS["energy"]       * fb.energy +
                PCA_WEIGHTS["valence"]      * fb.valence +
                PCA_WEIGHTS["danceability"] * fb.danceability +
                PCA_WEIGHTS["tempo"]        * fb.tempo
            )
            # bring sentiment into 0–10 range
            fb["model_sentiment"] = fb[f"pred_sentiment_{intensity}"].fillna(0) * 10
            # blend 50/50
            fb["combined_score"]  = 0.5 * fb["weighted_score"] + 0.5 * fb["model_sentiment"]

            picks = fb.sample(n=min(10, len(fb))).copy()
            recs = picks[[
                "uri","PCA1","PCA2","PCA3",
                "energy","valence","tempo","danceability",
                "weighted_score","model_sentiment","combined_score"
            ]].to_dict(orient="records")

            return {"cluster": cluster_id, "songs": recs}

        # 4) otherwise score the filtered set
        filtered["weighted_score"] = (
            PCA_WEIGHTS["energy"]       * filtered.energy +
            PCA_WEIGHTS["valence"]      * filtered.valence +
            PCA_WEIGHTS["danceability"] * filtered.danceability +
            PCA_WEIGHTS["tempo"]        * filtered.tempo
        )
        filtered["model_sentiment"] = filtered[f"pred_sentiment_{intensity}"].fillna(0) * 10
        filtered["combined_score"]  = 0.5 * filtered["weighted_score"] + 0.5 * filtered["model_sentiment"]

        # 5) sort & sample top 50 → 10
        ascending    = True if cluster_id == 3 else (intensity < 5)
        top50        = filtered.sort_values("combined_score", ascending=ascending).head(50)
        picks        = top50.sample(n=min(10, len(top50))).copy()

        recs = picks[[
            "uri","PCA1","PCA2","PCA3",
            "energy","valence","tempo","danceability",
            "weighted_score","model_sentiment","combined_score"
        ]].to_dict(orient="records")

        return {"cluster": cluster_id, "songs": recs}

    except Exception as e:
        logging.error(f"Error in recommend_songs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/feedback")
async def save_feedback(request: Request):
    body = await request.json()
    song           = body.get("song", "Unknown")
    liked          = body.get("liked")
    comment        = body.get("comment", "")
    cluster_id     = body.get("cluster_id")
    intensity      = body.get("intensity")
    weighted_score = body.get("weighted_score")

    entry = feedback_data.setdefault(song, {
        "likes": 0, "dislikes": 0, "comments": [], "sentiment": 0,
        "cluster_id": cluster_id, "intensity": intensity, "weighted_score": weighted_score
    })
    if liked is True:
        entry["likes"] += 1
    elif liked is False:
        entry["dislikes"] += 1

    if comment:
        entry["comments"].append(comment)
        entry["sentiment"] = SentimentIntensityAnalyzer().polarity_scores(comment)["compound"]

    with open(feedback_file, "w") as f:
        json.dump(feedback_data, f, indent=4)
    return {"message": f"Feedback received for {song}"}

def analyze_sentiment_vader(text):
    return SentimentIntensityAnalyzer().polarity_scores(text)["compound"]

@app.get("/spotify/login")
async def spotify_login():
    try:
        auth_url = get_auth_url()
        return RedirectResponse(url=auth_url, status_code=303)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/callback")
async def spotify_callback(code: str = Query(None), request: Request = None):
    if not code:
        raise HTTPException(status_code=400, detail="No authorization code provided")
    token_info = get_token_info(code)
    resp = RedirectResponse(url="http://127.0.0.1:5001/mood-selection", status_code=303)
    session_manager.set_session(resp, {"token_info": token_info})
    return resp

@app.get("/spotify/me")
async def get_spotify_user(request: Request):
    sess       = session_manager.get_session(request)
    token_info = sess.get("token_info")
    if not token_info:
        raise HTTPException(status_code=401, detail="No token info in session")
    token_info = refresh_token_if_expired(token_info)
    response   = JSONResponse(content={"message": "Success"})
    session_manager.set_session(response, {"token_info": token_info})
    return response


# Renk ve duygu eşleştirmeleri
COLOR_MOOD_MAPPING = {
    "red": {
        "moods": ["energetic", "passionate", "intense"],
        "description": "High energy and intense emotions",
        "genres": ["rock", "metal", "punk", "electronic"],
        "name": "Energetic"
    },
    "black": {
        "moods": ["mysterious", "deep", "emotional"],
        "description": "Deep and introspective feelings",
        "genres": ["alternative", "indie", "dark jazz", "ambient"],
        "name": "Sad"
    },
    "yellow": {
        "moods": ["happy", "upbeat", "cheerful"],
        "description": "Bright and positive vibes",
        "genres": ["pop", "dance", "funk", "disco"],
        "name": "Happy"
    },
    "green": {
        "moods": ["peaceful", "calm", "relaxed"],
        "description": "Peaceful and harmonious state",
        "genres": ["classical", "jazz", "acoustic", "ambient"],
        "name": "Calm"
    }
}

@app.get("/mood-map/{color}")
def get_mood_map(color: str):
    color_to_cluster = {
        "black": 0,  # Sad
        "yellow": 1, # Happy
        "red": 2,    # Energetic
        "green": 3   # Calm
    }
    
    cluster_id = color_to_cluster.get(color.lower())
    if cluster_id is None:
        raise HTTPException(status_code=404, detail="Invalid mood color")
    
    mood_names = {
        "black": "Sad",
        "yellow": "Happy",
        "red": "Energetic",
        "green": "Calm"
    }
    
    cluster_songs = data[data["Cluster"] == cluster_id].copy()
    songs = cluster_songs[["uri", "PCA1", "PCA2", "PCA3", "energy", "valence", "tempo", "danceability"]].to_dict(orient="records")
    
    return {
        "mood_name": mood_names.get(color.lower(), "Unknown"),
        "mood_color": color.lower(),
        "cluster_id": cluster_id,
        "all_songs": songs
    }

@app.get("/recommendations/{color}")
async def get_recommendations(color: str, request: Request):
    if color not in COLOR_MOOD_MAPPING:
        raise HTTPException(status_code=404, detail="Color not found")
    
    session_data = session_manager.get_session(request)
    token_info = session_data.get("token_info")
    
    if token_info:
        try:
            token_info = refresh_token_if_expired(token_info)
            sp = get_spotify_client(token_info)
            
            # Renk için uygun türleri al
            genres = COLOR_MOOD_MAPPING[color]["genres"]
            
            # Spotify'dan önerileri al
            recommendations = sp.recommendations(
                seed_genres=genres[:5],  # En fazla 5 tür kullanabiliriz
                limit=10,
                target_energy=0.8 if color == "red" else 0.6 if color == "yellow" else 0.4,
                target_valence=0.8 if color in ["yellow", "green"] else 0.4
            )
            
            # Önerileri formatlayıp döndür
            tracks = []
            for track in recommendations["tracks"]:
                tracks.append({
                    "name": track["name"],
                    "artist": track["artists"][0]["name"],
                    "album": track["album"]["name"],
                    "preview_url": track["preview_url"],
                    "external_url": track["external_urls"]["spotify"]
                })
            
            return {"tracks": tracks}
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Spotify girişi yapılmamışsa, varsayılan öneriler
        return {
            "tracks": [
                {
                    "name": "Please login with Spotify",
                    "artist": "to get personalized recommendations",
                    "album": "",
                    "preview_url": None,
                    "external_url": None
                }
            ]
        }

@app.post("/create-playlist")
async def create_playlist(request: Request):
    try:
        # Session'ı kontrol et
        session_data = session_manager.get_session(request)
        print("Session data:", session_data)
        
        token_info = session_data.get("token_info")
        print("Token info:", token_info)
        
        if not token_info:
            print("No token info found in session")
            return {"success": False, "message": "Not authenticated with Spotify"}

        # Request body'yi al
        data = await request.json()
        print("Request data:", data)
        
        playlist_name = data.get("name")
        track_uris = data.get("tracks", [])

        if not playlist_name or not track_uris:
            print("Missing playlist name or tracks")
            return {"success": False, "message": "Missing playlist name or tracks"}

        # Parse mood and intensity from playlist name
        # Expected format: "Mood X - Intensity Y"
        try:
            mood = int(playlist_name.split(" ")[1])
            intensity = int(playlist_name.split(" ")[-1])
        except:
            print("Could not parse mood and intensity from playlist name")
            mood = 0
            intensity = 5

        # Refresh token if expired
        try:
            token_info = refresh_token_if_expired(token_info)
            print("Refreshed token info:", token_info)
        except Exception as e:
            print(f"Error refreshing token: {str(e)}")
            return {"success": False, "message": f"Failed to refresh token: {str(e)}"}
        
        # Create Spotify client
        try:
            sp = get_spotify_client(token_info)
            print("Spotify client created successfully")
        except Exception as e:
            print(f"Error creating Spotify client: {str(e)}")
            return {"success": False, "message": f"Failed to create Spotify client: {str(e)}"}

        # Get user's ID
        try:
            user_id = sp.me()["id"]
            print("User ID retrieved:", user_id)
        except Exception as e:
            print(f"Error getting user ID: {str(e)}")
            return {"success": False, "message": f"Failed to get user ID: {str(e)}"}

        # Create playlist
        try:
            playlist = sp.user_playlist_create(
                user_id,
                playlist_name,
                public=True,
                description=f"Created by MoodMap - {playlist_name}"
            )
            print("Playlist created successfully:", playlist)
        except Exception as e:
            print(f"Error creating playlist: {str(e)}")
            return {"success": False, "message": f"Failed to create playlist: {str(e)}"}

        # Add tracks to playlist
        try:
            sp.playlist_add_items(playlist["id"], track_uris)
            print("Tracks added to playlist successfully")
            
            # Save mood history entry
            if user_id not in mood_history:
                mood_history[user_id] = []
            
            entry = MoodEntry(
                user_id=user_id,
                mood=mood,
                intensity=intensity,
                playlist_id=playlist["id"]
            )
            mood_history[user_id].append(entry)
            
        except Exception as e:
            print(f"Error adding tracks to playlist: {str(e)}")
            return {"success": False, "message": f"Failed to add tracks to playlist: {str(e)}"}

        return {"success": True, "message": "Playlist created successfully"}

    except Exception as e:
        print(f"Unexpected error in create_playlist: {str(e)}")
        return {"success": False, "message": f"Failed to create playlist: {str(e)}"}

@app.post("/logout")
async def logout(request: Request):
    try:
        # Session'ı temizle
        session_manager.clear_session(request)
        
        # CORS için response header'larını ayarla
        response = JSONResponse(
            content={"success": True, "message": "Logged out successfully"},
            headers={
                "Access-Control-Allow-Origin": "http://127.0.0.1:5001",
                "Access-Control-Allow-Credentials": "true"
            }
        )
        
        return response
    except Exception as e:
        print(f"Error during logout: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Failed to logout"},
            headers={
                "Access-Control-Allow-Origin": "http://127.0.0.1:5001",
                "Access-Control-Allow-Credentials": "true"
            }
        )

# Mood history için yeni model
class MoodEntry(BaseModel):
    user_id: str
    mood: int
    intensity: int
    timestamp: datetime = Field(default_factory=datetime.now)
    playlist_id: Optional[str] = None

# Mood history için in-memory storage (gerçek uygulamada veritabanı kullanılmalı)
mood_history = {}

@app.post("/mood-history/add")
async def add_mood_entry(token_info: dict = Depends(get_token_info)):
    """Kullanıcının mood geçmişine yeni bir kayıt ekler"""
    try:
        sp = get_spotify_client(token_info)
        user_id = sp.current_user()["id"]
        
        if user_id not in mood_history:
            mood_history[user_id] = []
        
        # Son 24 saat içindeki mood'ları say
        today = datetime.now()
        daily_entries = sum(1 for entry in mood_history[user_id] 
                          if (today - entry.timestamp).days < 1)
        
        if daily_entries >= 5:
            raise HTTPException(
                status_code=400,
                detail="Maximum 5 mood entries per day allowed"
            )
        
        entry = MoodEntry(
            user_id=user_id,
            mood=mood,
            intensity=intensity,
            playlist_id=playlist_id
        )
        mood_history[user_id].append(entry)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mood-history")
async def get_mood_history(request: Request):
    """Kullanıcının mood geçmişini döndürür"""
    try:
        session_data = session_manager.get_session(request)
        token_info = session_data.get("token_info")
        
        if not token_info:
            raise HTTPException(status_code=401, detail="Not authenticated")
            
        token_info = refresh_token_if_expired(token_info)
        sp = get_spotify_client(token_info)
        user_id = sp.current_user()["id"]
        
        if user_id not in mood_history:
            return {"history": []}
        
        # Son 30 günlük geçmişi döndür
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_history = [
            {
                "mood": entry.mood,
                "intensity": entry.intensity,
                "timestamp": entry.timestamp.isoformat(),
                "playlist_id": entry.playlist_id
            }
            for entry in mood_history[user_id]
            if entry.timestamp > thirty_days_ago
        ]
        
        return {"history": recent_history}
    except Exception as e:
        print(f"Error getting mood history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)