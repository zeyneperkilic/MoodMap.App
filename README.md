# Mood-Based Music Recommendation System

This project is a web application that recommends music based on moods using Spotify's API and machine learning techniques.

## Project Structure
```
MoodBasedMusicRec/
├── backend/           # FastAPI backend server
├── frontend/         # Flask frontend server
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

## Setup Instructions

1. Install Python dependencies:
```bash
pip3 install -r requirements.txt
```

2. Set up Spotify API credentials:
- Create a Spotify Developer account
- Create a new application
- Set the redirect URI to `http://localhost:5001/callback`
- Add your client ID and secret to `backend/config.py`

3. Start the backend server:
```bash
cd backend
python3 api.py
```

4. Start the frontend server:
```bash
cd frontend
python3 frontend.py
```

5. Open your browser and go to:
```
http://localhost:5001
```

## Features
- Mood-based song recommendations
- Interactive 3D visualization of song clusters
- Spotify player integration
- Adjustable recommendation intensity
- Playlist creation
- User feedback system

## Technologies Used
- Backend: FastAPI, pandas, scikit-learn
- Frontend: Flask, Plotly.js, Chart.js
- Database: CSV files for song data, JSON for feedback
- APIs: Spotify Web API

## Notes
- The backend runs on port 8000
- The frontend runs on port 5001
- Make sure both servers are running for the application to work properly 