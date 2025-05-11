# train_model.py
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROC_PATH = os.path.join(BASE_DIR, "processed_data.csv")
FB_PATH   = os.path.join(BASE_DIR, "data", "feedback_data.json")
MODEL_OUT = os.path.join(BASE_DIR, "sentiment_model.joblib")

print("Loading processed data…")
df_proc = pd.read_csv(PROC_PATH)

print("Loading feedback data…")
with open(FB_PATH, "r", encoding="utf-8") as f:
    raw_fb = json.load(f)

records = []
for uri, info in raw_fb.items():
    try:
        intensity = int(info.get("intensity", 0))
    except:
        intensity = 0
    records.append({
        "uri": uri,
        "cluster_id": info.get("cluster_id", np.nan),
        "intensity": intensity,
        "sentiment": info.get("sentiment", np.nan),
    })
df_fb = pd.DataFrame(records).dropna(subset=["sentiment"] )

print("Merging features with feedback…")
df = pd.merge(df_fb, df_proc, on="uri", how="inner")
if df.empty:
    raise RuntimeError("There is no song matched with the feedback!")


FEATURES = ["danceability", "energy", "valence", "tempo", "cluster_id", "intensity"]
X = df[FEATURES]
y = df["sentiment"]

print("Splitting data…")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training RandomForestRegressor…")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, MODEL_OUT)
print(f"Model saved: {MODEL_OUT}")

print("Evaluating…")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"  MSE: {mse:.4f}")
print(f"  R²:  {r2:.4f}")

print("Done.")
