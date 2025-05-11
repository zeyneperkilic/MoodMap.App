# backend/train_classifier.py
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROC_PATH = os.path.join(BASE_DIR, "processed_data.csv")
FB_PATH   = os.path.join(BASE_DIR, "data", "feedback_data.json")
MODEL_OUT = os.path.join(BASE_DIR, "sentiment_classifier.joblib")

df_proc = pd.read_csv(PROC_PATH)
with open(FB_PATH, "r", encoding="utf-8") as f:
    raw_fb = json.load(f)

# 2) Feedback → DataFrame
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
df_fb = pd.DataFrame(records).dropna(subset=["sentiment"])

# 3) combine
df = pd.merge(df_fb, df_proc, on="uri", how="inner")
if df.empty:
    raise RuntimeError("There is no song matched with the feedback!")

def to_label(x):
    if x <= -0.1: return 0    # negatif
    if x >=  0.1: return 2    # pozitif
    return 1                  # nötr

df["label"] = df["sentiment"].apply(to_label)

# 5) train the model
FEATURES = ["danceability", "energy", "valence", "tempo", "cluster_id", "intensity"]
X = df[FEATURES]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

joblib.dump(clf, MODEL_OUT)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f" Classifier saved: {MODEL_OUT}")
print(f"  Accuracy: {acc:.3f}")
print(classification_report(y_test, y_pred))