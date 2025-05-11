# backend/predict_all_sentiment.py
import os
import pandas as pd
import joblib

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV   = os.path.join(BASE_DIR, "processed_data.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "sentiment_model.joblib")
OUTPUT_CSV  = os.path.join(BASE_DIR, "processed_data_with_sentiment.csv")

print("🔄 Loading processed dataset…")
df = pd.read_csv(INPUT_CSV)

# print column list so we can verify what’s actually there
print("▶ Columns in CSV:", df.columns.tolist())

# if your clustering column is called "labels", rename it so our feature list matches
if "labels" in df.columns:
    df = df.rename(columns={"labels": "cluster_id"})

# now we MUST have cluster_id
if "cluster_id" not in df.columns:
    raise KeyError("Neither 'cluster_id' nor 'labels' found in processed_data.csv")

print("🔄 Loading sentiment regression model…")
model = joblib.load(MODEL_PATH)

# these are exactly the features you used at training time
features = ["danceability", "energy", "valence", "tempo", "cluster_id"]

for intensity in range(1, 11):
    print(f"👉 Predicting for intensity = {intensity}…")
    X = df[features].copy()
    X["intensity"] = intensity
    df[f"pred_sentiment_{intensity}"] = model.predict(X)

print(f"✅ Saving predictions to {OUTPUT_CSV}")
df.to_csv(OUTPUT_CSV, index=False)
print("🎉 All done!")