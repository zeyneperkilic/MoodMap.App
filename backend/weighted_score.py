import pandas as pd
from sklearn.linear_model import LinearRegression
import json

# CSV dosyanı oku
df = pd.read_csv("processed_data_with_sentiment.csv")

# Kullanacağımız özellikler
features = ["danceability", "energy", "valence", "tempo"]
X = df[features]

# Ağırlık sözlüğü
weights_by_intensity = {}

# Her intensity için regresyon modeli eğit ve ağırlıkları kaydet
for i in range(1, 11):
    y = df[f"pred_sentiment_{i}"] * 10  # Sentiment'ı 0–10 aralığına çekiyoruz
    model = LinearRegression()
    model.fit(X, y)
    weights_by_intensity[i] = {f: round(w, 4) for f, w in zip(features, model.coef_)}

# JSON dosyasına kaydet
with open("regression_weights.json", "w") as f:
    json.dump(weights_by_intensity, f, indent=4)

print("✅ scores are saved to regression_weights.json.")