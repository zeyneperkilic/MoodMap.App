#!/usr/bin/env python3
import os, json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def main():
    base_dir       = os.path.dirname(os.path.abspath(__file__))
    proc_csv       = os.path.join(base_dir, "processed_data.csv")
    pred_csv       = os.path.join(base_dir, "processed_data_with_sentiment.csv")
    # feedback_data.json burada data/ altındaysa:
    feedback_json  = os.path.join(base_dir, "data", "feedback_data.json")

    # 1) Load
    df_proc = pd.read_csv(proc_csv)
    df_pred = pd.read_csv(pred_csv)
    with open(feedback_json, "r", encoding="utf-8") as f:
        feedback = json.load(f)

    # 2) Build feedback DataFrame
    analyzer = SentimentIntensityAnalyzer()
    records = []
    for uri, info in feedback.items():
        comments = info.get("comments", [])
        if not comments: continue
        gt = np.mean([analyzer.polarity_scores(c)["compound"] for c in comments]) * 10
        intensity = int(info.get("intensity", 5))
        records.append({"uri": uri, "feedback_score": gt, "intensity": intensity})
    df_fb = pd.DataFrame(records)

    # 3) Merge
    df = (df_fb
          .merge(df_proc, on="uri", how="inner")
          .merge(df_pred, on="uri", how="inner")
          .dropna(subset=["feedback_score"]))

    # 4) Debug: Mevcut sütunlar
    print("\n▶ Available columns after merge:\n", df.columns.tolist(), "\n")

    # 5) Özellik isimlerini buraya uygun olarak düzeltin:
    audio_feats = ["danceability", "energy", "valence", "tempo"]
    # eğer çıktı print’inde farklı isimler (örn. 'Danceability' veya 'danceability ' vb.) görürsen 
    # audio_feats listesini ona göre güncelleyin.

    # 6) Hazırla
    X_audio   = df[audio_feats]
    y         = df["feedback_score"]
    model_sent= df.apply(lambda r: r[f"pred_sentiment_{r.intensity}"] * 10, axis=1)

    # 7) Split
    idx_tr, idx_te = train_test_split(df.index, test_size=0.2, random_state=42)
    X_tr, X_te     = X_audio.loc[idx_tr], X_audio.loc[idx_te]
    y_tr, y_te     = y.loc[idx_tr], y.loc[idx_te]
    text_te        = model_sent.loc[idx_te]

    # 8) Audio‑only
    audio_model  = LinearRegression().fit(X_tr, y_tr)
    weighted_te  = audio_model.predict(X_te)
    rho_audio, _ = spearmanr(weighted_te, y_te)

    # 9) Text‑only & Fixed blend
    rho_text, _ = spearmanr(text_te, y_te)
    rho_fixed, _ = spearmanr(0.5*weighted_te + 0.5*text_te, y_te)

    # 10) Learned blend (meta‑model)
    X_meta       = np.column_stack([weighted_te, text_te])
    meta_model   = LinearRegression().fit(X_meta, y_te)
    blended_te   = meta_model.predict(X_meta)
    rho_meta, _  = spearmanr(blended_te, y_te)

    # 11) Çıktı
    print(f"Audio-only Spearman:         {rho_audio:.3f}")
    print(f"Text-only Spearman:          {rho_text:.3f}")
    print(f"Fixed 50/50 blend Spearman:  {rho_fixed:.3f}")
    print(f"Learned blend weights:       {meta_model.coef_}")
    print(f"Learned blend Spearman:      {rho_meta:.3f}")

if __name__ == "__main__":
    main()