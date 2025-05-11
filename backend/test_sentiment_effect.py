#!/usr/bin/env python3
import json
import sys
import os
import random
import numpy as np
from datetime import datetime
from scipy.stats import spearmanr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from api import recommend_songs

# -- Ayar: deterministik sonuçlar için sabit seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Bootstrap ayarları
N_BOOTSTRAP = 10000  
ALPHA       = 0.05       

def load_feedback(path=None):
    fn = path if path else os.path.join(os.path.dirname(__file__), "feedback_data.json")
    if not os.path.exists(fn):
        raise FileNotFoundError(f"Could not find feedback file at {fn}")
    return json.load(open(fn))

def main():
    # 1) Feedback verisini yükle
    fb_path  = sys.argv[1] if len(sys.argv) > 1 else None
    feedback = load_feedback(fb_path)
    analyzer = SentimentIntensityAnalyzer()

    weighted_scores = []
    combined_scores = []
    ground_truths   = []
    clusters        = []
    intensities     = []

    # 2) Her şarkı için ground-truth ve model skorlarını topla
    for uri, entry in feedback.items():
        comments = entry.get("comments", [])
        if not comments:
            continue

        # ground truth: yorumların ortalama VADER compound değeri
        gt = np.mean([analyzer.polarity_scores(c)["compound"] for c in comments])

        cluster   = int(entry.get("cluster_id", -1))
        intensity = int(entry.get("intensity", 5))

        recs = recommend_songs(cluster, intensity).get("songs", [])
        for r in recs:
            if r.get("uri") == uri:
                # Doğrudan API'den gelen ağırlıklı ve kombine skorlar
                ws = r.get("weighted_score", 0)
                cs = r.get("combined_score", 0)

                weighted_scores.append(ws)
                combined_scores.append(cs)
                ground_truths.append(gt)
                clusters.append(cluster)
                intensities.append(intensity)
                break

    if not ground_truths:
        print("⚠️ No matching feedback URIs found in recommendations.")
        sys.exit(1)

    # 3) Global Spearman ρ
    w_corr, _     = spearmanr(weighted_scores, ground_truths)
    c_corr, _     = spearmanr(combined_scores, ground_truths)
    delta_rho     = c_corr - w_corr

    # 4) Bootstrap ile Δρ güven aralığı ve p-değeri
    diffs = []
    n     = len(ground_truths)
    for _ in range(N_BOOTSTRAP):
        idx = np.random.randint(0, n, n)
        w_b, _ = spearmanr([weighted_scores[i] for i in idx],
                           [ground_truths[i]   for i in idx])
        c_b, _ = spearmanr([combined_scores[i] for i in idx],
                           [ground_truths[i]   for i in idx])
        diffs.append(c_b - w_b)
    diffs = np.array(diffs)
    lo, hi  = np.percentile(diffs, [100 * ALPHA/2, 100 * (1-ALPHA/2)])
    p_value = np.mean(np.abs(diffs) >= abs(delta_rho))

    # 5) Küme-bazlı Spearman ρ
    cluster_stats = {}
    for c in sorted(set(clusters)):
        inds = [i for i, cl in enumerate(clusters) if cl == c]
        if len(inds) > 1:
            rho_c, _ = spearmanr([combined_scores[i] for i in inds],
                                 [ground_truths[i]   for i in inds])
            cluster_stats[f"spearman_combined_cluster{c}"] = rho_c

    # 6) Yoğunluk-bazlı Spearman ρ
    intensity_stats = {}
    for i in sorted(set(intensities)):
        inds = [j for j, it in enumerate(intensities) if it == i]
        if len(inds) > 1:
            rho_i, _ = spearmanr([combined_scores[j] for j in inds],
                                 [ground_truths[j]   for j in inds])
            intensity_stats[f"spearman_combined_{i}"] = rho_i

    # 7) stats.json’u oluştur ve yaz
    stats = {
        "version":          os.environ.get("CI_COMMIT_TAG", os.environ.get("CI_BUILD_NUMBER", "local")),
        "timestamp":        datetime.utcnow().isoformat() + "Z",
        "spearman_weighted":w_corr,
        "spearman_combined":c_corr,
        "delta_rho":        delta_rho,
        "delta_ci":         [lo, hi],
        "p_value":          p_value
    }
    stats.update(cluster_stats)
    stats.update(intensity_stats)

    with open("stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("✅ new stats.json saved.")

if __name__ == "__main__":
    main()