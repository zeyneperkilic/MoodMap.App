#!/usr/bin/env python3
import json
import sys
import os
import numpy as np
from datetime import datetime
from scipy.stats import spearmanr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from api import recommend_songs

def load_feedback(path=None):
    """
    Load the feedback JSON. If `path` is None, defaults to data/feedback_data.json
    """
    if path:
        fn = path
    else:
        fn = os.path.join(os.path.dirname(__file__), "data", "feedback_data.json")
    if not os.path.exists(fn):
        raise FileNotFoundError(f"Could not find feedback file at {fn}")
    return json.load(open(fn, "r", encoding="utf-8"))

def compute_correlations(feedback, analyzer):
    """
    For each track in feedback, get ground‐truth sentiment (mean VADER compound),
    then fetch recommendations for its cluster and intensity, and collect:
      - weighted_score (no-sentiment pipeline)
      - combined_score (with-sentiment pipeline)
      - ground_truth sentiment
    Returns three parallel lists: weighted_scores, combined_scores, ground_truths.
    """
    weighted_scores = []
    combined_scores = []
    ground_truths   = []

    for uri, entry in feedback.items():
        comments = entry.get("comments", [])
        if not comments:
            continue

        # 1) ground truth: avg VADER compound of all comments
        gt = np.mean([analyzer.polarity_scores(c)["compound"] for c in comments])

        # 2) read cluster & intensity from feedback
        try:
            cluster   = int(entry["cluster_id"])
            intensity = int(entry.get("intensity", 5))
        except (TypeError, ValueError):
            continue

        # 3) call your recommend_songs function
        recs = recommend_songs(cluster, intensity).get("songs", [])

        # 4) find our URI in the batch and collect scores
        for r in recs:
            if r.get("uri") == uri:
                weighted_scores.append(r.get("weighted_score", 0.0))
                combined_scores.append(r.get("combined_score", 0.0))
                ground_truths.append(gt)
                break

    return weighted_scores, combined_scores, ground_truths

def bootstrap_delta_ci(w, c, gt, n_boot=10000, alpha=0.05):
    """
    Bootstrap the difference in Spearman correlations:
      delta = spearman(c, gt) - spearman(w, gt)
    Returns (lower_ci, upper_ci, p_value_two_tailed).
    """
    diffs = np.empty(n_boot, dtype=float)
    n = len(gt)
    for i in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        w_i = spearmanr(w[idx], gt[idx]).correlation
        c_i = spearmanr(c[idx], gt[idx]).correlation
        diffs[i] = c_i - w_i

    # Confidence interval
    lower = np.percentile(diffs, 100 * (alpha/2))
    upper = np.percentile(diffs, 100 * (1 - alpha/2))

    # Approximate two-tailed p-value
    # H0: Δ ≤ 0 vs H1: Δ > 0
    p_one_tail = np.mean(diffs <= 0)
    p_two_tail = 2 * min(p_one_tail, 1 - p_one_tail)

    return lower, upper, p_two_tail

def main():
    # allow passing path as first arg
    fb_path = sys.argv[1] if len(sys.argv) > 1 else None
    feedback = load_feedback(fb_path)

    analyzer = SentimentIntensityAnalyzer()
    w_scores, c_scores, gt = compute_correlations(feedback, analyzer)

    if not gt:
        print("⚠️  No matching feedback URIs found in your recommendations.")
        sys.exit(1)

    # 5) Spearman correlations
    w_corr, _ = spearmanr(w_scores, gt)
    c_corr, _ = spearmanr(c_scores, gt)

    print(f"Spearman(weighted vs truth): {w_corr:.3f}")
    print(f"Spearman(combined vs truth): {c_corr:.3f}")

    # 6) Bootstrap confidence interval & p-value for Δρ
    lower, upper, p_val = bootstrap_delta_ci(
        np.array(w_scores),
        np.array(c_scores),
        np.array(gt),
        n_boot=10000,
        alpha=0.05
    )
    delta = c_corr - w_corr
    print(f"Δρ = {delta:.3f}")
    print(f"95% bootstrap CI for Δρ: [{lower:.3f}, {upper:.3f}]")
    print(f"Approximate two-tailed p-value: {p_val:.4f}")

    # 7) write stats.json for CI/CD tracking
    stats = {
        "version": os.environ.get("CI_COMMIT_TAG",
                   os.environ.get("CI_BUILD_NUMBER", "local")),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "spearman_weighted": w_corr,
        "spearman_combined": c_corr,
        "delta_rho": delta,
        "delta_ci_lower": lower,
        "delta_ci_upper": upper,
        "p_value": p_val
    }
    with open("stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("✅ stats.json written with correlation & significance info.")

if __name__ == "__main__":
    main()