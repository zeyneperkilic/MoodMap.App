#!/usr/bin/env python3
import sys
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from test_sentiment_effect import load_feedback    # use your original loader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from api import recommend_songs
from scipy.stats import spearmanr, ttest_rel, wilcoxon

def main():
    # --- Load stats.json for original plots ---
    stats_fn = "stats.json"
    if not os.path.exists(stats_fn):
        print(f"⚠️ File {stats_fn} not found.")
        return
    with open(stats_fn) as f:
        stats = json.load(f)
    version = stats.get("version", "unknown")

    # --- Attempt to load feedback for new plots ---
    fb_path = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        feedback = load_feedback(fb_path)
        analyzer = SentimentIntensityAnalyzer()
        weighted_scores, combined_scores, ground_truths = [], [], []
        clusters, intensities = [], []
        for uri, entry in feedback.items():
            comments = entry.get("comments", [])
            if not comments:
                continue
            gt = np.mean([analyzer.polarity_scores(c)["compound"] for c in comments])
            cluster = int(entry.get("cluster_id", -1))
            intensity = int(entry.get("intensity", 5))

            recs = recommend_songs(cluster, intensity).get("songs", [])
            for r in recs:
                if r.get("uri") == uri:
                    cs = r.get("combined_score", 0)
                    cluster_key = f"spearman_combined_cluster{cluster}"
                    cluster_weight = stats.get(cluster_key, stats.get("spearman_combined", 1.0))
                    ws = cs * cluster_weight
                    weighted_scores.append(ws)
                    combined_scores.append(cs)
                    ground_truths.append(gt)
                    clusters.append(cluster)
                    intensities.append(intensity)
                    break
        feedback_available = True
    except FileNotFoundError as e:
        print(f"⚠️ {e}. Skipping feedback-based plots.")
        feedback_available = False

    # --- 1.a) Global trend (Weighted vs Combined) + CI ---
    w = stats["spearman_weighted"]
    c = stats["spearman_combined"]
    lo, hi = stats["delta_ci"]
    plt.figure(figsize=(4,4))
    plt.bar(["Weighted","Combined"], [w, c], alpha=0.7)
    plt.errorbar(1, c, yerr=[[abs(c-lo)],[abs(hi-c)]], fmt='none', ecolor='black', capsize=5)
    plt.ylim(-1,1)
    plt.title(f"Global Spearman (v={version})")
    plt.ylabel("ρ")
    plt.tight_layout()
    plt.savefig("plot_global_spearman_ci.png")
    print("✅ plot_global_spearman_ci.png")

    # --- 1.b) Cluster-based Combined trend ---
    cluster_keys = sorted(k for k in stats if k.startswith("spearman_combined_cluster"))
    if cluster_keys:
        clusters_c = [int(k.split("cluster")[-1]) for k in cluster_keys]
        vals_c = [stats[k] for k in cluster_keys]
        plt.figure(figsize=(6,4))
        plt.plot(clusters_c, vals_c, marker='o')
        plt.xticks(clusters_c)
        plt.ylim(-1,1)
        plt.title(f"Spearman Combined by Cluster (v={version})")
        plt.xlabel("Cluster ID")
        plt.ylabel("ρ")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("plot_cluster_spearman.png")
        print("✅ plot_cluster_spearman.png")

    # --- 1.c) Intensity-based Combined trend ---
    int_keys = sorted((int(k.split("_")[-1]), k)
                      for k in stats
                      if k.startswith("spearman_combined_") and k.split("_")[-1].isdigit())
    if int_keys:
        ints = [i for i, _ in int_keys]
        vals_i = [stats[k] for _, k in int_keys]
        plt.figure(figsize=(6,4))
        plt.plot(ints, vals_i, marker='o')
        plt.xticks(ints)
        plt.ylim(-1,1)
        plt.title(f"Spearman Combined by Intensity (v={version})")
        plt.xlabel("Intensity")
        plt.ylabel("ρ")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("plot_intensity_spearman.png")
        print("✅ plot_intensity_spearman.png")

    if feedback_available:
        # Compute deltas
        diffs_all = [c_val - w_val for c_val, w_val in zip(combined_scores, weighted_scores)]
        n = len(diffs_all)
        # Genel Hipotez Testleri
        t_stat, t_p = ttest_rel(combined_scores, weighted_scores)
        w_stat, w_p = wilcoxon(combined_scores, weighted_scores)
        print(f"Genel Paired t-test: t={t_stat:.3f}, p={t_p:.3f}, n={n}")
        print(f"Genel Wilcoxon testi: W={w_stat:.0f}, p={w_p:.3f}, n={n}")

        # Küme bazlı testler
        from collections import defaultdict
        delta_by_cluster = defaultdict(list)
        for cl, diff in zip(clusters, diffs_all):
            delta_by_cluster[cl].append(diff)
        for cl, diffs in delta_by_cluster.items():
            n_cl = len(diffs)
            t_cl, p_cl = ttest_rel([combined_scores[i] for i in range(len(diffs_all)) if clusters[i]==cl],
                                  [weighted_scores[i] for i in range(len(diffs_all)) if clusters[i]==cl])
            w_cl, p_w = wilcoxon([combined_scores[i] for i in range(len(diffs_all)) if clusters[i]==cl],
                                  [weighted_scores[i] for i in range(len(diffs_all)) if clusters[i]==cl])
            print(f"Cluster {cl} testi: t={t_cl:.3f}, p={p_cl:.3f}, W={w_cl:.0f}, p_w={p_w:.3f}, n={n_cl}")

        # --- 1.d) Boxplot of Δ per Intensity ---
        delta_by_intensity = defaultdict(list)
        for it, diff in zip(intensities, diffs_all):
            delta_by_intensity[it].append(diff)
        levels = sorted(delta_by_intensity)
        data = [delta_by_intensity[l] for l in levels]
        plt.figure(figsize=(6,4))
        plt.boxplot(data, tick_labels=levels, showfliers=False)
        plt.axhline(0, linestyle='--')
        plt.xlabel("Intensity")
        plt.ylabel("Δ Score (combined − weighted)")
        plt.title("Intensity Bazlı Skor Farkı Dağılımı")
        plt.tight_layout()
        plt.savefig("box_delta_by_intensity.png")
        print("✅ box_delta_by_intensity.png")

        # --- 1.e) Bland–Altman Plot ---
        means = [(w_val + c_val)/2 for w_val, c_val in zip(weighted_scores, combined_scores)]
        md = np.mean(diffs_all)
        sd = np.std(diffs_all, ddof=1)
        plt.figure(figsize=(6,4))
        plt.scatter(means, diffs_all, alpha=0.5)
        plt.axhline(md, linestyle='--')
        plt.axhline(md + 1.96*sd, linestyle=':')
        plt.axhline(md - 1.96*sd, linestyle=':')
        plt.xlabel("Mean of Scores")
        plt.ylabel("Difference (combined − weighted)")
        plt.title("Bland–Altman Plot")
        plt.tight_layout()
        plt.savefig("bland_altman.png")
        print("✅ bland_altman.png")

if __name__ == "__main__":
    main()
