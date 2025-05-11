#!/usr/bin/env python3
import random
import numpy as np
import os
import subprocess
import sys
import json
from datetime import datetime
import hashlib

PY = sys.executable
BASE = os.path.dirname(__file__)
tasks = [
    "train_model.py",
    "train_classifier.py",
    "predict_all_sentiment.py",
    "test_sentiment_effect.py"
]

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def run(script):
    print(f"▶ Running {script}…")
    subprocess.run([PY, os.path.join(BASE, script)], check=True)

def update_history_and_plot():
    import pandas as pd
    import matplotlib.pyplot as plt

    # 1) stats.json'u oku
    with open(os.path.join(BASE, "stats.json")) as f:
        s = json.load(f)

    # 2) history.csv'ye ekle
    hist = os.path.join(BASE, "correlation_history.csv")
    if os.path.exists(hist):
        df = pd.read_csv(hist)
    else:
        df = pd.DataFrame(columns=["version","timestamp","spearman_weighted","spearman_combined"])
    df = df.append(s, ignore_index=True)
    df.to_csv(hist, index=False)

    # 3) grafik
    plt.figure()
    plt.plot(df["version"], df["spearman_combined"], marker="o", label="Combined ρ")
    plt.plot(df["version"], df["spearman_weighted"], marker="o", label="Weighted ρ")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "correlation.png"))

def main():
    set_global_seed(42)

    for script in tasks[:-1]:
        run(script)

   
    data = json.load(open(os.path.join(BASE, "data/feedback_data.json")))
    fp = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()[:8]
    os.environ["CI_COMMIT_TAG"] = fp

    run(tasks[-1])             
    update_history_and_plot()

if __name__ == "__main__":
    main()