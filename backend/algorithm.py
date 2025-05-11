import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1) Verinizi yükleyin
#    CSV’nizin yolunu kendi dosyanızın yoluyla değiştirin:
df = pd.read_csv("processed_data.csv")

# 2) Özellik matrisinizi oluşturun
X = df[["danceability", "energy", "valence", "tempo"]].values

# 3) Elbow Yöntemi (WCSS / inertia)
wcss = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    wcss.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K_range, wcss, 'o-', linewidth=2, markersize=6)
plt.xlabel("k değeri")
plt.ylabel("WCSS (inertia)")
plt.title("Elbow Yöntemi ile Optimal k")
plt.show()

# 4) Silhouette Analizi
sil_scores = []
K_range2 = range(2, 11)
for k in K_range2:
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    sil = silhouette_score(X, km.labels_)
    sil_scores.append(sil)

plt.figure(figsize=(6,4))
plt.plot(K_range2, sil_scores, 'o-', linewidth=2, markersize=6)
plt.xlabel("k değeri")
plt.ylabel("Ortalama Silhouette Skoru")
plt.title("Silhouette Analizi ile Optimal k")
plt.show()