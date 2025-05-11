import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json

# 1) Ham verinizi yükleyin
df = pd.read_csv("processed_data.csv")  

# 2) Kullanacağınız özellikler
features = ["danceability", "energy", "valence", "tempo"]
X_raw    = df[features].values

# 3) Özellikleri standardize edin
scaler   = StandardScaler()
X        = scaler.fit_transform(X_raw)

# 4) PCA(n_components=1) ile birinci ana bileşeni çıkarın
pca      = PCA(n_components=1)
pca.fit(X)

# 5) Birinci bileşenin (PC1) katsayıları—yani loadings—alın
loadings = np.abs(pca.components_[0])

# 6) Mutlak değerleri normalize edip toplam 1’e çevirin
weights  = loadings / loadings.sum()

# 7) Sonuçları sözlüğe dönüştürün
w_dict = dict(zip(features, weights.round(2)))
print("PCA-based normalized weights:", w_dict)

# 8) JSON dosyasına yazdırın
with open("pca_weights.json", "w", encoding="utf-8") as f:
    json.dump(w_dict, f, ensure_ascii=False, indent=2)

print("✅ pca_weights.json dosyası oluşturuldu.")