import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from numpy.linalg import norm

# Загружаем очищенные данные
df = pd.read_csv("data/airbnb_clean.csv")

# 1. Гистограмма распределения цен (после очистки) - имитация R визуализации
plt.figure(figsize=(10, 6))
plt.hist(df['price'], bins=50, edgecolor='black')
plt.title('Распределение цены (после очистки)')
plt.xlabel('Цена за ночь')
plt.ylabel('Количество объявлений')
plt.tight_layout()
plt.savefig('images/price_distribution.png')
plt.close()

# 2. Boxplot по числовым признакам (после очистки) - имитация R визуализации
features = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 
           'calculated_host_listings_count', 'availability_365']
df_subset = df[features]
df_melted = df_subset.melt(var_name='feature', value_name='value')

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_melted, x='feature', y='value')
plt.xticks(rotation=45)
plt.title('Распределение числовых признаков (после очистки)')
plt.tight_layout()
plt.savefig('images/numeric_features_boxplot.png')
plt.close()

# 3. Сравнительная гистограмма до и после масштабирования
features = [
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]

X = df[features].copy()
X = X.dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["price"], kde=True, ax=axes[0])
axes[0].set_title("price до масштабирования")

sns.histplot(X_scaled_df["price"], kde=True, ax=axes[1])
axes[1].set_title("price после StandardScaler")

plt.tight_layout()
plt.savefig('images/price_scaling_comparison.png')
plt.close()

# 4. Подготовка данных для визуализации кластеров
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=["pc1", "pc2"])

# 5. K-means (custom) - имитация функции из R скрипта
def kmeans_custom(X, n_clusters, max_iter=100, random_state=42):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)

    indices = rng.choice(len(X), size=n_clusters, replace=False)
    centroids = X[indices]

    for _ in range(max_iter):
        dists = np.stack([norm(X - c, axis=1) for c in centroids], axis=1)
        labels = np.argmin(dists, axis=1)

        new_centroids = np.vstack([
            X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
            for k in range(n_clusters)
        ])

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids

labels_custom, centers_custom = kmeans_custom(X_pca, n_clusters=4)

plt.figure(figsize=(6, 5))
plt.scatter(pca_df["pc1"], pca_df["pc2"], c=labels_custom, s=5)
plt.title("K-means (custom)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig('images/kmeans_custom.png')
plt.close()

# 6. KMeans (sklearn)
km = KMeans(n_clusters=4, random_state=42)
labels_km = km.fit_predict(X_pca)

plt.figure(figsize=(6, 5))
plt.scatter(pca_df["pc1"], pca_df["pc2"], c=labels_km, s=5)
plt.title("KMeans (sklearn)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig('images/kmeans_sklearn.png')
plt.close()

# 7. Agglomerative Clustering
n_sample = min(8000, X_pca.shape[0])
rng = np.random.RandomState(42)
idx_sub = rng.choice(X_pca.shape[0], size=n_sample, replace=False)
X_pca_sub = X_pca[idx_sub]
agg = AgglomerativeClustering(n_clusters=4)
labels_agg_sub = agg.fit_predict(X_pca_sub)

plt.figure(figsize=(6, 5))
plt.scatter(X_pca_sub[:, 0], X_pca_sub[:, 1], c=labels_agg_sub, s=5)
plt.title("AgglomerativeClustering (subsample ~8000 точек)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig('images/agglomerative_clustering.png')
plt.close()

# 8. Gaussian Mixture
gm = GaussianMixture(n_components=4, random_state=42)
labels_gm = gm.fit_predict(X_pca)

plt.figure(figsize=(6, 5))
plt.scatter(pca_df["pc1"], pca_df["pc2"], c=labels_gm, s=5)
plt.title("GaussianMixture")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig('images/gaussian_mixture.png')
plt.close()

print("Все изображения успешно сгенерированы в директории images/")