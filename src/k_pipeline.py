import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

pd.options.display.max_columns = 100

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_feather("data/scaled_features.feather")


# -------------------------------
# FEATURE SELECTION FOR CLUSTERING
# -------------------------------
kmeans_features = [
    "avg_spent",
    "max_txn",
    "active_days",
    "feature_adoption",
    "service_diversity",
    "unique_merchants",
    "recency_days",
    "txn_freq_per_month",
    "txn_time_spread",
    "spend_to_load_ratio",
    "has_failed_txn",
    "has_high_value_txn"
]

X = df[kmeans_features]

# # -------------------------------
# # FIND OPTIMAL K
# # -------------------------------
# inertia = []
# sil_scores = []

# K_range = range(2, 20)

# for k in K_range:
#     model = KMeans(n_clusters=k, random_state=42, n_init=10)
#     model.fit(X)

#     inertia.append(model.inertia_)
#     sil_scores.append(silhouette_score(X, model.labels_))

#     print(f"K={k}, Silhouette={sil_scores[-1]:.4f}")

# # -------------------------------
# # ELBOW PLOT
# # -------------------------------
# plt.figure()
# plt.plot(K_range, inertia, marker="o")
# plt.xlabel("K")
# plt.ylabel("Inertia")
# plt.title("Elbow Method")
# plt.savefig("visuals/k_results/elbow.png")
# plt.show()

# # -------------------------------
# # SILHOUETTE PLOT
# # -------------------------------
# plt.figure()
# plt.plot(K_range, sil_scores, marker="o")
# plt.xlabel("K")
# plt.ylabel("Silhouette Score")
# plt.title("Silhouette Score")
# plt.savefig("visuals/k_results/silhouette.png")
# plt.show()

# # -------------------------------
# # SELECT BEST K
# # -------------------------------
# best_k = K_range[sil_scores.index(max(sil_scores))]
# print(f"\nBest K: {best_k}")

# # -------------------------------
# # FINAL MODEL
# # -------------------------------
# final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
# df["cluster"] = final_model.fit_predict(X)


# -------------------------------
# hardcoding multiple k values as profiling>k selected
# -------------------------------
K_values = [3, 4, 5]
results = {}

for k in K_values:
    print(f"\n{'='*40}")
    print(f"Running KMeans with K = {k}")
    print(f"{'='*40}")

    # Train model
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    # Save cluster column
    df[f"cluster_{k}"] = labels

    # Silhouette score
    score = silhouette_score(X, labels)
    print(f"Silhouette Score: {score:.4f}")

    # Distribution
    print("\nCluster Distribution:")
    print(df[f"cluster_{k}"].value_counts())

    # Profile
    profile = df.groupby(f"cluster_{k}")[kmeans_features].mean()
    print("\nCluster Profile:")
    print(profile)

    results[k] = {"score": score, "profile": profile}

    # -------------------------------
    # PCA VISUALIZATION (clusterwise)
    # -------------------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure()
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=labels,
        s=20
    )
    plt.title(f"Clusters (PCA) - K={k}")
    plt.savefig(f"visuals/k_results/pca_k_{k}.png")
    plt.show()

print("\nDone!")
# # -------------------------------
# # PCA FOR VISUALIZATION
# # -------------------------------
# from sklearn.decomposition import PCA
# import seaborn as sns

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# plt.figure()
# sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df["cluster"], s=20)
# plt.title("Clusters (PCA)")
# plt.savefig("visuals/k_results/pca.png")
# plt.show()


# -------------------------------
# SAVE OUTPUT
# -------------------------------
df.to_feather("data/clustered_users.feather")

# -------------------------------
# CLUSTER DISTRIBUTION
# -------------------------------
print("\nCluster Counts:")
print(df["cluster"].value_counts())

# -------------------------------
# FINAL SCORE
# -------------------------------
score = silhouette_score(X, df["cluster"])
print(f"\nFinal Silhouette Score: {score:.4f}")