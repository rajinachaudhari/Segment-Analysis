# ============================================================
# FULL PIPELINE: EDA → PREPROCESS → KMEANS → VISUALIZATION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

import joblib

pd.options.display.max_columns = 100

# ============================================================
# LOAD DATA (USE USER-LEVEL FEATURES, NOT TRANSACTIONS)
# ============================================================
df = pd.read_parquet("data/scaled_features.parquet")

# ------------------------------------------------------------
# Use only ACTIVE USERS
# ------------------------------------------------------------
df_model = df.copy().reset_index(drop=True)

# ============================================================
# FEATURE SELECTION (MATCH YOUR FEATURE ENGINEERING)
# ============================================================
CLUSTER_FEATURES = [
                     "total_txn_count",
                     "avg_txn_value",
                     "active_days",
                     "recency_days",
                     "active_ratio",
                     "txn_freq_per_month",
                     "txn_freq_std",
                     "spend_to_load_ratio",
                     "balance",
                     "failed_txn_rate",
                     "high_value_ratio",
                     "spend_trend",
                     "new_payee_rate",
                     "channel_AGENT_ratio",
                     "channel_APP_ratio",
                     "channel_USSD_ratio"  
                     ]

X = df_model[CLUSTER_FEATURES].copy()

# ============================================================
# -------------------- EDA SECTION -----------------------------
# ============================================================

print("\nRunning EDA...")

# -------------------------------
# 1. DISTRIBUTIONS
# -------------------------------
import math

n_features = len(CLUSTER_FEATURES)
n_cols = 4
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(CLUSTER_FEATURES):
    axes[i].hist(X[col].dropna(), bins=50, alpha=0.8)
    axes[i].set_title(col)

# Remove empty plots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Feature Distributions (Active Users)")
plt.tight_layout()
plt.savefig("visuals/k_results/01_distributions.png", dpi=150)
plt.show()

# -------------------------------
# 2. CORRELATION HEATMAP
# -------------------------------
corr = X.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("visuals/k_results/02_correlation.png", dpi=150)
plt.show()

print("\nfinished EDA...")
# ============================================================
# PREPROCESSING
# ============================================================

# Handle nulls
print(f"\nNull values: {X.isnull().sum().sum()}")
X = X.fillna(0)

# -------------------------------
# SCALING (ROBUST → OUTLIER SAFE)
# -------------------------------
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=CLUSTER_FEATURES)
print(f"\nScaled data summary: {X_scaled.describe()}")


joblib.dump(scaler, "data/scaler.pkl")

# ============================================================
# FIND OPTIMAL K
# ============================================================

print("\nSearching for optimal K...")

inertias = []
sil_scores = []
db_scores = []

K_RANGE = range(2, 11)

for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

    print(f"K={k} | Silhouette={sil_scores[-1]:.3f} | DB={db_scores[-1]:.3f}")

# -------------------------------
# PLOT ELBOW + SILHOUETTE
# -------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(K_RANGE, inertias, marker='o')
ax1.set_title("Elbow Method")

ax2.plot(K_RANGE, sil_scores, marker='o')
ax2.set_title("Silhouette Score")

plt.tight_layout()
plt.savefig("visuals/k_results/03_optimal_k.png", dpi=150)
plt.show()

# Choose K (adjust based on graph)
OPTIMAL_K = 5
print(f"\nChosen K: {OPTIMAL_K}")

# ============================================================
# FINAL KMEANS MODEL
# ============================================================

kmeans_final = KMeans(
    n_clusters=OPTIMAL_K,
    n_init=20,
    max_iter=500,
    random_state=42
)

df_model["cluster_id"] = kmeans_final.fit_predict(X_scaled)

joblib.dump(kmeans_final, "data/kmeans_model.pkl")

print("\nCluster Distribution:")
print(df_model["cluster_id"].value_counts())


# ============================================================
# CLUSTER PROFILE
# ============================================================

cluster_profile = df_model.groupby("cluster_id")[CLUSTER_FEATURES].mean()
cluster_profile["user_count"] = df_model.groupby("cluster_id")["user_id"].count()

print("\nCluster Profiles:")
print(cluster_profile.T)


# ============================================================
# PCA VISUALIZATION
# ============================================================

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
print(f"\nPCA variance: {explained}")

plt.figure(figsize=(10, 7))

for cluster in range(OPTIMAL_K):
    mask = df_model["cluster_id"] == cluster
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Cluster {cluster}",
        alpha=0.5
    )

# Centroids
centers = pca.transform(kmeans_final.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c="black", marker="X", s=200)

plt.title("Clusters (PCA View)")
plt.legend()
plt.tight_layout()
plt.savefig("visuals/k_results/04_pca_clusters.png", dpi=150)
plt.show()


# ============================================================
# SAVE OUTPUT
# ============================================================

df_model.to_parquet("data/clustered_users.parquet", index=False)

print("\nPipeline Complete ")








# # ============================================================
# # FULL PIPELINE: EDA → PREPROCESS → KMEANS → VISUALIZATION
# # (WITH DORMANT + RISKY USER DETECTION)
# # ============================================================

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.preprocessing import RobustScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score, davies_bouldin_score
# from sklearn.decomposition import PCA

# import joblib

# pd.options.display.max_columns = 100

# # ============================================================
# # LOAD DATA
# # ============================================================
# df = pd.read_parquet("data/scaled_features.parquet")

# # ============================================================
# # 🔥 IMPORTANT: USE ALL USERS (NOT ONLY ACTIVE)
# # ============================================================
# df_model = df.copy().reset_index(drop=True)

# # ============================================================
# # 🔥 FEATURE ENGINEERING FOR DORMANT + RISK DETECTION
# # ============================================================

# # -------------------------------
# # 1. Strengthen recency (Dormant detection)
# # -------------------------------
# df_model["recency_days"] = df_model["recency_days"].clip(0, 90)
# df_model["recency_log"] = np.log1p(df_model["recency_days"])

# # -------------------------------
# # 2. Risk / Fraud signals
# # -------------------------------
# df_model["risk_score"] = (
#     df_model["failed_txn_rate"] * 2 +
#     df_model["high_value_ratio"] * 1.5 +
#     df_model["txn_freq_std"]
# )

# df_model["txn_intensity"] = (
#     df_model["total_txn_count"] / (df_model["active_days"] + 1)
# )

# # ============================================================
# # FEATURE SELECTION
# # ============================================================
# CLUSTER_FEATURES = [
#     "recency_log",          # Dormant signal
#     "total_txn_count",
#     "total_spend_npr",
#     "avg_txn_value",
#     "active_ratio",
#     "txn_freq_per_month",
#     "txn_freq_std",
#     "spend_to_load_ratio",
#     "new_payee_rate",
#     "channel_APP_ratio",

#     # 🔥 NEW FEATURES
#     "risk_score",
#     "txn_intensity"
# ]

# X = df_model[CLUSTER_FEATURES].copy()

# # ============================================================
# # -------------------- EDA SECTION -----------------------------
# # ============================================================

# print("\nRunning EDA...")

# # -------------------------------
# # 1. DISTRIBUTIONS
# # -------------------------------
# fig, axes = plt.subplots(3, 4, figsize=(18, 10))
# axes = axes.flatten()

# for i, col in enumerate(CLUSTER_FEATURES):
#     axes[i].hist(X[col].dropna(), bins=50, alpha=0.8)
#     axes[i].set_title(col)

# plt.suptitle("Feature Distributions (All Users)")
# plt.tight_layout()
# plt.savefig("visuals/k_results/01_distributions.png", dpi=150)
# plt.show()

# # -------------------------------
# # 2. CORRELATION HEATMAP
# # -------------------------------
# corr = X.corr()

# plt.figure(figsize=(12, 10))
# sns.heatmap(corr, cmap="coolwarm", center=0)
# plt.title("Correlation Heatmap")
# plt.tight_layout()
# plt.savefig("visuals/k_results/02_correlation.png", dpi=150)
# plt.show()

# print("\nFinished EDA...")

# # ============================================================
# # PREPROCESSING
# # ============================================================

# print(f"\nNull values: {X.isnull().sum().sum()}")
# X = X.fillna(0)

# # -------------------------------
# # SCALING
# # -------------------------------
# scaler = RobustScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled = pd.DataFrame(X_scaled, columns=CLUSTER_FEATURES)

# joblib.dump(scaler, "data/scaler.pkl")

# # ============================================================
# # FIND OPTIMAL K
# # ============================================================

# print("\nSearching for optimal K...")

# inertias = []
# sil_scores = []
# db_scores = []

# K_RANGE = range(2, 11)

# for k in K_RANGE:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(X_scaled)

#     inertias.append(kmeans.inertia_)
#     sil_scores.append(silhouette_score(X_scaled, labels))
#     db_scores.append(davies_bouldin_score(X_scaled, labels))

#     print(f"K={k} | Silhouette={sil_scores[-1]:.3f} | DB={db_scores[-1]:.3f}")

# # -------------------------------
# # PLOT
# # -------------------------------
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ax1.plot(K_RANGE, inertias, marker='o')
# ax1.set_title("Elbow Method")

# ax2.plot(K_RANGE, sil_scores, marker='o')
# ax2.set_title("Silhouette Score")

# plt.tight_layout()
# plt.savefig("visuals/k_results/03_optimal_k.png", dpi=150)
# plt.show()

# # 🔥 Slightly higher K for richer segmentation
# OPTIMAL_K = 8
# print(f"\nChosen K: {OPTIMAL_K}")

# # ============================================================
# # FINAL KMEANS
# # ============================================================

# kmeans_final = KMeans(
#     n_clusters=OPTIMAL_K,
#     n_init=20,
#     max_iter=500,
#     random_state=42
# )

# df_model["cluster_id"] = kmeans_final.fit_predict(X_scaled)

# joblib.dump(kmeans_final, "data/kmeans_model.pkl")

# print("\nCluster Distribution:")
# print(df_model["cluster_id"].value_counts())

# # ============================================================
# # CLUSTER PROFILE
# # ============================================================

# cluster_profile = df_model.groupby("cluster_id")[CLUSTER_FEATURES].mean()
# cluster_profile["user_count"] = df_model.groupby("cluster_id")["user_id"].count()

# print("\nCluster Profiles:")
# print(cluster_profile.T)

# # ============================================================
# # 🔥 IDENTIFY DORMANT & RISKY CLUSTERS
# # ============================================================

# print("\n--- Cluster Interpretation Guide ---")

# print("\nDormant cluster = highest recency_log")
# print(cluster_profile["recency_log"].sort_values(ascending=False))

# print("\nRisky cluster = highest risk_score")
# print(cluster_profile["risk_score"].sort_values(ascending=False))

# # ============================================================
# # PCA VISUALIZATION
# # ============================================================

# pca = PCA(n_components=2, random_state=42)
# X_pca = pca.fit_transform(X_scaled)

# explained = pca.explained_variance_ratio_
# print(f"\nPCA variance: {explained}")

# plt.figure(figsize=(10, 7))

# for cluster in range(OPTIMAL_K):
#     mask = df_model["cluster_id"] == cluster
#     plt.scatter(
#         X_pca[mask, 0],
#         X_pca[mask, 1],
#         label=f"Cluster {cluster}",
#         alpha=0.5
#     )

# centers = pca.transform(kmeans_final.cluster_centers_)
# plt.scatter(centers[:, 0], centers[:, 1], c="black", marker="X", s=200)

# plt.title("Clusters (PCA View)")
# plt.legend()
# plt.tight_layout()
# plt.savefig("visuals/k_results/04_pca_clusters.png", dpi=150)
# plt.show()

# # ============================================================
# # SAVE
# # ============================================================

# df_model.to_parquet("data/clustered_users.parquet", index=False)

# print("\nPipeline Complete ✅")