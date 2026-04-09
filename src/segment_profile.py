# ============================================================
# SEGMENT PROFILING & DASHBOARD VISUALIZATION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from pandas.plotting import parallel_coordinates
import os

pd.options.display.max_columns = 100

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet("data/clustered_users.parquet")

print("\nDataset Loaded. Shape:", df.shape)

# ============================================================
# CLUSTER → BUSINESS SEGMENT MAPPING
# ============================================================

CLUSTER_NAMES = {
    0: "Regular Users",
    1: "Rising Stars",
    2: "Casual Users",
    3: "Churn Users",
    4: "Premium / Risk Users"
}

df["segment_name"] = df["cluster_id"].map(CLUSTER_NAMES)

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "visuals/final_segments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FEATURES (MUST MATCH K-MEANS)
# ============================================================

FEATURES = [
    "total_txn_count",
    "avg_txn_value",
    "active_days",
    "recency_days",
    "account_age_days",
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

# ============================================================
# SEGMENT DISTRIBUTION
# ============================================================

print("\nSegment Distribution:")
print(df["segment_name"].value_counts())

plt.figure(figsize=(8,5))
sns.countplot(
    data=df,
    x="segment_name",
    order=df["segment_name"].value_counts().index
)
plt.title("Customer Segment Distribution")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_segment_distribution.png")
plt.show()

# ============================================================
# SEGMENT PROFILE
# ============================================================

profile = df.groupby("segment_name")[FEATURES].mean()

print("\nSegment Profile (Mean):")
print(profile)

# Normalized (0-1 scaling for comparison)
profile_norm = (profile - profile.min()) / (profile.max() - profile.min())

# ============================================================
# HEATMAP (CORE VISUAL)
# ============================================================

plt.figure(figsize=(12,6))
sns.heatmap(profile_norm, cmap="coolwarm", annot=False)

plt.title("Segment Behavior Heatmap")
plt.xlabel("Features")
plt.ylabel("Segments")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_heatmap.png")
plt.show()

# ============================================================
# DASHBOARD BOXPLOTS (MULTI-VIEW)
# ============================================================

important_features = [
    "total_txn_count",
    "avg_txn_value",
    "txn_freq_per_month",
    "recency_days",
    "spend_to_load_ratio"
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(important_features):
    sns.boxplot(data=df, x="segment_name", y=col, ax=axes[i])
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', rotation=30)

axes[-1].axis("off")

plt.suptitle("Customer Behavior Dashboard", fontsize=16)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_dashboard_boxplots.png")
plt.show()

# ============================================================
# STACKED BAR (CHANNEL USAGE)
# ============================================================

channel_cols = [
    "channel_AGENT_ratio",
    "channel_APP_ratio",
    "channel_USSD_ratio"
]

channel_profile = df.groupby("segment_name")[channel_cols].mean()

channel_profile.plot(
    kind="bar",
    stacked=True,
    figsize=(10,6)
)

plt.title("Channel Usage by Segment")
plt.ylabel("Ratio")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_channel_usage.png")
plt.show()

# ============================================================
# RADAR CHART (ADVANCED)
# ============================================================

features = profile_norm.columns.tolist()
angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)

plt.figure(figsize=(8,8))

for segment in profile_norm.index:
    values = profile_norm.loc[segment].values
    values = np.append(values, values[0])
    angle_loop = np.append(angles, angles[0])

    plt.polar(angle_loop, values, label=segment)

plt.xticks(angles, features, rotation=45)
plt.title("Segment Behavior Radar")
plt.legend(bbox_to_anchor=(1.3,1))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_radar_chart.png")
plt.show()

# ============================================================
# PCA VISUALIZATION
# ============================================================

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(df[FEATURES])

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=df["segment_name"],
    s=20
)

plt.title("Customer Segments (PCA Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.legend(bbox_to_anchor=(1.05,1))
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_pca_segments.png")
plt.show()

# ============================================================
# PARALLEL COORDINATES (ADVANCED)
# ============================================================

plot_df = profile_norm.reset_index()
plot_df.rename(columns={"segment_name": "segment"}, inplace=True)

plt.figure(figsize=(12,6))
parallel_coordinates(plot_df, "segment")

plt.title("Segment Comparison (Parallel Coordinates)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_parallel_plot.png")
plt.show()

# ============================================================
# SEGMENT BAR COMPARISON
# ============================================================

profile.plot(kind="bar", figsize=(12,6))

plt.title("Average Feature Values by Segment")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_segment_bar.png")
plt.show()

# ============================================================
# SAVE FINAL DATA
# ============================================================

df.to_parquet("data/final_segmented_users.parquet", index=False)

# ============================================================
# SEGMENT PERCENTAGE
# ============================================================

segment_ratio = df["segment_name"].value_counts(normalize=True) * 100

print("\nSegment Percentage (%):")
print(segment_ratio.round(2))

# ============================================================
# DONE
# ============================================================

print("\n Segmentation Analysis Complete!")