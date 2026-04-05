import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

pd.options.display.max_columns = 100

# -------------------------------
# LOAD FINAL SEGMENTED DATA
# -------------------------------
# This file already contains:
# - features
# - user_cluster_identity
# - final_segment
df = pd.read_feather("final_segmented_users.feather")

print(df.columns)


# -------------------------------
# FEATURES USED FOR ANALYSIS
# -------------------------------
# These are the behavioral features that define user differences
features = [
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


# =========================================================
# SEGMENT DISTRIBUTION (WHO ARE YOUR USERS?)
# =========================================================
# WHY THIS MATTERS:
# - Shows size of each segment
# - Helps business understand where most users belong
# - Important for prioritization strategy

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="final_segment",
    order=df["final_segment"].value_counts().index
)

plt.title("Customer Segment Distribution")
plt.xlabel("Segment")
plt.ylabel("Number of Users")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("visuals/final visuals/segment_distribution.png")
plt.show()


# =========================================================
# SEGMENT PROFILE HEATMAP 
# =========================================================
# WHY THIS MATTERS:
# - Clearly shows behavioral differences between segments
# - Easy to explain in interviews
# - Highlights what makes each segment unique

profile = df.groupby("final_segment")[features].mean()

# Normalize values (0–1 scale for fair comparison)
profile_norm = (profile - profile.min()) / (profile.max() - profile.min())

plt.figure(figsize=(12, 6))

sns.heatmap(
    profile_norm,
    cmap="coolwarm",
    annot=False
)

plt.title("Segment Behavior Heatmap (Normalized)")
plt.xlabel("Features")
plt.ylabel("Segments")
plt.tight_layout()

plt.savefig("visuals/final visuals/segment_heatmap.png")
plt.show()


# =========================================================
# PCA VISUALIZATION (SEGMENT SEPARATION)
# =========================================================
# WHY THIS MATTERS:
# - Reduces data into 2D for visualization
# - Shows how well segments are separated
# - Validates clustering quality visually

pca = PCA(n_components=2)
X_pca = pca.fit_transform(df[features])

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=df["final_segment"],
    s=20
)

plt.title("Customer Segments (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()

plt.savefig("visuals/final visuals/pca_segments.png")
plt.show()


# =========================================================
# KEY FEATURE COMPARISON (BOXPLOTS)
# =========================================================
# WHY THIS MATTERS:
# - Shows distribution (not just average)
# - Helps explain variability inside segments
# - Great for storytelling (real behavior differences)

important_features = [
    "avg_spent",
    "txn_freq_per_month",
    "recency_days"
]

for col in important_features:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="final_segment",
        y=col
    )

    plt.title(f"{col} by Segment")
    plt.xlabel("Segment")
    plt.ylabel(col)
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(f"visuals/final visuals/boxplot_{col}.png")
    plt.show()


# =========================================================
# SEGMENT MEAN COMPARISON (BAR CHART)
# =========================================================
# WHY THIS MATTERS:
# - Easy comparison across segments
# - Business-friendly visualization
# - Useful for presentations

profile.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Average Feature Values by Segment")
plt.xlabel("Segment")
plt.ylabel("Mean Value")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("visuals/final visuals/segment_bar_chart.png")
plt.show()


# =========================================================
#SEGMENT PERCENTAGE (BUSINESS VIEW)
# =========================================================
# WHY THIS MATTERS:
# - Shows proportion of each segment
# - Helps in business decision making

segment_ratio = df["final_segment"].value_counts(normalize=True) * 100

print("\nSegment Percentage (%):")
print(segment_ratio.round(2))


# -------------------------------
# DONE
# -------------------------------
print("\nAll visualizations generated successfully!")