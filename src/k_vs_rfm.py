import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


pd.options.display.max_columns = 100

# -------------------------------
# LOAD DATA
# -------------------------------
df_cluster = pd.read_parquet("data/clustered_users.parquet")
df_rfm = pd.read_parquet("data/rfm_segments.parquet")

# -------------------------------
# MERGE
# -------------------------------
df_compare = df_cluster.merge(
    df_rfm[["user_id", "rfm_segment"]],
    on="user_id",
    how="inner"
)

print("\nMerged shape:", df_compare.shape)

# -------------------------------
# CROSS TABULATION
# -------------------------------
cross_tab = pd.crosstab(
    df_compare["cluster_id"],
    df_compare["rfm_segment"],
    normalize="index"   # row-wise %
)

print("\nCluster vs RFM Distribution (%):")
print(cross_tab.round(2))

# -------------------------------
# HEATMAP (BEST VISUAL)
# -------------------------------
plt.figure(figsize=(12,6))
sns.heatmap(cross_tab, annot=True, cmap="coolwarm")

plt.title("Cluster vs RFM Segment Mapping")
plt.xlabel("RFM Segment")
plt.ylabel("Cluster ID")

plt.tight_layout()
plt.savefig("visuals/final_segments/cluster_vs_rfm.png")
plt.show()