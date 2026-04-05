import pandas as pd

pd.options.display.max_columns = 100

# -------------------------------
# LOAD CLUSTERED DATA
# -------------------------------
df = pd.read_feather("clustered_users.feather")
print(df.columns)


# -------------------------------
# FINAL SEGMENT MAPPING FUNCTION
# -------------------------------
def map_segment(cluster):

    # -------------------------------
    #  PREMIUM HIGH-VALUE USERS
    # -------------------------------
    # Clusters: 0, 8
    # Logic:
    # - High avg_spent and max_txn → strong monetary value
    # - High spend_to_load_ratio → efficient spenders
    # - High high_value_txn → premium transactions
    # - Low failed_txn → reliable users
    # Business meaning:
    # → Core revenue-generating customers
    if cluster in [0, 8]:
        return "Premium Users"

    # -------------------------------
    #  RISKY HIGH-VALUE USERS
    # -------------------------------
    # Cluster: 4
    # Logic:
    # - Very high spend and transaction size
    # - BUT extremely high failed_txn_rate
    # Business meaning:
    # → High-value but problematic users (fraud risk / payment friction)
    elif cluster == 4:
        return "Risky High-Value Users"

    # -------------------------------
    #  ENGAGED POWER USERS
    # -------------------------------
    # Cluster: 3
    # Logic:
    # - Very high active_days → frequent usage
    # - High feature_adoption → uses multiple features
    # - High service_diversity & merchants → explores ecosystem
    # Business meaning:
    # → Highly engaged users (platform champions)
    elif cluster == 3:
        return "Engaged Users"

    # -------------------------------
    #  MODERATE / REGULAR USERS
    # -------------------------------
    # Clusters: 1, 7
    # Logic:
    # - Around average across spend, activity, diversity
    # - No extreme behavior
    # Business meaning:
    # → Stable users with growth potential
    elif cluster in [1, 7]:
        return "Moderate Users"

    # -------------------------------
    #  FREQUENT LOW-VALUE USERS
    # -------------------------------
    # Cluster: 6
    # Logic:
    # - Very high txn frequency
    # - Low avg_spent and diversity
    # Business meaning:
    # → Many small transactions (e.g., recharges, micro-payments)
    elif cluster == 6:
        return "Frequent Low-Value Users"

    # -------------------------------
    #  LOW ENGAGEMENT / INACTIVE USERS
    # -------------------------------
    # Clusters: 2, 5, 9
    # Logic:
    # - Low spend, low activity, low diversity
    # - High recency → not recently active
    # Business meaning:
    # → Churn-risk or dormant users
    else:
        return "Inactive Users"


# -------------------------------
# APPLY SEGMENTATION
# -------------------------------
df["final_segment"] = df["user_cluster_identity"].apply(map_segment)
# -------------------------------
# SEGMENT DISTRIBUTION
# -------------------------------
print("\nSegment Distribution:")
print(df["final_segment"].value_counts())

# -------------------------------
# SEGMENT PROFILE (MEAN)
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

print("\nSegment Profile (Mean Values):")
print(df.groupby("final_segment")[kmeans_features].mean())

# -------------------------------
# OPTIONAL: NORMALIZED PROFILE (EASY TO READ)
# -------------------------------
profile = df.groupby("final_segment")[kmeans_features].mean()
profile_norm = (profile - profile.min()) / (profile.max() - profile.min())

print("\nNormalized Segment Profile (0–1 scale):")
print(profile_norm)

# -------------------------------
# SAVE FINAL SEGMENTED DATA
# -------------------------------
df.to_feather("final_segmented_users.feather")

print("\nFinal segmentation saved successfully!")