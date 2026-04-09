import pandas as pd
import numpy as np

pd.options.display.max_columns = 100

# -----------------------------------------------
# LOAD YOUR ENGINEERED USER DATA
# -----------------------------------------------
user_df = pd.read_parquet("data/scaled_features.parquet")

# NOTE: Use ORIGINAL (non-scaled) values for RFM
# So reload pre-scaled data
user_raw = pd.read_parquet("data/clean_fintech_data.parquet")

# Recompute minimal required RFM fields from your pipeline
df_feat = user_raw.copy()

df_feat["txn_day"] = df_feat["created_at"].dt.date
grp = df_feat.groupby("user_id")

latest_date = df_feat["created_at"].max()

rfm_df = grp.agg(
    total_txn_count=("txn_id", "count"),
    total_spend_npr=("amount", "sum"),
    last_txn=("created_at", "max")
).reset_index()

rfm_df["recency_days"] = (latest_date - rfm_df["last_txn"]).dt.days

# Active flag (same logic as yours)
rfm_df["is_active"] = (rfm_df["recency_days"] <= 30).astype(int)

# -----------------------------------------------
# FILTER ACTIVE USERS
# -----------------------------------------------
df_rfm = rfm_df[rfm_df["is_active"] == 1].copy()

# -----------------------------------------------
# RFM SCORING
# -----------------------------------------------

# Recency (LOW = better → reverse scoring)
df_rfm['R'] = pd.qcut(
    df_rfm['recency_days'],
    q=5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

# Frequency
df_rfm['F'] = pd.qcut(
    df_rfm['total_txn_count'].rank(method='first'),
    q=5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

# Monetary
df_rfm['M'] = pd.qcut(
    df_rfm['total_spend_npr'].rank(method='first'),
    q=5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

# -----------------------------------------------
# COMBINE RFM
# -----------------------------------------------
df_rfm['rfm_score'] = df_rfm['R'] + df_rfm['F'] + df_rfm['M']
df_rfm['rfm_label'] = (
    df_rfm['R'].astype(str) +
    df_rfm['F'].astype(str) +
    df_rfm['M'].astype(str)
)

# -----------------------------------------------
# SEGMENT MAPPING
# -----------------------------------------------
def rfm_segment(row):
    r, f, m = row['R'], row['F'], row['M']
    score = row['rfm_score']

    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 4 and f >= 3:
        return 'Loyal Users'
    elif r >= 4 and m >= 4:
        return 'Big Spenders'
    elif r >= 3 and f >= 3:
        return 'Promising'
    elif r >= 4 and f <= 2:
        return 'New Users'
    elif r <= 2 and f >= 4:
        return 'At Risk'
    elif r <= 2 and m >= 4:
        return 'Cant Lose Them'
    elif score >= 8:
        return 'Need Attention'
    elif r <= 2:
        return 'Hibernating'
    else:
        return 'About To Sleep'

df_rfm['rfm_segment'] = df_rfm.apply(rfm_segment, axis=1)

# -----------------------------------------------
# HANDLE DORMANT USERS
# -----------------------------------------------
df_dormant = rfm_df[rfm_df["is_active"] == 0].copy()
df_dormant['rfm_segment'] = 'Dormant'
df_dormant['rfm_score'] = 0

# -----------------------------------------------
# FINAL DATASET
# -----------------------------------------------
df_scored = pd.concat([df_rfm, df_dormant], ignore_index=True)

# -----------------------------------------------
# SUMMARY
# -----------------------------------------------
segment_summary = df_scored.groupby('rfm_segment').agg(
    users=('user_id', 'count'),
    avg_recency=('recency_days', 'mean'),
    avg_txn_count=('total_txn_count', 'mean'),
    avg_spend_npr=('total_spend_npr', 'mean'),
    total_revenue=('total_spend_npr', 'sum')
).round(2).sort_values('total_revenue', ascending=False)

print(segment_summary)

# -----------------------------------------------
# SAVE
# -----------------------------------------------
df_scored.to_parquet("data/rfm_segments.parquet", index=False)

