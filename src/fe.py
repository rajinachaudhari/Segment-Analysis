import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

pd.options.display.max_columns = 100

# -----------------------------------------------
# LOAD DATA
# -----------------------------------------------
df = pd.read_parquet("data/clean_fintech_data.parquet")
df_feat = df.copy()

# -----------------------------------------------
# TIME FEATURES
# -----------------------------------------------
df_feat["month"] = df_feat["created_at"].dt.to_period("M")
df_feat["txn_hour"] = df_feat["created_at"].dt.hour
df_feat["txn_day"] = df_feat["created_at"].dt.date

# -----------------------------------------------
# FLAGS
# -----------------------------------------------
df_feat["is_topup"] = (df_feat["txn_type"] == "TOPUP").astype(int)
df_feat["is_high_txn"] = (df_feat["amount"] > 10000).astype(int)

# -----------------------------------------------
# GROUPING
# -----------------------------------------------
grp = df_feat.groupby("user_id")
grp_month = df_feat.groupby(["user_id", "month"])

# -----------------------------------------------
# BASIC AGGREGATIONS 
# -----------------------------------------------
user_df = grp.agg(
    total_txn_count=("txn_id", "count"),
    total_spend_npr=("amount", "sum"),
    avg_txn_value=("amount", "mean"),
    max_txn=("amount", "max"),
    active_days=("txn_day", "nunique"),
    txn_types=("txn_type", "nunique"),
    channels=("channel", "nunique"),
).reset_index()

# -----------------------------------------------
# RECENCY & ACCOUNT AGE
# -----------------------------------------------
latest_date = df_feat["created_at"].max()

last_txn = grp["created_at"].max()
first_reg = grp["registration_date"].first()

user_df["recency_days"] = (latest_date - user_df["user_id"].map(last_txn)).dt.days
user_df["account_age_days"] = (latest_date - user_df["user_id"].map(first_reg)).dt.days

# -----------------------------------------------
# ACTIVE RATIO (SAFE)
# -----------------------------------------------
user_df["active_ratio"] = np.where(
    user_df["account_age_days"] > 0,
    user_df["active_days"] / user_df["account_age_days"],
    0
)

# -----------------------------------------------
# MONTHLY FEATURES
# -----------------------------------------------
monthly_txn = grp_month.size()

user_df["txn_freq_per_month"] = monthly_txn.groupby("user_id").mean().values
user_df["txn_freq_std"] = monthly_txn.groupby("user_id").std().fillna(0).values

# -----------------------------------------------
# FINANCIAL FEATURES (OPTIMIZED)
# -----------------------------------------------
topup_df = df_feat[df_feat["is_topup"] == 1]
spend_df = df_feat[df_feat["is_topup"] == 0]

load = topup_df.groupby("user_id")["amount"].sum()
spend = spend_df.groupby("user_id")["amount"].sum()

user_df["load"] = user_df["user_id"].map(load).fillna(0)
user_df["spend"] = user_df["user_id"].map(spend).fillna(0)

user_df["spend_to_load_ratio"] = np.where(
    user_df["load"] > 0,
    user_df["spend"] / user_df["load"],
    0
)

# Wallet features
wallet_cols = grp[["balance", "total_loaded_npr", "total_spent_npr"]].first()
user_df = user_df.merge(wallet_cols, on="user_id", how="left")

# -----------------------------------------------
# RISK FEATURES (FAST)
# -----------------------------------------------
status_counts = df_feat.groupby(["user_id", "status"]).size().unstack(fill_value=0)

failed = status_counts.get("FAILED", 0)
total = status_counts.sum(axis=1)

user_df["failed_txn_rate"] = user_df["user_id"].map(failed / total).fillna(0)
user_df["high_value_ratio"] = grp["is_high_txn"].mean().values

# -----------------------------------------------
# TREND FEATURE
# -----------------------------------------------
monthly_spend = grp_month["amount"].sum()

def calc_trend(x):
    if len(x) > 1:
        return np.polyfit(range(len(x)), x, 1)[0]
    return 0

trend = monthly_spend.groupby("user_id").apply(calc_trend)
user_df["spend_trend"] = trend.values

# -----------------------------------------------
# NEW PAYEE RATE (OPTIMIZED)
# -----------------------------------------------
payee_counts = df_feat.groupby("user_id")["receiver_user_id"].nunique()
txn_counts = df_feat.groupby("user_id")["txn_id"].count()

user_df["new_payee_rate"] = (
    user_df["user_id"].map(payee_counts) /
    user_df["user_id"].map(txn_counts)
).fillna(0)

# -----------------------------------------------
# CHANNEL RATIOS
# -----------------------------------------------
channel_ratio = df_feat.pivot_table(
    index="user_id",
    columns="channel",
    values="txn_id",
    aggfunc="count",
    fill_value=0
)

channel_ratio = channel_ratio.div(channel_ratio.sum(axis=1), axis=0)
channel_ratio.columns = [f"channel_{c}_ratio" for c in channel_ratio.columns]

user_df = user_df.merge(channel_ratio, on="user_id", how="left")

# -----------------------------------------------
# IS ACTIVE FLAG (for RFM)
# -----------------------------------------------
user_df["is_active"] = (user_df["recency_days"] <= 30).astype(int)

# -----------------------------------------------
# CLEANING (CRITICAL FIXES)
# -----------------------------------------------
user_df = user_df.replace([np.inf, -np.inf], np.nan)

# Cap extreme values
user_df["spend_to_load_ratio"] = user_df["spend_to_load_ratio"].clip(0, 10)

# Fill NaNs
user_df = user_df.fillna(0)

# -----------------------------------------------
# DROP USELESS FEATURES
# -----------------------------------------------
user_df.drop([
    "txn_types",
    "channels",
    "load",
    "spend"
], axis=1, inplace=True)

# -----------------------------------------------
# LOG TRANSFORM
# -----------------------------------------------
log_cols = [
    "total_spend_npr",
    "avg_txn_value",
    "max_txn",
    "txn_freq_per_month",
    "recency_days",
    "spend_to_load_ratio",
    "total_txn_count"
]

for col in log_cols:
    user_df[col] = np.log1p(user_df[col])

# -----------------------------------------------
# BINARY FLAGS
# -----------------------------------------------
user_df["has_failed_txn"] = (user_df["failed_txn_rate"] > 0).astype(int)
user_df["has_high_value_txn"] = (user_df["high_value_ratio"] > 0).astype(int)

# Clip
user_df["failed_txn_rate"] = user_df["failed_txn_rate"].clip(0, 0.5)
user_df["high_value_ratio"] = user_df["high_value_ratio"].clip(0, 0.8)

# -----------------------------------------------
# SCALING
# -----------------------------------------------

binary_cols = ["is_active", "has_failed_txn", "has_high_value_txn"]

X = user_df.drop("user_id", axis=1)

X_num = X.drop(columns=binary_cols)
X_bin = X[binary_cols]

scaler = StandardScaler()
X_scaled_num = scaler.fit_transform(X_num)

X_scaled_df = pd.DataFrame(X_scaled_num, columns=X_num.columns)

# Add binary features back WITHOUT scaling
X_scaled_df[binary_cols] = X_bin.reset_index(drop=True)
X_scaled_df["user_id"] = user_df["user_id"].values
cols = ["user_id"] + [c for c in X_scaled_df.columns if c != "user_id"]
X_scaled_df = X_scaled_df[cols]


# Memory optimization
for col in X_scaled_df.select_dtypes(include="float64").columns:
    X_scaled_df[col] = X_scaled_df[col].astype("float32")

# -----------------------------------------------
# SAVE
# -----------------------------------------------
X_scaled_df.to_parquet("data/scaled_features.parquet", index=False)
X_scaled_df.to_csv("scaled_features.csv", index=False)


# -----------------------------------------------
# VALIDATION
# -----------------------------------------------
print("Shape:", X_scaled_df.shape)
print("Means:\n", X_scaled_df.select_dtypes(include='number').mean())
print("Std:\n", X_scaled_df.select_dtypes(include='number').std())