# import pandas as pd 
# import numpy as np

# pd.options.display.max_columns = 100

# df = pd.read_csv("data/clean_fintech_data.csv")

# # re-converting date columns as pandas csv read them as object
# df["txn_date"] = pd.to_datetime(df["txn_date"])
# df["account_open_date"] = pd.to_datetime(df["account_open_date"])
# df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])


# #-----------------------------------------------
# #feature creation
# #----------------------------------------------

# # creating customer-level features
# #i.e c1 spent total x amt,average y amt, their max transaction amt was z and they made n transactions in total.
# customer_df = df.groupby("customer_id").agg(
#     total_spent=("amount_npr", "sum"),
#     avg_spent=("amount_npr", "mean"),  #average transaction value per customer
#     max_txn=("amount_npr", "max"),
#     txn_count=("txn_id", "count")
# ).reset_index()

# # 2. Calculate average transactions per month per customer
# txn_freq = (
#     df.groupby(["customer_id", df["txn_date"].dt.to_period("M")])
#     .size()
#     .groupby("customer_id")
#     .mean()
#     .reset_index(name="txn_freq_per_month")
# )
# #this shows their transaction behavior  

# #===========================================

# #how frequent customer use this service
# latest_date = df["txn_date"].max()
# recency = df.groupby("customer_id")["txn_date"].max().reset_index()
# recency["recency_days"] = (latest_date - recency["txn_date"]).dt.days
# #less the recency_days more frequent the customer is.

# #How many different days the customer transacted i.e their active days
# active_days = df.groupby("customer_id")["txn_date"].nunique().reset_index(name="active_days")

# #How many products the customer uses.
# feature_adoption = df.groupby("customer_id")["product_name"].nunique().reset_index(name="feature_adoption")

# #this shows their engagement behavior

# #===========================================
# #what is the rate of failed transactions for each customer
# failed_txn_rate = df.groupby("customer_id")["status"].apply(
#     lambda x: (x == "failed").mean()
# ).reset_index(name="failed_txn_rate")

# #transaction inconsistency: how much variation in transaction amount for each customer
# txn_std = (
#     df.groupby(["customer_id", df["txn_date"].dt.to_period("M")])
#     .size()
#     .groupby("customer_id")
#     .std()
#     .reset_index(name="txn_freq_std")
# )
# #if std is high then it means the customer has inconsistent transaction behavior, 
# # if low then they have consistent behavior.this helps to detect spike/salary days.

# #time of transaction: does the customer transact at a specific time of the day or is it spread out throughout the day.
# df["txn_hour"] = df["txn_date"].dt.hour
# txn_time_spread = df.groupby("customer_id")["txn_hour"].std().reset_index(name="txn_time_spread")
# #if specific time then low std, if spread out then high std. this helps to detect if they are habitual time users or not.

# #high value transaction ratio: what percentage of their transactions are above a certain threshold (e.g., 50,000 NPR). if they have high ratio then they are more likely to be high value customers but also more risky.
# high_txn = df[df["amount_npr"] > 50000].groupby("customer_id").size()
# total_txn = df.groupby("customer_id").size()
# high_value_ratio = (high_txn / total_txn).reset_index(name="high_value_txn_ratio")

# #new payee: how many unique merchants does the customer transact with. if they transact with more merchants then they are more likely to be engaged but also more risky.
# new_merchant = df.groupby("customer_id")["merchant_id"].nunique().reset_index(name="unique_merchants")

# #this shows their risk behavior

# #============================================
# #spend to load ratio
# load = df[df["txn_type"] == "topup"].groupby("customer_id")["amount_npr"].sum()
# spend = df[df["txn_type"] != "topup"].groupby("customer_id")["amount_npr"].sum()
# ratio = (spend / load).reset_index(name="spend_to_load_ratio")


# #how many different categories of services the customer uses. if they use more categories then they are more likely to be engaged and loyal.
# service_diversity = df.groupby("customer_id")["category"].nunique().reset_index(name="service_diversity")

# #spend trend: is the customer's spending increasing, decreasing, or stable over time. 
# monthly_spend = df.groupby(["customer_id", "month"])["amount_npr"].sum().reset_index()
# trend = monthly_spend.groupby("customer_id")["amount_npr"].apply(
#     lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
# ).reset_index(name="spend_trend")
# #if increasing then engaged else churn


# # print(df.info())




import pandas as pd
import numpy as np

pd.options.display.max_columns = 100


df = pd.read_csv("data/clean_fintech_data.csv")

# Convert date columns to datetime
df["txn_date"] = pd.to_datetime(df["txn_date"])
df["account_open_date"] = pd.to_datetime(df["account_open_date"])
df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])

# -----------------------------------------------
# PRECOMPUTE COMMON COLUMNS (for efficiency)
# -----------------------------------------------
df["month"] = df["txn_date"].dt.to_period("M")
df["txn_hour"] = df["txn_date"].dt.hour
df["is_topup"] = (df["txn_type"] == "topup").astype(int)
df["high_txn"] = (df["amount_npr"] > 50000).astype(int)

# -----------------------------------------------
# CREATE GROUPED OBJECTS (IMPORTANT OPTIMIZATION)
# -----------------------------------------------
grp = df.groupby("customer_id")
grp_month = df.groupby(["customer_id", "month"])

# -----------------------------------------------
# 1. TRANSACTION BEHAVIOR FEATURES
# -----------------------------------------------
customer_df = grp.agg(
    total_spent=("amount_npr", "sum"),        # total money spent
    avg_spent=("amount_npr", "mean"),         # avg transaction value
    max_txn=("amount_npr", "max"),            # highest transaction
    txn_count=("txn_id", "count"),            # total transactions
    
    active_days=("txn_date", "nunique"),      # unique days active
    feature_adoption=("product_name", "nunique"),  # number of products used
    service_diversity=("category", "nunique"),     # number of categories used
    unique_merchants=("merchant_id", "nunique")    # unique merchants interacted
).reset_index()

# -----------------------------------------------
# 2. RECENCY (ENGAGEMENT)
# -----------------------------------------------
latest_date = df["txn_date"].max()

customer_df["recency_days"] = (
    latest_date - grp["txn_date"].max()
).dt.days.values

# Lower recency_days = more active user

# -----------------------------------------------
# 3. FAILED TRANSACTION RATE (RISK)
# -----------------------------------------------
customer_df["failed_txn_rate"] = grp["status"].apply(
    lambda x: (x == "failed").mean()
).values

# -----------------------------------------------
# 4. MONTHLY TRANSACTION FEATURES
# -----------------------------------------------
monthly_txn = grp_month.size().rename("monthly_txn")
monthly_spend = grp_month["amount_npr"].sum()

# Average transactions per month
customer_df["txn_freq_per_month"] = (
    monthly_txn.groupby("customer_id").mean().values
)

# Transaction consistency (std)
customer_df["txn_freq_std"] = (
    monthly_txn.groupby("customer_id").std().values
)

# -----------------------------------------------
# 5. TIME BEHAVIOR (RISK SIGNAL)
# -----------------------------------------------
customer_df["txn_time_spread"] = grp["txn_hour"].std().values

# High spread = irregular timing (potential anomaly)

# -----------------------------------------------
# 6. HIGH VALUE TRANSACTION RATIO
# -----------------------------------------------
customer_df["high_value_txn_ratio"] = grp["high_txn"].mean().values

# -----------------------------------------------
# 7. SPEND TO LOAD RATIO (FINTECH CORE METRIC)
# -----------------------------------------------
load = grp.apply(lambda x: x.loc[x["is_topup"] == 1, "amount_npr"].sum())
spend = grp.apply(lambda x: x.loc[x["is_topup"] == 0, "amount_npr"].sum())

customer_df["spend_to_load_ratio"] = (spend / load).replace(
    [np.inf, -np.inf], 0
).values

# -----------------------------------------------
# 8. SPENDING TREND (GROWTH / CHURN SIGNAL)
# -----------------------------------------------
def calc_trend(x):
    if len(x) > 1:
        return np.polyfit(range(len(x)), x, 1)[0]
    return 0

trend = monthly_spend.groupby("customer_id").apply(calc_trend)

customer_df["spend_trend"] = trend.values

# Positive → increasing usage
# Negative → declining (churn risk)

# -----------------------------------------------
# FINAL CLEANING
# -----------------------------------------------
customer_df = customer_df.fillna(0)

# -----------------------------------------------
# READY FOR MODELING
# -----------------------------------------------
print(customer_df.head())
print(customer_df.shape)
print(customer_df.info())



#------------------------------------
# FEATURE TRANSFORMATION 
#------------------------------------


print(customer_df.skew(numeric_only=True).sort_values(ascending=False))


# LOG TRANSFORM (only skewed continuous features):
log_features = [
    "total_spent",
    "avg_spent",
    "max_txn",
    "txn_freq_per_month",
    "recency_days",
    "spend_to_load_ratio"
]

for col in log_features:
    customer_df[col] = np.log1p(customer_df[col])

# DO NOT log these:"failed_txn_rate"and "high_value_txn_ratio" as they are already in [0,1] range and log transform would distort them.

# instead flag them as binary features;
customer_df["has_failed_txn"] = (customer_df["failed_txn_rate"] > 0).astype(int)
customer_df["has_high_value_txn"] = (customer_df["high_value_txn_ratio"] > 0).astype(int)

# or clipping only the failed_txn_rate to 0.5 as it is a risk signal and we want to cap extreme values but not distort the distribution too much.
customer_df["failed_txn_rate"] = customer_df["failed_txn_rate"].clip(upper=0.5)   
customer_df["high_value_txn_ratio"] = customer_df["high_value_txn_ratio"].clip(upper=0.8) 

# SCALING:
from sklearn.preprocessing import StandardScaler

#before scaling we need to drop customer_id as it is not a feature.
X_df = customer_df.drop("customer_id", axis=1).copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

#re-create dataframe with scaled features and add customer_id back for reference
X_scaled_df = pd.DataFrame(X_scaled, columns=X_df.columns)

X_scaled_df["customer_id"] = customer_df["customer_id"]
cols = ["customer_id"] + list(X_df.columns) # ensure customer_id is the first column
X_scaled_df = X_scaled_df[cols]

print(X_scaled_df.head())
print(X_scaled_df.describe())

#saving the scaled features for modeling
X_scaled_df.to_csv("data/scaled_features.csv", index=False)
X_scaled_df.to_feather("data/scaled_features.feather")

# check that the scaled features have mean ~0 and std ~1
print(X_scaled_df.select_dtypes(include='number').mean())
print(X_scaled_df.select_dtypes(include='number').std())