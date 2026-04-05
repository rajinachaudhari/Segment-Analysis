import pandas as pd

pd.options.display.max_columns = 100

# load data
df = pd.read_csv("data/fintech_data.csv")


# # Shows rows where either marital_status OR product_name is missing
#  print(df[df['marital_status'].isnull() | df['product_name'].isnull()]) 
# # #only 0.006% null value so drop it

# remove rows with missing values
df = df.dropna(subset=["marital_status", "product_name"])

# convert date columns
df["txn_date"] = pd.to_datetime(df["txn_date"])
df["account_open_date"] = pd.to_datetime(df["account_open_date"])
df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])

# drop unnecessary columns
df.drop(
    columns=["first_name", "last_name", "product_id"],
    inplace=True,
    errors="ignore"
)

# check duplicate transactions
print("Duplicate txn_id:", df["txn_id"].duplicated().sum())

# check negative transaction amounts
print("Negative transactions:", (df["amount_npr"] < 0).sum())

# sort by transaction date
df = df.sort_values("txn_date")

# save cleaned dataset
df.to_csv("data/clean_fintech_data.csv", index=False)

# final dataset info
df.info()

#NOTE: The outliers present in the dataset may be valid information like high-value transactions, so we will not remove them without further analysis.