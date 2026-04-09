# Step 1: Understanding the Problem and the Data
# Step 2: Importing and Inspecting the Data
# Step 3: Handling Missing Data
# Step 4: Exploring Data Characteristics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path




#step 1 and 2:

df = pd.read_parquet("data/final_dataset.parquet")

pd.options.display.max_columns = 100


print(df.head())
# print(df.tail())


# What variables are present in the dataset and what do they represent?
print(df.info())
# What types of data are available (numerical, categorical, text etc.)?
# print(df.dtypes)  #same result as of info()
# Are there any data quality issues or limitations?
print(df.isnull().sum())  #finds the total number of null value in each column by adding number of true values(boolean values) given by isnull()


print(df.shape)
print(df.describe())

print("\nFAILED TXNS:", df["status"].value_counts().get("FAILED", 0))


# DUPLICATE CHECK

dup_count = df["txn_id"].duplicated().sum()
print("\nDuplicate txn_id:", dup_count)

# NEGATIVE VALUE CHECK

neg_amount = (df["amount"] < 0).sum()
print("Negative transactions:", neg_amount)




# #step 4:


# #step4.1:check data distribution 

output_dir = Path("visuals")
raw_dir = output_dir / "raw_data_visuals"
raw_dir.mkdir(parents=True, exist_ok=True)


numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

categorical_cols = [
    "kyc_level",
    "account_type",
    "txn_type",
    "status",
    "channel",
    "is_frozen"
]

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# Split columns by type
def create_subplot_grid(total_plots, max_cols=3, figsize_per_plot=(6, 4)):
    cols = min(max_cols, max(1, total_plots))
    rows = int(np.ceil(total_plots / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows),
    )
    axes = np.array(axes).reshape(-1)
    return fig, axes




if numeric_cols:
    # Dashboard 1: Boxplots for all numeric columns
    fig, axes = create_subplot_grid(len(numeric_cols), max_cols=3, figsize_per_plot=(6, 4))
    fig.suptitle("Boxplots for Numeric Columns", fontsize=18)

    for ax, col in zip(axes, numeric_cols):
        ax.boxplot(df[col])
        ax.set_title(col)

    for ax in axes[len(numeric_cols):]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(raw_dir / "boxplot_dashboard.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Dashboard 2: Histograms for all numeric columns
    fig, axes = create_subplot_grid(len(numeric_cols), max_cols=3, figsize_per_plot=(6, 4))
    fig.suptitle("Histograms for Numeric Columns", fontsize=18)

    for ax, col in zip(axes, numeric_cols):
        ax.hist(df[col], bins=30)
        ax.set_title(col)
        
    for ax in axes[len(numeric_cols):]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(raw_dir / "histogram_dashboard.png", dpi=150, bbox_inches="tight")
    plt.show()

  

if categorical_cols:
    # Dashboard 3: Bar plots for all categorical columns
   if categorical_cols:
    fig, axes = create_subplot_grid(len(categorical_cols), max_cols=2)
    fig.suptitle("Categorical Distributions", fontsize=18)

    for ax, col in zip(axes, categorical_cols):
        top_values = df[col].value_counts().head(10)
        ax.bar(top_values.index.astype(str), top_values.values)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=45)

    for ax in axes[len(categorical_cols):]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(raw_dir / "categorical_bar_dashboard.png")
    plt.show()

    

#==================
#CLEANING STEPS
#==================

df["registration_date"] = pd.to_datetime(df["registration_date"])



# NOTE:
# merchant_id & receiver_user_id are STRUCTURAL NULLS
# → do NOT drop or impute with mean

# -------------------------------
# CREATE HELPFUL FLAGS FOR MODELING MISSING VALUES
# -------------------------------
df["is_merchant_txn"] = df["merchant_id"].notnull().astype(int)
df["is_p2p_txn"] = df["receiver_user_id"].notnull().astype(int)
df["is_success"] = (df["status"] == "SUCCESS").astype(int)

#  fill for modeling convenience
df["merchant_id"] = df["merchant_id"].fillna("NO_MERCHANT")
df["receiver_user_id"] = df["receiver_user_id"].fillna("NO_RECEIVER")


# -------------------------------
# SORT DATA
# -------------------------------
df = df.sort_values("created_at")

# -------------------------------
# SAVE CLEAN DATA
# -------------------------------
df.to_parquet("data/clean_fintech_data.parquet", index=False)

# -------------------------------
# FINAL INFO
# -------------------------------
print("\nFinal dataset info:")
df.info()

# -------------------------------
# NOTE
# -------------------------------
# Outliers are NOT removed because:
# - High-value transactions are realistic in fintech
# - Important for fraud detection / segmentation later