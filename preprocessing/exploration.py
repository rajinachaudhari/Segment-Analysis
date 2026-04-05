# Step 1: Understanding the Problem and the Data
# Step 2: Importing and Inspecting the Data
# Step 3: Handling Missing Data
# Step 4: Exploring Data Characteristics

import pandas as pd 




#step 1 and 2:

df = pd.read_csv("data/fintech_data.csv")

pd.options.display.max_columns = 100


print(df.head())
print(df.tail())


# What variables are present in the dataset and what do they represent?
print(df.info())
# What types of data are available (numerical, categorical, text etc.)?
print(df.dtypes)  #same result as of info()
# Are there any data quality issues or limitations?
print(df.isnull().sum())  #finds the total number of null value in each column by adding number of true values(boolean values) given by isnull()
print(df["status"].value_counts()["failed"])  #counts no. of failed transactions


print(df.shape)
print(df.describe())





# #step 4:

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import re

# #step4.1:check data distribution 

output_dir = Path("visuals")

raw_dir = output_dir / "raw_data_visuals"


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


# Split columns by type
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols =[
    "gender",
    "employment_status",
    "marital_status",
    "account_type",
    "category",          # merchant category
    "product_name",
    "txn_type",
    "status"
]

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

if numeric_cols:
    # Dashboard 1: Boxplots for all numeric columns
    fig, axes = create_subplot_grid(len(numeric_cols), max_cols=3, figsize_per_plot=(6, 4))
    fig.suptitle("Boxplots for Numeric Columns", fontsize=18)

    for ax, col in zip(axes, numeric_cols):
        sns.boxplot(x=df[col], ax=ax, color="skyblue")
        ax.set_title(col)
        ax.set_xlabel(col)

    for ax in axes[len(numeric_cols):]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(raw_dir / "boxplot_dashboard.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Dashboard 2: Histograms for all numeric columns
    fig, axes = create_subplot_grid(len(numeric_cols), max_cols=3, figsize_per_plot=(6, 4))
    fig.suptitle("Histograms for Numeric Columns", fontsize=18)

    for ax, col in zip(axes, numeric_cols):
        sns.histplot(df[col], kde=False, ax=ax, color="orange", bins=30)
        ax.set_title(col)
        ax.set_xlabel(col)

    for ax in axes[len(numeric_cols):]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(raw_dir / "histogram_dashboard.png", dpi=150, bbox_inches="tight")
    plt.show()

  

if categorical_cols:
    # Dashboard 3: Bar plots for all categorical columns
    fig, axes = create_subplot_grid(len(categorical_cols), max_cols=2, figsize_per_plot=(8, 5))
    fig.suptitle("Top Frequency Values for Categorical Columns", fontsize=18)

    for ax, col in zip(axes, categorical_cols):
        top_values = df[col].value_counts(dropna=False).head(15)
        sns.barplot(x=top_values.index.astype(str), y=top_values.values, ax=ax, palette="viridis")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

    for ax in axes[len(categorical_cols):]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(raw_dir / "categorical_bar_dashboard.png", dpi=150, bbox_inches="tight")
    plt.show()


    

