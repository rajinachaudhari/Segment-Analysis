
import pandas as pd
import numpy as np
import uuid, random
from datetime import date, timedelta
from pathlib import Path

np.random.seed(2025)
random.seed(2025)

# -------------------------------
# CONFIGURATION
# -------------------------------
CFG = {
    "N_USERS": 50_000,
    "N_MERCHANTS": 2_000,
    "START_DATE": date(2025, 1, 1),
    "END_DATE": date(2025, 12, 31),
    "CHUNK_SIZE": 50_000,
    "FRAUD_RATE": 0.008,
    "DORMANT_RATE": 0.22,
    "OUTPUT_DIR": Path("data"),
}
CFG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)

# -------------------------------
# USER TABLE
# -------------------------------
def generate_users(n=CFG["N_USERS"]):
    user_ids = [str(uuid.uuid4()) for _ in range(n)]

    df = pd.DataFrame({
        "user_id": user_ids,
        "kyc_level": np.random.choice([0,1,2], size=n, p=[0.3,0.4,0.3]),
        "registration_date": [
            CFG["START_DATE"] + timedelta(days=np.random.randint(0, 365))
            for _ in range(n)
        ],
        "account_type": np.random.choice(['CUSTOMER','AGENT'], size=n, p=[0.97,0.03])
    })

    df.to_parquet(CFG["OUTPUT_DIR"] / "users.parquet", index=False)
    return df

# -------------------------------
# WALLET TABLE
# -------------------------------
def generate_wallets(users_df):
    n = len(users_df)

    balances = np.random.uniform(100, 50000, n).round(2)
    total_loaded = balances + np.random.uniform(0, 100000, n)
    total_spent = total_loaded - balances

    now = pd.Timestamp("2025-12-31")

    df = pd.DataFrame({
        "wallet_id": [str(uuid.uuid4()) for _ in range(n)],
        "user_id": users_df["user_id"],
        "balance": balances,
        "total_loaded_npr": total_loaded,
        "total_spent_npr": total_spent,
        "last_topup_at": now - pd.to_timedelta(np.random.randint(0, 90, n), unit='d'),
        "last_transaction_at": now - pd.to_timedelta(np.random.randint(0, 30, n), unit='d'),
        "is_frozen": np.random.choice([True, False], size=n, p=[0.02, 0.98])
    })

    df.to_parquet(CFG["OUTPUT_DIR"] / "wallets.parquet", index=False)
    return df

# -------------------------------
# TRANSACTIONS TABLE
# -------------------------------
TXN_TYPES = ['TOPUP','P2P_TRANSFER','MERCHANT_PAY','CASHOUT']

def generate_transactions(users_df):
    rows = []

    start = pd.Timestamp(CFG["START_DATE"])  

    for _, user in users_df.iterrows():

        # skip dormant users
        if np.random.random() < CFG["DORMANT_RATE"]:
            continue

        n_txn = np.random.randint(5, 50)

        for _ in range(n_txn):
            txn_type = np.random.choice(TXN_TYPES)

            #  realistic amount 
            amount = np.random.lognormal(mean=8, sigma=1)
            amount = float(np.clip(amount, 50, 50000))

            rows.append({
                "txn_id": str(uuid.uuid4()),
                "user_id": user["user_id"],   
                "txn_type": txn_type,
                "amount": round(amount, 2),
                "status": np.random.choice(['SUCCESS','FAILED'], p=[0.93,0.07]),
                "channel": np.random.choice(['APP','AGENT','USSD']),
                "created_at": start + pd.to_timedelta(np.random.randint(0,365), unit='d'),
                "merchant_id": str(uuid.uuid4()) if txn_type=="MERCHANT_PAY" else None,
                "receiver_user_id": random.choice(users_df["user_id"].tolist()) if txn_type=="P2P_TRANSFER" else None
            })

    df = pd.DataFrame(rows)
    df.to_parquet(CFG["OUTPUT_DIR"] / "transactions.parquet", index=False)
    return df

# -------------------------------
# MERGE ALL TABLES
# -------------------------------
def create_final_dataset(users, wallets, transactions):

    # Start from transactions since it's the largest table, and merge user/wallet info
    df = transactions.merge(users, on="user_id", how="left")
    df = df.merge(wallets, on="user_id", how="left")

    df.to_parquet(CFG["OUTPUT_DIR"] / "final_dataset.parquet", index=False)

    print("Final dataset saved")
    print("Shape:", df.shape)

    return df

# -------------------------------
# MAIN
# -------------------------------
def main():
    print("Generating data...")

    users = generate_users()
    wallets = generate_wallets(users)
    transactions = generate_transactions(users)

    final_df = create_final_dataset(users, wallets, transactions)

    print(" Data generation complete")

if __name__ == "__main__":
    main()