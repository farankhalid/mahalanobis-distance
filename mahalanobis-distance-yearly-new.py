import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv
from numpy.linalg import matrix_rank
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Loading datasets...")
df_main = pd.read_csv("data/MainfilrARS2.csv")
df_share = pd.read_csv("data/MainfilrARS2_share.csv")
logging.info("Datasets loaded.")

# Preprocessing
logging.info("Starting preprocessing...")
df_main["permno_year"] = df_main["permno_adj"].astype(str) + df_main[
    "publn_year"
].astype(str)
df_share["permno_year"] = df_share["permno_adj"].astype(str) + df_share[
    "publn_year"
].astype(str)
logging.info("Preprocessing completed.")


def aggregate_ipc_shares(df, df_type):
    logging.info(f"Aggregating IPC shares for {df_type} data...")
    if df_type == "main":
        # Calculate IPC count and share for df_main
        df_agg = (
            df.groupby(["permno_year", "IPC1"]).size().reset_index(name="ipc_count")
        )
        total_ipc = df_agg.groupby("permno_year")["ipc_count"].transform("sum")
        df_agg["share"] = df_agg["ipc_count"] / total_ipc
    elif df_type == "share":
        # Sum up shares for each permno_year and IPC1 combination in df_share
        df_agg = (
            df.groupby(["permno_year", "IPC1"]).agg({"share_pat": "sum"}).reset_index()
        )
        df_agg.rename(columns={"share_pat": "share"}, inplace=True)

    pivot_df = df_agg.pivot(index="permno_year", columns="IPC1", values="share").fillna(
        0
    )
    logging.info(f"Aggregation complete for {df_type} data.")
    return pivot_df


# Function to compute Mahalanobis distance
def compute_mahalanobis(df_firm, df_patent, cov_inv):
    logging.info("Computing Mahalanobis distances...")
    distances = []
    for index, row in df_patent.iterrows():
        if index in df_firm.index:
            firm_vector = df_firm.loc[index].values
            patent_vector = row.values
            try:
                distance = mahalanobis(patent_vector, firm_vector, cov_inv)
                distances.append(
                    {"permno_year": index, "mahalanobis_distance": distance}
                )
            except ValueError as e:
                logging.error(f"Error computing distance for {index}: {e}")
                distances.append({"permno_year": index, "mahalanobis_distance": np.nan})
    logging.info("Mahalanobis distances computed.")
    return pd.DataFrame(distances)


# Process each year separately
logging.info("Starting yearly processing...")
years = sorted(df_main["publn_year"].unique())

all_results = pd.DataFrame()

for year in years:
    logging.info(f"Processing year: {year}")
    df_main_year = df_main[df_main["publn_year"] == year]
    df_share_year = df_share[df_share["publn_year"] == year]

    # Aggregate shares for both datasets
    df_main_agg = aggregate_ipc_shares(df_main_year, "main")
    df_share_agg = aggregate_ipc_shares(df_share_year, "share")

    # Compute covariance matrix and its inverse
    logging.info("Computing covariance matrix and its inverse...")
    cov_matrix = np.cov(df_main_agg.T)
    if matrix_rank(cov_matrix) == cov_matrix.shape[0]:
        cov_inv = pinv(cov_matrix)
    else:
        # Regularization to handle singular matrices
        cov_inv = pinv(cov_matrix + np.eye(cov_matrix.shape[0]) * 0.01)
    logging.info("Covariance matrix processing completed.")

    # Compute Mahalanobis distances
    yearly_results = compute_mahalanobis(df_main_agg, df_share_agg, cov_inv)
    all_results = pd.concat([all_results, yearly_results], ignore_index=True)

logging.info("Yearly processing completed. Saving results...")
all_results.to_csv("output/mahalanobis-distances-yearly-new.csv", index=False)
logging.info("Results saved.")
