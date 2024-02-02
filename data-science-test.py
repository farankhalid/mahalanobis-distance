import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load and preprocess the data
logging.debug("Loading CSV files.")
patent_shares = pd.read_csv("data/MainfilrARS2_share.csv")
firm_data = pd.read_csv("data/MainfilrARS2.csv")

logging.debug(f"Loaded patent_shares with shape {patent_shares.shape}.")
logging.debug(f"Loaded firm_data with shape {firm_data.shape}.")

# Preprocess and calculate shares for firm-year level
logging.debug("Preprocessing firm_data and calculating shares.")
firm_data["permno_year"] = firm_data["permno_adj"].astype(str) + firm_data[
    "publn_year"
].astype(str)
firm_shares = (
    firm_data.groupby(["permno_year", "IPC1"])
    .size()
    .groupby(level=0)
    .transform(lambda x: x / x.sum())
    .reset_index()
)
logging.info(f"firm shares: {firm_shares}")
firm_shares.columns = ["permno_year", "IPC1", "share"]

logging.debug(f"Processed firm_shares with shape {firm_shares.shape}.")

# Ensure the data types match between dataframes
patent_shares["permno_year"] = patent_shares["permno_adj"].astype(str) + patent_shares[
    "publn_year"
].astype(str)

# Prepare the patent-level DataFrame
logging.debug("Pivoting patent_shares to create patent_vectors.")
patent_vectors = patent_shares.pivot(
    index="publn_nr", columns="IPC1", values="share_pat"
).fillna(0)

# Prepare the firm-year level DataFrame
logging.debug("Pivoting firm_shares to create firm_vectors.")
firm_vectors = firm_shares.pivot(
    index="permno_year", columns="IPC1", values="share"
).fillna(0)

# Align the columns of both DataFrames
logging.debug("Aligning IPC classes between patent_vectors and firm_vectors.")
all_ipc_classes = firm_vectors.columns.union(patent_vectors.columns)
firm_vectors = firm_vectors.reindex(columns=all_ipc_classes, fill_value=0)
patent_vectors = patent_vectors.reindex(columns=all_ipc_classes, fill_value=0)

# Calculate the pseudo-inverse of the covariance matrix
logging.debug(
    "Calculating the pseudo-inverse of the covariance matrix for Mahalanobis distance."
)
cov_matrix = np.cov(firm_vectors.T)
inv_cov_matrix = pinv(cov_matrix)

# Initialize an empty list for the output data
output_data = []

# Calculate Mahalanobis distance for each patent
logging.debug(
    "Calculating Mahalanobis distance for each patent compared to its firm-year vector."
)
for publn_nr, row in patent_vectors.iterrows():
    patent_vector = row.values
    permno_year = patent_shares[patent_shares["publn_nr"] == publn_nr][
        "permno_year"
    ].iloc[0]
    if permno_year in firm_vectors.index:
        firm_vector = firm_vectors.loc[permno_year].values
        distance = mahalanobis(patent_vector, firm_vector, inv_cov_matrix)
        output_data.append(
            {
                "permno_year": permno_year,
                "publn_nr": publn_nr,
                "mahalanobis_distance": distance,
            }
        )

# Convert the output data to a DataFrame and save
logging.debug("Converting output data to DataFrame and saving to CSV.")
output_df = pd.DataFrame(output_data)
output_df.to_csv("output/mahalanobis-distances.csv", index=False)

logging.debug(
    "Calculation completed and output saved. Check mahalanobis_distances.csv for results."
)
