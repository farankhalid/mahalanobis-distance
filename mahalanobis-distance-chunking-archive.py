import pandas as pd
import numpy as np
from scipy.spatial import distance
import logging

# Configure logging
log_filename = "log/mahalanobis.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def regularize_cov_matrix(cov_matrix, alpha=1e-5):
    logging.info("Regularizing covariance matrix.")
    regularized_cov = cov_matrix + alpha * np.eye(cov_matrix.shape[0])
    return regularized_cov


logging.info("Reading data from CSV files.")
patents = pd.read_csv("data/MainfilrARS2.csv")
patent_shares = pd.read_csv("data/MainfilrARS2_share.csv")

logging.info("Processing patents DataFrame (Firm-Year Level).")
patents["permno_year"] = patents["permno_adj"].astype(str) + patents[
    "publn_year"
].astype(str)
patents_grouped = (
    patents.groupby(["permno_year", "IPC1"])["publn_nr"].nunique().reset_index()
)
patents_grouped.columns = ["permno_year", "IPC1", "ipc_patents"]
patents_grouped["total_patents"] = patents_grouped.groupby(["permno_year"])[
    "ipc_patents"
].transform("sum")
patents_grouped["share"] = (
    patents_grouped["ipc_patents"] / patents_grouped["total_patents"]
)

chunk_size = 5000
logging.info(f"Chunking patents_grouped DataFrame with chunk size: {chunk_size}.")
chunks = [
    patents_grouped.iloc[i : i + chunk_size]
    for i in range(0, patents_grouped.shape[0], chunk_size)
]

processed_chunks = 0

firm_vector = pd.DataFrame()
logging.info("Processing each chunk to create firm_vector.")
for chunk in chunks:
    chunk_pivoted = chunk.pivot(
        index="permno_year", columns="IPC1", values="share"
    ).fillna(0)
    firm_vector = pd.concat([firm_vector, chunk_pivoted])

logging.info("Processing patent_shares DataFrame (Patent Level).")
patent_shares["permno_year"] = patent_shares["permno_adj"].astype(str) + patent_shares[
    "publn_year"
].astype(str)
patent_vector = patent_shares.pivot(
    index="publn_nr", columns="IPC1", values="share_pat"
).fillna(0)

logging.info("Aligning columns of firm_vector and patent_vector.")
common_columns = sorted(list(set(firm_vector.columns) & set(patent_vector.columns)))
firm_vector_aligned = firm_vector[common_columns].fillna(0)
patent_vector_aligned = patent_vector[common_columns].fillna(0)

logging.info("Converting DataFrames to NumPy arrays for Mahalanobis calculation.")
firm_vector_np = firm_vector_aligned.to_numpy()
cov_matrix = np.cov(firm_vector_np, rowvar=False)
inv_cov_matrix = np.linalg.inv(regularize_cov_matrix(cov_matrix))

results = []
logging.info("Calculating Mahalanobis distance.")
for index, row in patent_vector_aligned.iterrows():
    if row.name in patent_shares["publn_nr"].values:
        firm_year = patent_shares[patent_shares["publn_nr"] == row.name][
            "permno_year"
        ].iloc[0]
        if firm_year in firm_vector_aligned.index:
            firm_row = firm_vector_np[firm_vector_aligned.index == firm_year][0]
            patent_row = row.values.flatten()

            try:
                mahal_dist = distance.mahalanobis(patent_row, firm_row, inv_cov_matrix)
                results.append([firm_year, row.name, mahal_dist])
            except ValueError as e:
                logging.error(
                    f"Error calculating Mahalanobis distance for {row.name}: {e}"
                )

logging.info("Converting results to DataFrame.")
results_df = pd.DataFrame(results, columns=["permno_year", "publn_nr", "mahal_dist"])

logging.info("Saving the results to a CSV file.")
results_df.to_csv("output/mahalanobis-results-chunking-archived.csv", index=False)
print(
    "Mahalanobis distance calculation is completed and saved to output/mahalanobis-results-chunking-archived.csv"
)
