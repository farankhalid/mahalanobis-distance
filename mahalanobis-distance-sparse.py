import pandas as pd
import numpy as np
from scipy.spatial import distance


# Regularize the covariance matrix
def regularize_cov_matrix(cov_matrix, alpha=1e-5):
    regularized_cov = cov_matrix + alpha * np.eye(cov_matrix.shape[0])
    return regularized_cov


# Read data from the CSV files
patents = pd.read_csv("oldest-data/test1.csv")
patent_shares = pd.read_csv("oldest-data/sample_w_patshare2.csv")

# Process patents DataFrame (Firm-Year Level)
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

# Convert the 'share' column to a sparse dtype
patents_grouped["share"] = patents_grouped["share"].astype(
    pd.SparseDtype(float, fill_value=0)
)

# Creating firm vector with sparse representation
firm_vector = patents_grouped.pivot(index="permno_year", columns="IPC1", values="share")

# Process patent_shares DataFrame (Patent Level)
patent_shares["permno_year"] = patent_shares["permno_adj"].astype(str) + patent_shares[
    "publn_year"
].astype(str)
patent_vector = patent_shares.pivot(
    index="publn_nr", columns="IPC1", values="share_pat"
).fillna(0)

# Ensure that the columns in both vectors match
common_columns = firm_vector.columns.intersection(patent_vector.columns)
firm_vector_filtered = firm_vector[common_columns].sparse.to_dense().to_numpy()
patent_vector_filtered = patent_vector[common_columns].to_numpy()

# Calculate Mahalanobis distance
cov_matrix = np.cov(firm_vector_filtered, rowvar=False)
inv_cov_matrix = np.linalg.inv(regularize_cov_matrix(cov_matrix))
results = []
# processed_chunks = processed_chunks + 1
# logging.info(f"Chunks processed: {processed_chunks}")

for index, row in patent_vector.iterrows():
    if row.name in patent_shares["publn_nr"].values:
        firm_year = patent_shares[patent_shares["publn_nr"] == row.name][
            "permno_year"
        ].iloc[0]
        if firm_year in firm_vector.index:
            firm_row = firm_vector.loc[firm_year].sparse.to_dense().to_numpy()
            mahal_dist = distance.mahalanobis(
                row[common_columns], firm_row, inv_cov_matrix
            )
            print(mahal_dist)
            results.append([firm_year, row.name, mahal_dist])

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=["permno_year", "publn_nr", "mahal_dist"])

# Save the results to a CSV file
results_df.to_csv("output/mahalanobis-results-sparse.csv", index=False)
print(
    "Mahalanobis distance calculation is completed and saved to output/mahalanobis-results-sparse.csv"
)
