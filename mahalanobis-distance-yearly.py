import pandas as pd
import numpy as np
from scipy.spatial import distance


def regularize_cov_matrix(cov_matrix, alpha=1e-5):
    if cov_matrix.size == 0 or np.isscalar(cov_matrix) or cov_matrix.ndim < 2:
        # Return None if cov_matrix is not valid
        return None
    return cov_matrix + alpha * np.eye(cov_matrix.shape[0])


# Load data
patents = pd.read_csv("oldest-data/test1.csv")
patent_shares = pd.read_csv("oldest-data/sample_w_patshare2.csv")

# Preprocess data
patents["permno_year"] = patents["permno_adj"].astype(str) + patents[
    "publn_year"
].astype(str)
patent_shares["permno_year"] = patent_shares["permno_adj"].astype(str) + patent_shares[
    "publn_year"
].astype(str)

# Aggregate and prepare vectors
patents_grouped = (
    patents.groupby(["permno_year", "IPC1"])["publn_nr"]
    .nunique()
    .reset_index(name="ipc_patents")
)
patents_grouped["total_patents"] = patents_grouped.groupby("permno_year")[
    "ipc_patents"
].transform("sum")
patents_grouped["share"] = (
    patents_grouped["ipc_patents"] / patents_grouped["total_patents"]
)

firm_vector = patents_grouped.pivot(
    index="permno_year", columns="IPC1", values="share"
).fillna(0)
patent_vector = patent_shares.pivot(
    index="publn_nr", columns="IPC1", values="share_pat"
).fillna(0)

# Ensure columns match
common_columns = firm_vector.columns.intersection(patent_vector.columns)
firm_vector_filtered = firm_vector[common_columns].fillna(0).to_numpy()
patent_vector_filtered = patent_vector.reindex(
    columns=common_columns, fill_value=0
).to_numpy()

results = []

# Iterate over each patent
for index, row in patent_vector.iterrows():
    firm_year = patent_shares.loc[
        patent_shares["publn_nr"] == index, "permno_year"
    ].iloc[0]
    if firm_year in firm_vector.index:
        firm_row = firm_vector.loc[firm_year, common_columns].fillna(0).to_numpy()
        cov_matrix = np.cov(firm_vector_filtered, rowvar=False)
        inv_cov_matrix = regularize_cov_matrix(cov_matrix)

        # Skip if inv_cov_matrix is None or ill-conditioned
        if (
            inv_cov_matrix is None
            or np.linalg.cond(inv_cov_matrix) > 1 / np.finfo(inv_cov_matrix.dtype).eps
        ):
            print(
                f"Skipping {index} due to singular or ill-conditioned covariance matrix."
            )
            continue

        try:
            mahal_dist = distance.mahalanobis(row.to_numpy(), firm_row, inv_cov_matrix)
            results.append([firm_year, index, mahal_dist])
        except ValueError as e:
            print(f"Error calculating Mahalanobis distance for {index}: {e}")

# Convert results to DataFrame and save
results_df = pd.DataFrame(results, columns=["permno_year", "publn_nr", "mahal_dist"])
results_df.to_csv("output/mahalanobis-results-updated.csv", index=False)
print(
    "Mahalanobis distance calculation is completed and saved to output/mahalanobis-results-updated.csv."
)
