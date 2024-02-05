import dask.dataframe as dd
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.debug("Loading CSV files with Dask.")
patent_shares = dd.read_csv("data/MainfilrARS2_share.csv")
firm_data = dd.read_csv("data/MainfilrARS2.csv")

# Create 'permno_year' column
patent_shares["permno_year"] = (
    patent_shares["permno_adj"].astype(str)
    + "_"
    + patent_shares["publn_year"].astype(str)
)
firm_data["permno_year"] = (
    firm_data["permno_adj"].astype(str) + "_" + firm_data["publn_year"].astype(str)
)

# Compute aggregation in Dask, then reset index and rename
logging.debug("Aggregating data.")
firm_aggregated = firm_data.groupby(["permno_year", "IPC1"]).size().reset_index()
firm_aggregated = firm_aggregated.rename(
    columns={0: "counts"}
)  # Correct way to rename the aggregated column in Dask

# Convert aggregated data to pandas DataFrame for further processing (if necessary)
firm_shares_pd = firm_aggregated.compute()
patent_shares_pd = patent_shares.compute()

# Calculate shares for each permno_year and IPC1
firm_shares_pd["share"] = firm_shares_pd.groupby("permno_year")["counts"].transform(
    lambda x: x / x.sum()
)

# Now pivot data using pandas
patent_vectors_pd = patent_shares_pd.pivot_table(
    index="publn_nr", columns="IPC1", values="share_pat", fill_value=0
)
firm_vectors_pd = firm_shares_pd.pivot_table(
    index="permno_year", columns="IPC1", values="share", fill_value=0
)

# Align the columns of both DataFrames
all_ipc_classes = patent_vectors_pd.columns.union(firm_vectors_pd.columns)
patent_vectors_pd = patent_vectors_pd.reindex(columns=all_ipc_classes, fill_value=0)
firm_vectors_pd = firm_vectors_pd.reindex(columns=all_ipc_classes, fill_value=0)

# Calculate the pseudo-inverse of the covariance matrix for Mahalanobis distance
cov_matrix = np.cov(firm_vectors_pd.T)
inv_cov_matrix = pinv(cov_matrix)

output_data = []

# Calculate Mahalanobis distance for each patent
logging.debug(
    "Calculating Mahalanobis distance for each patent compared to its firm-year vector."
)
for publn_nr, patent_vector in patent_vectors_pd.iterrows():
    permno_year = patent_shares_pd[patent_shares_pd["publn_nr"] == publn_nr][
        "permno_year"
    ].iloc[0]
    if permno_year in firm_vectors_pd.index:
        firm_vector = firm_vectors_pd.loc[permno_year].values
        distance = mahalanobis(patent_vector, firm_vector, inv_cov_matrix)
        output_data.append(
            {
                "permno_year": permno_year,
                "publn_nr": publn_nr,
                "mahalanobis_distance": distance,
            }
        )

# Convert the output data to a DataFrame and save
output_df = pd.DataFrame(output_data)
output_df.to_csv("mahalanobis_distances_dask_corrected.csv", index=False)
logging.debug(
    "Calculation completed and output saved. Check mahalanobis_distances_dask_corrected.csv for results."
)
