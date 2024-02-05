import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def optimize_dtypes(df):
    """Optimize data types of a DataFrame to reduce memory usage."""
    for col in df.columns:
        if df[col].dtype == float:
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif df[col].dtype == int:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif df[col].dtype == object:
            df[col] = df[col].astype("category")


# Load CSV files
logging.debug("Loading CSV files.")
patent_shares = pd.read_csv("data/MainfilrARS2_share.csv")
firm_data = pd.read_csv("data/MainfilrARS2.csv")

# Optimize data types
optimize_dtypes(patent_shares)
optimize_dtypes(firm_data)

# Log shapes of loaded dataframes
logging.debug(f"Loaded patent_shares with shape {patent_shares.shape}.")
logging.debug(f"Loaded firm_data with shape {firm_data.shape}.")

# Prepare permno_year column
patent_shares["permno_year"] = (
    patent_shares["permno_adj"].astype(str)
    + "_"
    + patent_shares["publn_year"].astype(str)
)
firm_data["permno_year"] = (
    firm_data["permno_adj"].astype(str) + "_" + firm_data["publn_year"].astype(str)
)

# Process data by year
years = patent_shares["publn_year"].unique()
output_data = []

for year in years:
    logging.debug(f"Processing year: {year}")
    year_patents = patent_shares[patent_shares["publn_year"] == year]
    year_firms = firm_data[firm_data["publn_year"] == year]

    # Calculate shares for firm-year level
    year_firm_shares = (
        year_firms.groupby(["permno_year", "IPC1"]).size().reset_index(name="count")
    )
    total_patents = year_firm_shares.groupby("permno_year")["count"].transform("sum")
    year_firm_shares["share"] = year_firm_shares["count"] / total_patents

    # Pivot tables
    year_patent_vectors = year_patents.pivot(
        index="publn_nr", columns="IPC1", values="share_pat"
    ).fillna(0)
    year_firm_vectors = year_firm_shares.pivot(
        index="permno_year", columns="IPC1", values="share"
    ).fillna(0)

    # Align IPC classes
    all_ipc_classes = year_patent_vectors.columns.union(year_firm_vectors.columns)
    year_patent_vectors = year_patent_vectors.reindex(
        columns=all_ipc_classes, fill_value=0
    )
    year_firm_vectors = year_firm_vectors.reindex(columns=all_ipc_classes, fill_value=0)

    # Calculate pseudo-inverse of covariance matrix for Mahalanobis distance
    cov_matrix = np.cov(year_firm_vectors.values.T)
    inv_cov_matrix = pinv(cov_matrix)

    for publn_nr, patent_vector in year_patent_vectors.iterrows():
        permno_year = year_patents[year_patents["publn_nr"] == publn_nr][
            "permno_year"
        ].iloc[0]
        if permno_year in year_firm_vectors.index:
            firm_vector = year_firm_vectors.loc[permno_year].values
            distance = mahalanobis(patent_vector, firm_vector, inv_cov_matrix)
            output_data.append(
                {
                    "year": year,
                    "permno_year": permno_year,
                    "publn_nr": publn_nr,
                    "mahalanobis_distance": distance,
                }
            )

# Convert the output data to a DataFrame and save
output_df = pd.DataFrame(output_data)
output_df.to_csv("mahalanobis_distances_yearly.csv", index=False)
logging.debug(
    "Calculation completed and output saved. Check mahalanobis_distances_yearly.csv for results."
)
