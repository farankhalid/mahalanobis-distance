
# Mahalanobis Distance Solution

This repository contains a solution for computing Mahalanobis distances between firm vectors and patent vectors. The Mahalanobis distance is a measure of the distance between a point and a distribution. In this context, it is used to quantify the dissimilarity between firms and patents based on their respective IPC (International Patent Classification) shares.
## Installation

To use this solution, make sure you have the required libraries installed. You can install them using pip:
```bash
pip install -r requirements.txt
```
    
## Usage
**1. Importing Libraries**
```python
import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv
from numpy.linalg import matrix_rank
import logging
```
**2. Loading Datasets**
```python
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Loading datasets...")
df_main = pd.read_csv("data/MainfilrARS2.csv")
df_share = pd.read_csv("data/MainfilrARS2_share.csv")
logging.info("Datasets loaded.")
```
**3. Preprocessing**
```python
logging.info("Starting preprocessing...")
df_main["permno_year"] = df_main["permno_adj"].astype(str) + df_main[
    "publn_year"
].astype(str)
df_share["permno_year"] = df_share["permno_adj"].astype(str) + df_share[
    "publn_year"
].astype(str)
logging.info("Preprocessing completed.")
```
**4. Aggregating IPC Shares**
```python
def aggregate_ipc_shares(df, df_type):
    logging.info(f"Aggregating IPC shares for {df_type} data...")
    if df_type == "main":
        # Aggregation process for main dataset
    elif df_type == "share":
        # Aggregation process for share dataset
    logging.info(f"Aggregation complete for {df_type} data.")
    return pivot_df
```
**5. Computing Mahalanobis Distance**
```python
def compute_mahalanobis(df_firm, df_patent, cov_inv):
    logging.info("Computing Mahalanobis distances...")
    distances = []
    for index, row in df_patent.iterrows():
        # Compute Mahalanobis distance for each patent vector
    logging.info("Mahalanobis distances computed.")
    return pd.DataFrame(distances)
```
**6. Yearly Processing**
```python
logging.info("Starting yearly processing...")
# Iterate through each year
for year in years:
    logging.info(f"Processing year: {year}")
    # Process data for each year
logging.info("Yearly processing completed. Saving results...")
# Save results
logging.info("Results saved.")
```
## Output
The Mahalanobis distances for each year are saved in a CSV file named `mahalanobis-distances-*.csv` in the output directory.
## Authors

- [@farankhalid](https://www.github.com/farankhalid)
- [@airaj-raza](https://github.com/airaj-raza)
- [@Tabed23](https://github.com/Tabed23)
