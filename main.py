"""
Kernel PCA

Kernel Types:
    - Radial Basis Function (RBF)
    - Polynomial
    - Cosine Similarity
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Examine list of companies in the dataset
    tickers = df['Name'].unique()
    print(f"Tickers: {tickers}")

    # Prep Data Table
    first_stock = df[df["Name"] == tickers[0]]
    row_count = len(first_stock)
    print(f"Prepping data table of length: {row_count}")
    data = pd.DataFrame(index=range(row_count), columns=tickers)

    # Build datatable using closing prices
    """
    Aim:
      - Extract stocks with same "row_count" as the reference stock i.e. "AAL"
      - drop stocks with missing closing price data
    """
    unuseable_tickers = []
    for ticker in tickers:
        x = df[ df["Name"] == ticker ]["close"]

        if x.isnull().any():
            # print(f"Warn: Missing closing price data for Company: {x}")
            unuseable_tickers.append(ticker)
        elif len(x) != row_count:
            # print(f"Warn: Got row count {len(x)}, expected: {row_count}")
            # print(f"Dropping ticker: {ticker}")
            unuseable_tickers.append(ticker)
        else:
            data[ticker] = np.array(x)
        
    # Drop Unuseable tickers
    data = data.drop(columns=unuseable_tickers)
    
    assertion_msg = 'Failed to remove corrupt data'
    assert any([col in data.columns for col in unuseable_tickers]) == False, assertion_msg

    return data

def standardise_data(data: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler().fit(data)
    return pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)

KERNEL_DATA = {
    'linear': [0, 0],
    'rbf': [0, 1],
    'poly': [1, 0],
    'cosine': [1, 1]
}

def analyse_kernels(df: pd.DataFrame, n_components = 10) -> pd.DataFrame:
    l = df.shape[0]
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

    for kernel in KERNEL_DATA.keys():
        print(f"Fitting and analysing kernel({kernel}).")
        kpca = KernelPCA(kernel=kernel, n_components=n_components)
        X_kpca = kpca.fit_transform(df)

        # Plot kernel
        x, y = X_kpca[:, 0], X_kpca[:, 1]
        axs[*KERNEL_DATA[kernel]].scatter(x, y, c = np.arange(l)/l)
        axs[*KERNEL_DATA[kernel]].set_title(f"{kernel} kernel")
    plt.show()

def main() -> None:
    print("Running Kernel PCA")

    # Import dataset
    df = pd.read_csv('data/raw/all_stocks_5yr.csv')

    # Clean and sanitise data set
    data = prepare_data(df)
    data_scaled = standardise_data(data)
    print(data_scaled.head())
    print(data_scaled.mean())
    print(data_scaled.std(ddof=1))

    # Fit to kernels
    analyse_kernels(data_scaled)

    print("End of Kernel PCA!")

if __name__ == "__main__":
    main()