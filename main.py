"""
Kernel PCA

Kernel Types:
    - Radial Basis Function (RBF)
    - Polynomial
    - Cosine Similarity
"""

import pandas as pd
import numpy as np

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
            print(f"Warn: Missing closing price data for Company: {x}")
            unuseable_tickers.append(ticker)
        elif len(x) != row_count:
            print(f"Warn: Got row count {len(x)}, expected: {row_count}")
            print(f"Dropping ticker: {ticker}")
            unuseable_tickers.append(ticker)
        else:
            data[ticker] = np.array(x)
        
    # Drop Unuseable tickers
    data = data.drop(columns=unuseable_tickers)
    
    assertion_msg = 'Failed to remove corrupt data'
    assert any([col in data.columns for col in unuseable_tickers]) == False, assertion_msg

    return data

def main() -> None:
    print("Running Kernel PCA")

    # Import dataset
    df = pd.read_csv('data/raw/all_stocks_5yr.csv')

    # Clean and sanitise data set
    data = prepare_data(df)

    print("End of Kernel PCA!")

if __name__ == "__main__":
    main()