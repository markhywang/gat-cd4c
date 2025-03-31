"""Python file to perform data cleaning and filter."""

import pandas as pd
import numpy as np
from scipy.stats import zscore


def filter_dataset() -> None:
    """Remove all the outliers in the dataset with z-score greater than or equal to 2.5"""
    input_file = 'data/filtered_cancer_all.csv'
    output_file = 'data/updated3_dataset.csv'

    df = pd.read_csv(input_file)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    z_scores = np.abs(df[numeric_cols].apply(zscore))

    threshold = 2.5

    mask = (z_scores < threshold).all(axis=1)

    filtered_df = df[mask]

    filtered_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['numpy', 'pandas', 'scipy.stats'],
        'disable': [],  
        'allowed-io': [],
        'max-line-length': 120,
    })
