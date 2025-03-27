import pandas as pd
import numpy as np
from scipy.stats import zscore

input_file = 'data/filtered_cancer_all.csv'
output_file = 'data/updated3_dataset.csv'

df = pd.read_csv(input_file)

numeric_cols = df.select_dtypes(include=[np.number]).columns

z_scores = np.abs(df[numeric_cols].apply(zscore))

threshold = 2.5

mask = (z_scores < threshold).all(axis=1)

filtered_df = df[mask]

filtered_df.to_csv(output_file, index=False)