"""
Module for training and evaluating an XGBoost model on cancer data.
This module loads data, computes molecular fingerprints and protein embeddings,
trains an XGBoost regression model, and evaluates its performance.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
import rdkit.Chem.rdFingerprintGenerator as fpGen
from rdkit import DataStructs
import sklearn.model_selection as ms
from sklearn import metrics
import xgboost as xgb


def compute_accuracy(preds: np.ndarray, true_labels: np.ndarray, threshold: float = 1.0) -> float:
    """Compute the fraction of predictions within a threshold of the true labels."""
    correct = (np.abs(true_labels - preds) < threshold).sum()
    return correct / len(preds)


def load_data(data_path: str, seed: int = 42, frac_val: float = 0.1,
              frac_test: float = 0.1, use_small: bool = False) -> tuple:
    """
    Load and split the dataset into training, validation, and test sets along with protein embeddings.

    Parameters:
        data_path: Path to the directory containing data files.
        seed: Random seed for reproducibility.
        frac_val: Fraction of data to use for validation.
        frac_test: Fraction of data to use for testing.
        use_small: Whether to use a smaller dataset variant.

    Returns:
        A tuple containing (train_df, val_df, test_df, protein_embeddings_df).
    """
    dataset_file = 'filtered_cancer_small.csv' if use_small else 'filtered_cancer_all.csv'
    data_df = pd.read_csv(f'{data_path}/{dataset_file}')
    protein_embeddings_df = pd.read_csv(f'{data_path}/protein_embeddings.csv', index_col=0)

    # Create label based on the pChEMBL_Value column
    data_df['label'] = data_df['pChEMBL_Value'].astype(float)

    # Split the data
    train_df, temp_df = ms.train_test_split(
        data_df,
        test_size=frac_val + frac_test,
        random_state=seed
    )
    val_df, test_df = ms.train_test_split(
        temp_df,
        test_size=frac_test / (frac_val + frac_test),  # added spaces around the operator
        random_state=seed
    )

    # Drop the label column if no longer needed
    train_df = train_df.drop(columns=['label'])
    val_df = val_df.drop(columns=['label'])
    test_df = test_df.drop(columns=['label'])

    return train_df, val_df, test_df, protein_embeddings_df


def compute_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Compute the Morgan fingerprint for a given SMILES string using rdFingerprintGenerator.

    Parameters:
        smiles: A SMILES string representing the molecule.
        radius: The radius parameter for the Morgan fingerprint.
        n_bits: The size of the fingerprint.

    Returns:
        A numpy array representing the fingerprint.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    generator = fpGen.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def main() -> None:
    """Main function to train the XGBoost model and evaluate its performance."""
    # Load datasets
    train_df, val_df, test_df, protein_embeddings_df = load_data('../data', use_small=False)

    # Compute fingerprints
    train_fps = np.array([compute_morgan_fp(smiles) for smiles in train_df['smiles']])
    val_fps = np.array([compute_morgan_fp(smiles) for smiles in val_df['smiles']])
    test_fps = np.array([compute_morgan_fp(smiles) for smiles in test_df['smiles']])

    # Get protein embeddings
    train_protein_emb = protein_embeddings_df.loc[train_df['Target_ID']].values
    val_protein_emb = protein_embeddings_df.loc[val_df['Target_ID']].values
    test_protein_emb = protein_embeddings_df.loc[test_df['Target_ID']].values

    # Concatenate features
    train_x = np.hstack((train_fps, train_protein_emb))
    val_x = np.hstack((val_fps, val_protein_emb))
    test_x = np.hstack((test_fps, test_protein_emb))

    # Use the correct target column from the dataset
    train_y = train_df['pChEMBL_Value'].values
    val_y = val_df['pChEMBL_Value'].values
    test_y = test_df['pChEMBL_Value'].values

    # Train XGBoost model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    xgb_model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)

    # Predict and evaluate
    train_preds = xgb_model.predict(train_x)
    val_preds = xgb_model.predict(val_x)
    test_preds = xgb_model.predict(test_x)

    # Compute regression metrics
    train_mse = metrics.mean_squared_error(train_y, train_preds)
    val_mse = metrics.mean_squared_error(val_y, val_preds)
    test_mse = metrics.mean_squared_error(test_y, test_preds)

    train_r2 = metrics.r2_score(train_y, train_preds)
    val_r2 = metrics.r2_score(val_y, val_preds)
    test_r2 = metrics.r2_score(test_y, test_preds)

    # Compute accuracy (threshold-based)
    train_acc = compute_accuracy(train_preds, train_y, threshold=1.0)
    val_acc = compute_accuracy(val_preds, val_y, threshold=1.0)
    test_acc = compute_accuracy(test_preds, test_y, threshold=1.0)

    # Output results (IO functions are flagged; disable warning with noqa)
    print("XGBoost Model Performance:")
    print(f"Train MSE: {train_mse:.5f}, Val MSE: {val_mse:.5f}, Test MSE: {test_mse:.5f}")
    print(f"Train R²: {train_r2:.5f}, Val R²: {val_r2:.5f}, Test R²: {test_r2:.5f}")
    print(f"Train Accuracy: {train_acc:.5f}, Val Accuracy: {val_acc:.5f}, Test Accuracy: {test_acc:.5f}")


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': [
            'numpy',
            'pandas',
            'sklearn.model_selection',
            'sklearn.metrics',
            'rdkit',
            'xgboost',
            'rdkit.Chem.rdFingerprintGenerator',
            'Chem.MolFromSmiles',
            'DataStructs.ConvertToNumpyArray'
        ],
        'disable': ['R0914', 'E1101'],  # R0914 for local variable, E1101 for attributes for imported modules
        'allowed-io': ['main'],
        'max-line-length': 120,
    })

    main()
