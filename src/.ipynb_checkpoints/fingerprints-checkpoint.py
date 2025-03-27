import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem  # still needed for MolFromSmiles
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

# Define accuracy function (mimicking the GAT model's approach)
def compute_accuracy(preds, true_labels, threshold=7.0):
    preds_binary = (preds >= threshold).astype(int)
    true_binary = (true_labels >= threshold).astype(int)
    correct = (preds_binary == true_binary).sum()
    return correct / len(preds)

# Load data (adapted from provided load_data function)
def load_data(data_path, seed=42, frac_train=0.8, frac_val=0.1, frac_test=0.1, use_small=False):
    dataset_file = 'filtered_cancer_small.csv' if use_small else 'filtered_cancer_all.csv'
    data_df = pd.read_csv(f'{data_path}/{dataset_file}')
    protein_embeddings_df = pd.read_csv(f'{data_path}/protein_embeddings.csv', index_col=0)
    
    # Create binary label based on the threshold 7.0 (using the pChEMBL_Value column)
    data_df['binary_label'] = (data_df['pChEMBL_Value'] >= 7).astype(int)
    # Construct a stratification column by combining Target_ID and binary label
    data_df['stratify_col'] = data_df['Target_ID'].astype(str) + '_' + data_df['binary_label'].astype(str)
    
    train_df, temp_df = train_test_split(
        data_df, 
        test_size=frac_val + frac_test, 
        stratify=data_df['stratify_col'], 
        random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=frac_test/(frac_val + frac_test), 
        stratify=temp_df['stratify_col'], 
        random_state=seed
    )
    
    # Drop the extra columns used for stratification
    train_df = train_df.drop(columns=['stratify_col', 'binary_label'])
    val_df = val_df.drop(columns=['stratify_col', 'binary_label'])
    test_df = test_df.drop(columns=['stratify_col', 'binary_label'])
    
    return train_df, val_df, test_df, protein_embeddings_df

# Compute Morgan Fingerprint using the new rdFingerprintGenerator API
def compute_morgan_fp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    # Use fpSize instead of numBits to match the expected parameter name.
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

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
train_X = np.hstack((train_fps, train_protein_emb))
val_X = np.hstack((val_fps, val_protein_emb))
test_X = np.hstack((test_fps, test_protein_emb))

# Use the correct target column from the dataset
train_y = train_df['pChEMBL_Value'].values
val_y = val_df['pChEMBL_Value'].values
test_y = test_df['pChEMBL_Value'].values

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
xgb_model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)

# Predict and evaluate
train_preds = xgb_model.predict(train_X)
val_preds = xgb_model.predict(val_X)
test_preds = xgb_model.predict(test_X)

# Compute regression metrics
train_mse = mean_squared_error(train_y, train_preds)
val_mse = mean_squared_error(val_y, val_preds)
test_mse = mean_squared_error(test_y, test_preds)

train_r2 = r2_score(train_y, train_preds)
val_r2 = r2_score(val_y, val_preds)
test_r2 = r2_score(test_y, test_preds)

# Compute accuracy (threshold-based)
train_acc = compute_accuracy(train_preds, train_y, threshold=7.0)
val_acc = compute_accuracy(val_preds, val_y, threshold=7.0)
test_acc = compute_accuracy(test_preds, test_y, threshold=7.0)

# Print results
print("XGBoost Model Performance:")
print(f"Train MSE: {train_mse:.5f}, Val MSE: {val_mse:.5f}, Test MSE: {test_mse:.5f}")
print(f"Train R²: {train_r2:.5f}, Val R²: {val_r2:.5f}, Test R²: {test_r2:.5f}")
print(f"Train Accuracy: {train_acc:.5f}, Val Accuracy: {val_acc:.5f}, Test Accuracy: {test_acc:.5f}")
