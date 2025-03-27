import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

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
    
    # Stratify split as in the original code
    data_df['stratify_col'] = data_df['Target_ID'] + '_' + data_df['label'].astype(str)
    train_df, temp_df = train_test_split(data_df, test_size=frac_val + frac_test, 
                                         stratify=data_df['stratify_col'], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=frac_test/(frac_val + frac_test), 
                                       stratify=temp_df['stratify_col'], random_state=seed)
    
    # Drop stratify column
    train_df = train_df.drop(columns='stratify_col')
    val_df = val_df.drop(columns='stratify_col')
    test_df = test_df.drop(columns='stratify_col')
    
    return train_df, val_df, test_df, protein_embeddings_df

# Compute Morgan Fingerprint
def compute_morgan_fp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)  # Handle invalid SMILES
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return np.array(fp)

# Load datasets
train_df, val_df, test_df, protein_embeddings_df = load_data('../data', use_small=True)

# Compute fingerprints
train_fps = np.array([compute_morgan_fp(smiles) for smiles in train_df['SMILES']])
val_fps = np.array([compute_morgan_fp(smiles) for smiles in val_df['SMILES']])
test_fps = np.array([compute_morgan_fp(smiles) for smiles in test_df['SMILES']])

# Get protein embeddings
train_protein_emb = protein_embeddings_df.loc[train_df['Target_ID']].values
val_protein_emb = protein_embeddings_df.loc[val_df['Target_ID']].values
test_protein_emb = protein_embeddings_df.loc[test_df['Target_ID']].values

# Concatenate features
train_X = np.hstack((train_fps, train_protein_emb))
val_X = np.hstack((val_fps, val_protein_emb))
test_X = np.hstack((test_fps, test_protein_emb))

train_y = train_df['pchembl_score'].values
val_y = val_df['pchembl_score'].values
test_y = test_df['pchembl_score'].values

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
xgb_model.fit(train_X, train_y, eval_set=[(val_X, val_y)], early_stopping_rounds=10, verbose=False)

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