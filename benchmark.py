"""Module to benchmark the trained GAT model."""
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from src.model import GraphAttentionNetwork
from src.utils.dataset import DrugProteinDataset
from src.utils.helper_functions import set_seeds, accuracy_func

# Set device
DEVICE = torch.device("cpu")

# Define the thereshold of pCHEMBL in training
PCHEMBL_THRESHOLD = 7.0


def load_test_data(data_path: str, seed: int,
                   frac_validation: float, frac_test: float, use_small_dataset: bool) -> None:
    """Copy the load_data function from train.py but only returns test dataest. """

    # Choose the appropriate file based on dataset size
    dataset_file = '../data/filter_cancer_small.csv' if use_small_dataset else 'filtered_cancer_all.csv'
    data_df = pd.read_csv(f'{data_path}/{dataset_file}')
    protein_embeddings_df = pd.read_csv(f'{data_path}/protein_embeddings.csv', index_col=0)

    # Create a stratification column for balanced splitting
    data_df['stratify_col'] = data_df['Target_ID'] + "_" + data_df['label'].astype(str)

    _, remaining_df = train_test_split(data_df,
                                       test_size=frac_validation + frac_test,
                                       stratify=data_df['stratify_col'],
                                       random_state=seed)

    _, test_df = train_test_split(remaining_df,
                                  test_size=frac_test / (frac_validation + frac_test),
                                  stratify=remaining_df['stratify_col'],
                                  random_state=seed)

    test_df = test_df.drop(columns='stratify_col')

    test_dataset = DrugProteinDataset(test_df, protein_embeddings_df)
    return test_dataset


def load_model(device: torch.device, model_path: str, in_features: int = 333, out_features: int = 1,
               num_edge_features: int = 16, hidden_size: int = 64, num_layers: int = 3, num_attn_heads: int = 4,
               dropout: float = 0.2, pooling_dropout: float = 0.2, pooling_dim: int = 128) -> None:
    """
    Initializes the model with the given parameters, loads saved weights,
    and returns the model in evaluation mode.
    """
    model = GraphAttentionNetwork(
        device,
        in_features,
        out_features,
        num_edge_features,
        hidden_size,
        num_layers,
        num_attn_heads,
        dropout,
        pooling_dropout,
        pooling_dim
    ).to(torch.float32).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    return model


def evaluate_model(model: nn.Module, test_loader: DataLoader, huber_beta: float) -> tuple:
    """
    Evaluates the model on the test dataset, computes loss and classification metrics.
    """
    model.eval()
    criterion = nn.SmoothL1Loss(beta=huber_beta)
    cum_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            node_features, edge_features, adjacency_matrix, pchembl_scores = [
                x.to(torch.float32).to(DEVICE) for x in batch
            ]
            preds = model(node_features, edge_features, adjacency_matrix).squeeze(-1)
            loss = criterion(preds, pchembl_scores)
            cum_loss += loss.item() * preds.size(0)
            all_labels.extend(pchembl_scores.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = cum_loss / len(test_loader.dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Convert continuous predictions to binary based on threshold
    binary_preds = (all_preds >= PCHEMBL_THRESHOLD).astype(int)
    binary_labels = (all_labels >= PCHEMBL_THRESHOLD).astype(int)

    # Compute classification metrics
    accuracy = accuracy_func(all_labels, all_preds, 1.0) / len(all_labels)
    f1 = f1_score(binary_labels, binary_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    precision = precision_score(binary_labels, binary_preds)
    recall = recall_score(binary_labels, binary_preds)
    try:
        auc_roc = roc_auc_score(binary_labels, all_preds)
    except ValueError:
        auc_roc = float('nan')

    metrics = {
        "avg_loss": avg_loss,
        "accuracy": accuracy,
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc_roc": auc_roc,
    }

    return metrics, binary_labels, all_preds


def main() -> None:
    """Add training arguments and run the benchmark on the model."""

    # Must match training args
    args_dict = {
        "use_small_dataset": False,
        "batch_size": 64,
        "stoppage_epochs": 64,
        "max_epochs": 512,
        "seed": 0,
        "data_path": "data",
        "frac_train": 0.8,
        "frac_validation": 0.1,
        "frac_test": 0.1,
        "huber_beta": 0.5,
        "weight_decay": 1e-3,
        "lr": 3e-4,
        "scheduler_patience": 10,
        "scheduler_factor": 0.5,
        "hidden_size": 96,
        "num_layers": 8,
        "num_attn_heads": 6,
        "dropout": 0.2,
        "pooling_dropout": 0.1,
        "pooling_dim": 96,
    }

    args = argparse.Namespace(**args_dict)

    # Set seeds for reproducibility
    set_seeds(seed=0)

    model = load_model(
        DEVICE, "models/model.pth",
        in_features=349,
        out_features=1,
        num_edge_features=16,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attn_heads=args.num_attn_heads,
        dropout=args.dropout,
        pooling_dropout=args.pooling_dropout,
        pooling_dim=args.pooling_dim
    )

    # Load test dataset and create DataLoader
    test_dataset = load_test_data(
        args.data_path,
        args.seed,
        frac_validation=args.frac_validation,
        frac_test=args.frac_test,
        use_small_dataset=args.use_small_dataset
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluate the model
    metrics, _, _ = evaluate_model(model, test_loader, args.huber_beta)

    print("\n--- Evaluation Metrics ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.5f}")


if __name__ == "__main__":
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
            'DataStructs.ConvertToNumpyArray',
            'math',
            'torch',
            'torch.nn',
            'torch.nn.functional',
            'torch.utils.data',
            'argparse',
            'src.model',
            'src.utils.dataset',
            'src.utils.helper_functions'
        ],
        'disable': ['R0914', 'E1101', 'R0913', 'R0902', 'E9959'],  
        # R0914 for local variable, E1101 for attributes for imported modules
        # R0913 for arguments, R0902 for instance attributes in class
        # E9959 for instance annotation
        'allowed-io': ['main', 'load_model'],
        'max-line-length': 120,
    })

    main()
