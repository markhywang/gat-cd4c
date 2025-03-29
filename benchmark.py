import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score, 
    roc_curve
)

from src.model import GraphAttentionNetwork
from src.utils.dataset import DrugProteinDataset
from src.utils.helper_functions import set_seeds

# Set device
device = torch.device("cpu")

# Define the thereshold of pCHEMBL in training
pCHEMBL_THRESHOLD = 7.0

def load_test_data(data_path: str, seed: int, frac_train: float, frac_validation: float,
                  frac_test: float, use_small_dataset: bool):
    """
    Copy the load_data function from train.py but only returns test dataest.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Choose the appropriate file based on dataset size
    dataset_file = '../data/filter_cancer_small.csv' if use_small_dataset else 'filtered_cancer_all.csv'
    data_df = pd.read_csv(f'{data_path}/{dataset_file}')
    protein_embeddings_df = pd.read_csv(f'{data_path}/protein_embeddings.csv', index_col=0)

    # Create a stratification column for balanced splitting
    data_df['stratify_col'] = data_df['Target_ID'] + "_" + data_df['label'].astype(str)

    traing_df, remaining_df = train_test_split(data_df,
                                               test_size=frac_validation + frac_test,
                                               stratify=data_df['stratify_col'],
                                               random_state=seed)
    
    validation_df, test_df = train_test_split(remaining_df,
                                              test_size=frac_test / (frac_validation + frac_test),
                                              stratify=remaining_df['stratify_col'],
                                              random_state=seed)
    
    test_df = test_df.drop(columns='stratify_col')

    test_dataset = DrugProteinDataset(test_df, protein_embeddings_df)
    return test_dataset


def load_model(device, model_path, in_features=333, out_features=1, num_edge_features=16,
               hidden_size=64, num_layers=3, num_attn_heads=4, dropout=0.2, pooling_dropout=0.2, pooling_dim=128):
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


def evaluate_model(model: nn.Module, test_loader: DataLoader, deveice: torch.device, huber_beta: float) -> tuple:
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
                x.to(torch.float32).to(device) for x in batch
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
    binary_preds = (all_preds >= pCHEMBL_THRESHOLD).astype(int)
    binary_labels = (all_labels >= pCHEMBL_THRESHOLD).astype(int)

    # Compute classification metrics
    accuracy = accuracy_score(binary_labels, binary_preds)
    f1 = f1_score(binary_labels, binary_preds)
    precision = precision_score(binary_labels, binary_preds)
    recall = recall_score(binary_labels, binary_preds)
    try:
        auc_roc = roc_auc_score(binary_labels, all_preds)
    except ValueError:
        auc_roc = float('nan')
    
    metrics = {
        "avg_loss": avg_loss,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc_roc": auc_roc,
    }

    return metrics, binary_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Graph Attention Network model.")
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data folder")
    parser.add_argument("--model_path", type=str, default="models/model.pth", help="Path to the saved model weights")
    parser.add_argument("--use_small_dataset", action="store_true", help="Use small dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument("--frac_train", type=float, default=0.7, help="Fraction of data for training split")
    parser.add_argument("--frac_validation", type=float, default=0.15, help="Fraction of data for validation split")
    parser.add_argument("--frac_test", type=float, default=0.15, help="Fraction of data for test split")
    parser.add_argument("--huber_beta", type=float, default=1.0, help="Beta parameter for SmoothL1Loss")
    
    # Model parameters (should match those used during training)
    parser.add_argument("--in_features", type=int, default=349, help="Input feature dimension")
    parser.add_argument("--out_features", type=int, default=1, help="Output feature dimension")
    parser.add_argument("--num_edge_features", type=int, default=16, help="Edge feature dimension")
    parser.add_argument("--hidden_size", type=int, default=96, help="Hidden layer size")
    parser.add_argument("--num_layers", type=int, default=7, help="Number of GAT layers")
    parser.add_argument("--num_attn_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--pooling_dropout", type=float, default=0.2, help="Dropout rate for global pooling")
    parser.add_argument("--pooling_dim", type=float, default=96, help="Dimension for global pooling")
    args = parser.parse_args()

    # Set seeds for reproducibility
    set_seeds()

    model = load_model(
        device, args.model_path,
        in_features=args.in_features,
        out_features=args.out_features,
        num_edge_features=args.num_edge_features,
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
        frac_train=args.frac_train,
        frac_validation=args.frac_validation,
        frac_test=args.frac_test,
        use_small_dataset=args.use_small_dataset
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate the model
    metrics, binary_labels, all_preds = evaluate_model(model, test_loader, device, args.huber_beta)
    
    print("\n--- Evaluation Metrics ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.5f}")


if __name__ == "__main__":
    main()