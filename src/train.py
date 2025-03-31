"""
Module for training the Graph Attention Network using custom-set hyperparameters.
This module 
    - retrieves configurations from parser
    - Trains model
    - Evaluate the model's training and validation metrics
"""

import math
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch import nn

from model import GraphAttentionNetwork
from utils.dataset import DrugProteinDataset
from utils.helper_functions import set_seeds, count_model_params, plot_loss_curves, accuracy_func, mse_func, mae_func

# Set device default as CPU (can change later when training model)
device = torch.device("cpu")


def train_model(args: argparse.Namespace, m_device: torch.device = device) -> None:
    """
    Main training loop used to train the Graph Attention Network.
    Note that after eawch training epoch, training and validation metrics are outputted.
    """
    set_seeds()

    # Set device to device
    global device
    device = m_device

    model = GraphAttentionNetwork(
        device,
        in_features=349,
        out_features=1,
        num_edge_features=16,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attn_heads=args.num_attn_heads,
        dropout=args.dropout,
        pooling_dropout=args.pooling_dropout,
        pooling_dim=args.pooling_dim
    ).to(torch.float32).to(device)

    print(f'Model parameters: {count_model_params(model)}')

    datasets = load_data(args.data_path, args.seed, args.frac_train,
                         args.frac_validation, args.frac_test,
                         args.use_small_dataset)
    train_dataset, validation_dataset = datasets[0], datasets[1]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    loss_func = nn.SmoothL1Loss(beta=args.huber_beta)  # Initialize the Huber loss function.
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.scheduler_factor,
        patience=args.scheduler_patience
    )

    best_validation_loss = float('inf')
    # Initialize a counter that tracks the number of consecutive epochs without an improvement in validation loss.
    no_validation_loss_improvement = 0
    metrics_df = pd.DataFrame(columns=['train_loss', 'validation_loss',
                                       'train_acc', 'validation_acc',
                                       'train_mse', 'validation_mse',
                                       'train_mae', 'validation_mae'], 
                              index=range(args.max_epochs))

    for epoch in range(args.max_epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.max_epochs}", leave=True)
        avg_train_metrics = run_training_epoch(progress_bar, optimizer, model, loss_func)
        avg_val_metrics = get_validation_metrics(validation_loader, model, loss_func)

        # Unpack all metrics
        avg_train_loss, avg_train_acc, avg_train_mse, avg_train_mae = avg_train_metrics
        avg_val_loss, avg_val_acc, avg_val_mse, avg_val_mae = avg_val_metrics

        lr_scheduler.step(avg_train_loss)

        # Store all metrics
        metrics_df.at[epoch, 'train_loss'] = avg_train_loss
        metrics_df.at[epoch, 'validation_loss'] = avg_val_loss
        metrics_df.at[epoch, 'train_acc'] = avg_train_acc
        metrics_df.at[epoch, 'validation_acc'] = avg_val_acc
        metrics_df.at[epoch, 'train_mse'] = avg_train_mse
        metrics_df.at[epoch, 'validation_mse'] = avg_val_mse
        metrics_df.at[epoch, 'train_mae'] = avg_train_mae
        metrics_df.at[epoch, 'validation_mae'] = avg_val_mae

        print(f"Epoch {epoch + 1}/{args.max_epochs}: \n"
              f"Train Loss = {avg_train_loss:.5f}, "
              f"Train MSE = {avg_train_mse:.5f}, "
              f"Train MAE = {avg_train_mae:.5f}, "
              f"Train Acc = {avg_train_acc:.5f}\n"
              f"Val Loss = {avg_val_loss:.5f}, "
              f"Val MSE = {avg_val_mse:.5f}, "
              f"Val MAE = {avg_val_mae:.5f}, "
              f"Val Acc = {avg_val_acc:.5f}")

        if avg_val_loss < best_validation_loss:
            best_validation_loss = avg_val_loss
            no_validation_loss_improvement = 0
            torch.save(model.state_dict(), '../models/model.pth')
        else:
            no_validation_loss_improvement += 1
            if no_validation_loss_improvement == args.stoppage_epochs:
                break

    plot_loss_curves(metrics_df)


def run_training_epoch(progress_bar: tqdm, optimizer: optim.Optimizer, model: nn.Module,
                       loss_func: nn.Module) -> tuple[float, float, float, float]:
    """
    Method for running one single training epoch.
    Returns the training loss and error metrics.
    """
    model.train()
    cum_training_samples = 0
    cum_training_loss = 0
    cum_training_acc_preds = 0
    cum_training_mse = 0
    cum_training_mae = 0

    for batch_data in progress_bar:
        node_features, edge_features, adjacency_matrix, pchembl_score = [
            x.to(torch.float32).to(device) for x in batch_data
        ]

        preds = model(node_features, edge_features, adjacency_matrix).squeeze(-1)
        loss = loss_func(preds, pchembl_score)

        cum_training_samples += preds.shape[0]
        cum_training_loss += loss.item() * preds.shape[0]
        cum_training_acc_preds += accuracy_func(preds, pchembl_score, threshold=1.0)
        cum_training_mse += mse_func(preds, pchembl_score) * preds.shape[0]
        cum_training_mae += mae_func(preds, pchembl_score) * preds.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = cum_training_loss / cum_training_samples
    avg_acc = cum_training_acc_preds / cum_training_samples
    avg_mse = cum_training_mse / cum_training_samples
    avg_mae = cum_training_mae / cum_training_samples
    return avg_loss, avg_acc, avg_mse, avg_mae


def get_validation_metrics(validation_loader: DataLoader, model: nn.Module,
                           loss_func: nn.Module) -> tuple[float, float, float, float]:
    """
    This method computes and returns validation loss and error metrics.
    """
    model.eval()
    cum_validation_samples = 0
    cum_validation_loss = 0
    cum_validation_acc_preds = 0
    cum_validation_mse = 0
    cum_validation_mae = 0

    for batch in validation_loader:
        node_features, edge_features, adjacency_matrix, pchembl_scores = [
            x.to(torch.float32).to(device) for x in batch
        ]
        preds = model(node_features, edge_features, adjacency_matrix).squeeze(-1)
        loss = loss_func(preds, pchembl_scores).item()
        acc = accuracy_func(preds, pchembl_scores, threshold=1.0)
        mse = mse_func(preds, pchembl_scores)
        mae = mae_func(preds, pchembl_scores)

        cum_validation_samples += preds.shape[0]
        cum_validation_loss += loss * preds.shape[0]
        cum_validation_acc_preds += acc
        cum_validation_mse += mse * preds.shape[0]
        cum_validation_mae += mae * preds.shape[0]

    avg_loss = cum_validation_loss / cum_validation_samples
    avg_acc = cum_validation_acc_preds / cum_validation_samples
    avg_mse = cum_validation_mse / cum_validation_samples
    avg_mae = cum_validation_mae / cum_validation_samples
    return avg_loss, avg_acc, avg_mse, avg_mae


# Rest of the code (load_data and get_parser) remains unchanged
def load_data(data_path: str, seed: int, frac_train: float, frac_validation: float,
              frac_test: float, use_small_dataset: bool) -> tuple[Dataset, Dataset, Dataset]:
    """
    Loads the CandidateDrug4Cancer dataset and then initializes the train-validation-test splits.
    Returns the processed train, validation, and test datasets.
    """
    assert math.isclose(frac_train + frac_validation + frac_test, 1), \
        (f"Sum of training ({frac_train}), validation ({frac_validation}), "
         f"and testing ({frac_test}) splits don't add up to 1")

    dataset_file = 'filtered_cancer_small.csv' if use_small_dataset else 'filtered_cancer_all.csv'
    data_df = pd.read_csv(f'{data_path}/{dataset_file}')
    protein_embeddings_df = pd.read_csv(f'{data_path}/protein_embeddings.csv', index_col=0)

    data_df['stratify_col'] = data_df['Target_ID'] + '_' + data_df['label'].astype(str)
    train_df, remaining_df = train_test_split(data_df,
                                              test_size=frac_validation + frac_test,
                                              stratify=data_df['stratify_col'],
                                              random_state=seed)
    validation_df, test_df = train_test_split(remaining_df,
                                              test_size=frac_test / (frac_validation + frac_test),
                                              stratify=remaining_df['stratify_col'],
                                              random_state=seed)

    train_df = train_df.drop(columns='stratify_col')
    validation_df = validation_df.drop(columns='stratify_col')
    test_df = test_df.drop(columns='stratify_col')

    train_dataset = DrugProteinDataset(train_df, protein_embeddings_df)
    validation_dataset = DrugProteinDataset(validation_df, protein_embeddings_df)
    test_dataset = DrugProteinDataset(test_df, protein_embeddings_df)

    return train_dataset, validation_dataset, test_dataset


def get_parser() -> argparse.ArgumentParser:
    """
    Adds all necessary configurations (e.g. hyperparameters) to argparser.
    Returns the argparse.ArgumentParser instance to be using during training.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_small_dataset", action="store_true",
                        help="Whether to use the small dataset instead of the entire dataset")
    parser.add_argument("--batch_size", type=int, required=False, default=64,
                        help="Batch size for data loader")
    parser.add_argument("--stoppage_epochs", type=int, required=False, default=10,
                        help="Number of consecutive epochs with no improvement to validation "
                             "loss before stopping training")
    parser.add_argument("--max_epochs", type=int, required=False, default=128,
                        help="Maximum number of epochs to run training")
    parser.add_argument("--seed", type=int, required=False, default=42,
                        help="The seed used to control any stochastic operations")

    # Data parameters
    parser.add_argument("--data_path", type=str, required=False, default='../data',
                        help="Path to the folder with the data")
    parser.add_argument("--frac_train", type=float, required=False, default=0.7,
                        help="Fraction of data to use for training dataset")
    parser.add_argument("--frac_validation", type=float, required=False, default=0.15,
                        help="Fraction of data to use for validation dataset")
    parser.add_argument("--frac_test", type=float, required=False, default=0.15,
                        help="Fraction of data to use for test dataset")

    # Loss parameters
    parser.add_argument("--huber_beta", type=float, required=False, default=1.0,
                        help="Beta parameter for Huber loss function")

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, required=False, default=1e-3,
                        help="Weight decay for optimizer")
    parser.add_argument("--lr", type=float, required=False, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--momentum", type=float, required=False, default=0.9,
                        help="Momentum for appropriate optimizers (e.g. SGD)")
    parser.add_argument("--scheduler_patience", type=int, required=False, default=10,
                        help="Number of epochs before reducing the learning rate")
    parser.add_argument("--scheduler_factor", type=float, required=False, default=0.5,
                        help="The factor in which the learning rate scheduler adjusts learning rate")

    # Model parameters
    parser.add_argument("--hidden_size", type=int, required=False, default=64,
                        help="The size of embeddings for hidden layers")
    parser.add_argument("--num_layers", type=int, required=False, default=3,
                        help="The number of graph attention layers to use")
    parser.add_argument("--num_attn_heads", type=int, required=False, default=4,
                        help="The number of attention heads to use for every attention block")
    parser.add_argument("--dropout", type=float, required=False, default=0.2,
                        help="Dropout percentage for graph attention layers")
    parser.add_argument("--pooling_dropout", type=float, required=False, default=0.2,
                        help="Dropout percentage for graph attention layers")
    parser.add_argument("--leaky_relu_slope", type=float, required=False, default=0.2,
                        help="The slope for the Leaky ReLU activation function")
    parser.add_argument("--pooling_dim", type=int, required=False, default=128,
                        help="Pooling dimension")

    return parser


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': [
            'argparse',
            'pandas',
            'sklearn.model_selection',
            'tqdm',
            'math',
            'torch',
            'torch.utils.data',
            'torch.optim',
            'torch.nn',
            'model',
            'utils.dataset',
            'utils.helper_functions'
        ],
        'disable': ['C9103', 'R0913', 'R0914', 'E9997', 'E1101', 'E9992'],
        'allowed-io': ['train_model'],
        'max-line-length': 120,
    })

    arg_parser = get_parser()
    training_args = arg_parser.parse_args()

    train_model(training_args)
