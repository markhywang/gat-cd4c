import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn

from model import GraphAttentionNetwork
from utils.dataset import DrugProteinDataset
from utils.helper_functions import *

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train_model(args: argparse.Namespace) -> None:
    # TODO - remove hard-coded specifications for the model
    model = GraphAttentionNetwork(
        333,
        1,
        16,
        args.hidden_size,
        args.num_layers,
        args.num_attn_heads
    ).to(device)
    train_dataset, validation_dataset, test_dataset = load_data(args.data_path, args.seed, args.percent_train,
                                                                args.percent_validation, args.percent_test,
                                                                args.use_small_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False)

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

    for epoch in range(args.max_epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.max_epochs}", leave=True)
        avg_train_loss = run_training_epoch(progress_bar, optimizer, model, loss_func)
        avg_validation_loss = get_validation_metrics(validation_loader, model, loss_func)

        # Step the learning rate scheduler based on mean training loss
        lr_scheduler.step(avg_train_loss)

        print(f"Epoch {epoch + 1}/{args.max_epochs}: Train Loss = {avg_train_loss:.5f}, "
              f"Validation Loss = {avg_validation_loss:.5f}")


def run_training_epoch(progress_bar: tqdm, optimizer: optim.Optimizer, model: nn.Module,
                       loss_func: nn.Module) -> float:
    # Ensure model is in training mode.
    model.train()

    training_loss = []
    for batch_data in progress_bar:
        node_features, edge_features, adjacency_matrix, pchembl_score = [
            x.to(torch.float32).to(device) for x in batch_data
        ]

        pred = model(node_features, edge_features, adjacency_matrix).squeeze(-1)
        loss = loss_func(pred, pchembl_score)
        training_loss.append(float(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = sum(training_loss) / len(training_loss)
    return avg_loss


def get_validation_metrics(validation_loader: DataLoader, model: nn.Module, loss_func: nn.Module) -> float:
    # Ensure model is in evaluation mode.
    model.eval()

    node_features, edge_features, adjacency_matrix, pchembl_scores = [
        x.to(torch.float32).to(device) for x in next(iter(validation_loader))
    ]
    preds = model(node_features, edge_features, adjacency_matrix).squeeze(-1)
    loss = loss_func(preds, pchembl_scores)

    return float(loss)


def load_data(data_path: str, seed: int, percent_train: float, percent_validation: float,
              percent_test: float, use_small_dataset: bool) -> tuple[Dataset, Dataset, Dataset]:
    assert math.isclose(percent_train + percent_validation + percent_test, 1), \
        (f"Sum of percentage splits for training ({percent_train}), validation ({percent_validation}), "
         f"and testing ({percent_test}) don't add up to 1")

    dataset_file = 'filtered_cancer_small.csv' if use_small_dataset else 'filtered_cancer_all.csv'
    data_df = pd.read_csv(f'{data_path}/{dataset_file}')
    protein_embeddings_df = pd.read_csv(f'{data_path}/protein_embeddings.csv', index_col=0)

    # Create a column that combines the protein ID and whether there was a significant drug-protein
    # interaction. This column will be used to split the data to ensure that each dataset has an
    # appropriate balance of proteins and interaction types.
    data_df['stratify_col'] = data_df['Target_ID'] + '_' + data_df['label'].astype(str)

    # Split part of the data into the training dataset (maintaining an equal split of proteins and interactions types).
    train_df, remaining_df = train_test_split(data_df,
                                              test_size=percent_validation + percent_test,
                                              stratify=data_df['stratify_col'],
                                              random_state=seed)

    # Split the remaining data to get the validation and test datasets (maintaining an equal split of proteins and
    # interaction types).
    validation_df, test_df = train_test_split(remaining_df,
                                              test_size=percent_test / (percent_validation + percent_test),
                                              stratify=remaining_df['stratify_col'],
                                              random_state=seed)

    # Remove the stratify column from all the datasets.
    train_df = train_df.drop(columns='stratify_col')
    validation_df = validation_df.drop(columns='stratify_col')
    test_df = test_df.drop(columns='stratify_col')

    train_dataset = DrugProteinDataset(train_df, protein_embeddings_df)
    validation_dataset = DrugProteinDataset(validation_df, protein_embeddings_df)
    test_dataset = DrugProteinDataset(test_df, protein_embeddings_df)

    return train_dataset, validation_dataset, test_dataset


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_small_dataset", action="store_true",
                        help="Whether to use the small dataset instead of the entire dataset")
    parser.add_argument("--batch_size", type=int, required=False, default=64,
                        help="Batch size for data loader")
    parser.add_argument("--stoppage_epochs", type=int, required=False, default=5,
                        help="Number of consecutive epochs with no improvement to validation "
                             "loss before stopping training")
    parser.add_argument("--max_epochs", type=int, required=False, default=128,
                        help="Maximum number of epochs to run training")
    parser.add_argument("--seed", type=int, required=False, default=0,
                        help="The seed used to control any stochastic operations")
    # Data parameters
    parser.add_argument("--data_path", type=str, required=False, default='../data',
                        help="Path to the folder with the data")
    parser.add_argument("--percent_train", type=float, required=False, default=0.7,
                        help="Percentage of data to use for training dataset")
    parser.add_argument("--percent_validation", type=float, required=False, default=0.15,
                        help="Percentage of data to use for validation dataset")
    parser.add_argument("--percent_test", type=float, required=False, default=0.15,
                        help="Percentage of data to use for test dataset")
    # Loss parameters
    parser.add_argument("--huber_beta", type=float, required=False, default=1.0,
                        help="Beta parameter for Huber loss function")

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, required=False, default=1e-4,
                        help="Weight decay for optimizer")
    parser.add_argument("--lr", type=float, required=False, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--scheduler_patience", type=int, required=False, default=10,
                        help="Number of epochs before reducing the learning rate")
    parser.add_argument("--scheduler_factor", type=float, required=False, default=0.5,
                        help="The factor in which the learning rate scheduler adjusts learning rate")

    # Model parameters
    parser.add_argument("--hidden_size", type=int, required=False, default=32,
                        help="The size of embeddings for hidden layers")
    parser.add_argument("--num_layers", type=int, required=False, default=4,
                        help="The number of graph attention layers to use")
    parser.add_argument("--num_attn_heads", type=int, required=False, default=8,
                        help="The number of attention heads to use for every attention block")
    parser.add_argument("--dropout", type=float, required=False, default=0.2,
                        help="Dropout percentage for graph attention layers")
    parser.add_argument("--leaky_relu_slope", type=float, required=False, default=0.2,
                        help="The slope for the Leaky ReLU activation function")

    # Output paramters
    parser.add_argument("--plot_steps", type=int, required=False, default=1,
                        help="Number of batches represented by each data point on the plots")
    return parser


if __name__ == '__main__':
    arg_parser = get_parser()
    training_args = arg_parser.parse_args()

    train_model(training_args)
