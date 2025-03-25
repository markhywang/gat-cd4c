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
device = torch.device("cpu")

print(f"Using device: {device}")

def train_model(args: argparse.Namespace, m_device = device) -> None:
    # TODO - remove hard-coded specifications for the model
    # Set the same seed every time for deterministic behaviour.
    set_seeds()

    # Set device to device
    global device
    device = m_device

    model = GraphAttentionNetwork(
        device,
        350,
        1,
        16,
        args.hidden_size,
        args.num_layers,
        args.num_attn_heads,
        args.dropout,
        args.pooling_dim
    ).to(torch.float32).to(device)

    print(f'Model parameters: {count_model_params(model)}')

    train_dataset, validation_dataset, test_dataset = load_data(args.data_path, args.seed, args.frac_train,
                                                                args.frac_validation, args.frac_test,
                                                                args.use_small_dataset)
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
    metrics_df = pd.DataFrame(columns=['train_loss', 'validation_loss'], index=range(args.max_epochs))

    for epoch in range(args.max_epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.max_epochs}", leave=True)
        avg_train_loss, avg_train_acc = run_training_epoch(progress_bar, optimizer, model, loss_func)
        avg_validation_loss, avg_validation_acc = get_validation_metrics(validation_loader, model, loss_func)

        # Step the learning rate scheduler based on mean training loss
        lr_scheduler.step(avg_train_loss)

        # Save the training and validation metrics in a dataframe.
        metrics_df.at[epoch, 'train_loss'] = avg_train_loss
        metrics_df.at[epoch, 'validation_loss'] = avg_validation_loss
        metrics_df.at[epoch, 'train_acc'] = avg_train_acc
        metrics_df.at[epoch, 'validation_acc'] = avg_validation_acc

        print(f"Epoch {epoch + 1}/{args.max_epochs}: Train Loss = {avg_train_loss:.5f}, "
              f"Validation Loss = {avg_validation_loss:.5f}, Train Accuracy = {avg_train_acc:.5f}, "
              f"Validation Accuracy = {avg_validation_acc:.5f}")

        if avg_validation_loss < best_validation_loss:
            # Update the best validation loss seen so far.
            best_validation_loss = avg_validation_loss
            no_validation_loss_improvement = 0
            # Save the weights of the model which gives the lowest validation loss so far.
            torch.save(model.state_dict(), '../models/model.pth')
        else:
            no_validation_loss_improvement += 1
            # If the validation hasn't improved for a certain number of epochs, end training.
            if no_validation_loss_improvement == args.stoppage_epochs:
                break

    plot_loss_curves(metrics_df)


def run_training_epoch(progress_bar: tqdm, optimizer: optim.Optimizer, model: nn.Module,
                       loss_func: nn.Module) -> tuple[float, float]:
    # Ensure model is in training mode.
    model.train()

    cum_training_samples = 0
    cum_training_loss = 0
    cum_training_acc_preds = 0

    for batch_data in progress_bar:
        node_features, edge_features, adjacency_matrix, pchembl_score = [
            x.to(torch.float32).to(device) for x in batch_data
        ]

        preds = model(node_features, edge_features, adjacency_matrix).squeeze(-1)
        loss = loss_func(preds, pchembl_score)

        cum_training_samples += preds.shape[0]
        cum_training_loss += loss.item() * preds.shape[0]
        # Threshold of 7.0 is chosen based on the CD4C paper's claim that a pChEMBL score >= 7.0 signifies a
        # significant drug-protein interaction.
        cum_training_acc_preds += accuracy_func(preds, pchembl_score, threshold=7.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = cum_training_loss / cum_training_samples
    avg_acc = cum_training_acc_preds / cum_training_samples
    return avg_loss, avg_acc


def get_validation_metrics(validation_loader: DataLoader, model: nn.Module, loss_func: nn.Module) \
        -> tuple[float, float]:
    # Ensure model is in evaluation mode.
    model.eval()

    cum_validation_samples = 0
    cum_validation_loss = 0
    cum_validation_acc_preds = 0

    for batch in validation_loader:
        node_features, edge_features, adjacency_matrix, pchembl_scores = [
            x.to(torch.float32).to(device) for x in batch
        ]
        preds = model(node_features, edge_features, adjacency_matrix).squeeze(-1)
        loss = loss_func(preds, pchembl_scores).item()
        # Threshold of 7.0 is chosen based on the CD4C paper's claim that a pChEMBL score >= 7.0 signifies a
        # significant drug-protein interaction.
        acc = accuracy_func(preds, pchembl_scores, threshold=7.0)

        cum_validation_samples += preds.shape[0]
        cum_validation_loss += loss * preds.shape[0]
        cum_validation_acc_preds += acc

    # TODO - only plot preds vs labels for the final training epoch
    # plot_preds_vs_labels(preds, pchembl_scores)

    return cum_validation_loss / cum_validation_samples, cum_validation_acc_preds / cum_validation_samples


def load_data(data_path: str, seed: int, frac_train: float, frac_validation: float,
              frac_test: float, use_small_dataset: bool) -> tuple[Dataset, Dataset, Dataset]:
    assert math.isclose(frac_train + frac_validation + frac_test, 1), \
        (f"Sum of training ({frac_train}), validation ({frac_validation}), "
         f"and testing ({frac_test}) splits don't add up to 1")

    dataset_file = 'filtered_cancer_small.csv' if use_small_dataset else 'filtered_cancer_all.csv'
    data_df = pd.read_csv(f'{data_path}/{dataset_file}')
    protein_embeddings_df = pd.read_csv(f'{data_path}/protein_embeddings.csv', index_col=0)

    # Create a column that combines the protein ID and whether there was a significant drug-protein
    # interaction. This column will be used to split the data to ensure that each dataset has an
    # appropriate balance of proteins and interaction types.
    data_df['stratify_col'] = data_df['Target_ID'] + '_' + data_df['label'].astype(str)

    # Split part of the data into the training dataset (maintaining an equal split of proteins and interactions types).
    train_df, remaining_df = train_test_split(data_df,
                                              test_size=frac_validation + frac_test,
                                              stratify=data_df['stratify_col'],
                                              random_state=seed)

    # Split the remaining data to get the validation and test datasets (maintaining an equal split of proteins and
    # interaction types).
    validation_df, test_df = train_test_split(remaining_df,
                                              test_size=frac_test / (frac_validation + frac_test),
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
    parser.add_argument("--leaky_relu_slope", type=float, required=False, default=0.2,
                        help="The slope for the Leaky ReLU activation function")
    parser.add_argument("--pooling_dim", type=int, required=False, default=128,
                        help="Pooling dimension")

    return parser


if __name__ == '__main__':
    arg_parser = get_parser()
    training_args = arg_parser.parse_args()

    train_model(training_args)
