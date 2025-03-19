import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

from model import GraphAttentionNetwork
from utils.dataset import DrugProteinDataset


def train_model(args: argparse.Namespace) -> None:
    # TODO - remove hard-coded specifications for the model
    model = GraphAttentionNetwork(333, 1, 16,
                                  args.hidden_size, args.num_layers, args.num_attn_heads)
    train_dataset, validation_dataset, test_dataset = load_data(args.data_path, args.seed)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


def load_data(data_path: str, seed: int) -> tuple[Dataset, Dataset, Dataset]:
    data_df = pd.read_csv(f'{data_path}/filtered_cancer_all.csv')
    protein_embeddings_df = pd.read_csv(f'{data_path}/protein_embeddings.csv', index_col=0)

    # Create a column that combines the protein ID and whether there was a significant drug-protein
    # interaction. This column will be used to split the data to ensure that each dataset has an
    # appropriate balance of proteins and interaction types.
    data_df['stratify_col'] = data_df['Target_ID'] + '_' + data_df['label'].astype(str)

    # Split 70% of the data into the training dataset (maintaing an equal split of proteins and interactions types).
    train_df, remaining_df = train_test_split(data_df, test_size=0.3,
                                              stratify=data_df['stratify_col'], random_state=seed)
    # Split the remaining 30% of the data in half to get the validation and test datasets (maintaining an equal
    # split of proteins and interaction types). Thus, the validation and test datasets will each contain 15% of
    # the data.
    validation_df, test_df = train_test_split(remaining_df, test_size=0.5,
                                              stratify=remaining_df['stratify_col'], random_state=seed)

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
    parser.add_argument("--batch_size", type=int, required=False, default=64,
                        help="Batch size for data loader")
    parser.add_argument("--stoppage_epochs", type=int, required=False, default=5,
                        help="Number of consecutive epochs with no improvement to validation "
                             "loss before stopping training")
    parser.add_argument("--max_epochs", type=int, required=False, default=128,
                        help="Maximum number of epochs to run training")
    parser.add_argument("--data_path", type=str, required=False, default='../data',
                        help="Path to the folder with the data")
    parser.add_argument("--seed", type=int, required=False, default=0,
                        help="The seed used to control any stochastic operations")
    # Loss parameters
    parser.add_argument("--huber_delta", type=float, required=False, default=0.2,
                        help="Delta parameter for Huber loss function")
    # Optimizer paramters
    parser.add_argument("--weight_decay", type=float, required=False, default=1e-3,
                        help="Weight decay for optimizer")
    parser.add_argument("--lr", type=float, required=False, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--scheduler_rate", type=int, required=False, default=5,
                        help="Number of epochs before reducing the learning rate")
    # Model parameters
    parser.add_argument("--hidden_size", type=int, required=False, default=256,
                        help="The size of embeddings for hidden layers")
    parser.add_argument("--num_layers", type=int, required=False, default=16,
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
    parser.add_argument("--print_steps", type=int, required=False, default=1,
                        help="Number of batches before printing loss")
    return parser


if __name__ == '__main__':
    arg_parser = get_parser()
    training_args = arg_parser.parse_args()

    train_model(training_args)
