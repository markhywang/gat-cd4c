import argparse

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from model import GraphAttentionNetwork
from dataset import DrugProteinDataset


def train_model(args: argparse.Namespace) -> None:
    # TODO - remove hard-coded specifications for the model
    model = GraphAttentionNetwork(333, 1, 16,
                                  args.hidden_size, args.num_layers, args.num_attn_heads)
    dataset = DrugProteinDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


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
    parser.add_argument("--hidden_size", type=int, required=False, default=32,
                        help="The size of embeddings for hidden layers")
    parser.add_argument("--num_layers", type=int, required=False, default=4,
                        help="The number of graph attention layers to use")
    parser.add_argument("--num_attn_heads", type=int, required=False, default=4,
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
