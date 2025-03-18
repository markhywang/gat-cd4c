import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from model import GraphAttentionNetwork
from dataset import DrugProteinDataset


def train_model():
    model = GraphAttentionNetwork(333, 1, 16, 100, 4, 2)
    dataset = DrugProteinDataset('../data')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)


if __name__ == '__main__':
    #mol = DrugMolecule("O=C(NO)[C@H]1C[C@@H](OC(=O)N2CCCC2)CN[C@@H]1C(=O)N1CC=C(c2ccccc2)CC1")
    #model = GraphAttentionNetwork(13, 1, 16, 20, 4, 2)
    #print(model(mol.node_tensor.unsqueeze(0), mol.edge_tensor.unsqueeze(0), mol.adjacency_tensor.unsqueeze(0)))

    train_model()
