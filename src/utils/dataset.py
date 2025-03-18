from rdkit import Chem
from typing import Any
import pandas as pd

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class DrugMolecule:
    def __init__(self, smiles_str: str) -> None:
        self.node_features, self.edge_features, self.adjacency_list = self._construct_molecular_graph(smiles_str)
        self.num_nodes = len(self.node_features)

        self.node_tensor, self.edge_tensor, self.adjacency_tensor = self._tensor_preprocess()

    def _construct_molecular_graph(self, smiles_str: str) -> tuple[list[Any], list[Any], list[Any]]:
        mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles_str))  # remove explicit H atoms

        node_features = []
        adjacency_list = []
        edge_features = []

        for atom in mol.GetAtoms():
            feats = {
                "atomic_num": atom.GetAtomicNum(),
                "formal_charge": atom.GetFormalCharge(),
                "degree": atom.GetDegree(),
                "hybridization": str(atom.GetHybridization()),
                "aromatic": int(atom.GetIsAromatic())
            }
            node_features.append(feats)

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_feat = {
                "bond_type": str(bond.GetBondType()),
                "conjugated": int(bond.GetIsConjugated()),
                "ring": int(bond.IsInRing())
            }
            edge_features.append(((i, j), bond_feat))
            # Undirected adjacency
            adjacency_list.append((i, j))
            adjacency_list.append((j, i))

        return node_features, edge_features, adjacency_list

    def _tensor_preprocess(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        processed_node_features = []
        for features in self.node_features:
            processed_node_features.append(self._process_node_features(features))
        node_tensor = torch.tensor(processed_node_features)
        node_tensor = node_tensor.type(torch.float32)
        # Pad the dimensions that are dependent on the number of nodes.
        node_tensor = self._pad_tensor(node_tensor, [0])

        # 14 different bond types plus 2 numerical variables
        num_edge_features = 16
        edge_tensor = torch.zeros((self.num_nodes, self.num_nodes, num_edge_features))
        for (node_1, node_2), features in self.edge_features:
            edge_tensor[node_1, node_2, :] = torch.tensor(self._process_edge_features(features))
        edge_tensor = edge_tensor.type(torch.float32)
        # Pad the dimensions that are dependent on the number of nodes.
        edge_tensor = self._pad_tensor(edge_tensor, [0, 1])

        # Add self-loops to the adjacency matrix.
        adjacency_tensor = torch.diag(torch.ones(self.num_nodes))
        # For each edge, add a 1 to the adjacency matrix. Note that the adjacency list contains two separate
        # entries for each edge (to ensure that all edges are bidirectional).
        for node_1, node_2 in self.adjacency_list:
            adjacency_tensor[node_1, node_2] = 1
        adjacency_tensor = adjacency_tensor.type(torch.float32)
        # Pad the dimensions that are dependent on the number of nodes.
        adjacency_tensor = self._pad_tensor(adjacency_tensor, [0, 1])

        return node_tensor, edge_tensor, adjacency_tensor

    def _pad_tensor(self, x: torch.Tensor, dims: list[int], size_after_padding: int = 50) -> torch.Tensor:
        # Pad the tensors so that every dimension passed in has a size of size_after_padding.
        padding_list = []
        for i in range(len(x.shape)-1, -1, -1):
            if i in dims:
                assert x.shape[i] <= size_after_padding, \
                    (f"Current size of dimension {i} ({x.shape[i]}) is larger than "
                     f"the required size after padding ({size_after_padding})")
                padding_list.extend([0, size_after_padding - x.shape[i]])
            else:
                padding_list.extend([0, 0])
        # Pad the tensor with zeroes to achieve the required size.
        return F.pad(x, tuple(padding_list), mode='constant', value=0)

    def _process_node_features(self, features: dict[str, int | str]) -> list[int]:
        hybridization_encoder_dict = {
            "UNSPECIFIED": 0, "S": 1, "SP": 2, "SP2": 3, "SP3": 4,
            "SP2D": 5, "SP3D": 6, "SP3D2": 7, "OTHER": 8
        }

        processed_features = []
        for key, val in features.items():
            if key == 'hybridization':
                one_hot_list = [0] * len(hybridization_encoder_dict)
                one_hot_list[hybridization_encoder_dict[val]] = 1
                processed_features.extend(one_hot_list)
            else:
                processed_features.append(val)
        return processed_features

    def _process_edge_features(self, features: dict[str, int | str]) -> list[int]:
        bond_type_encoder_dict = {
            "UNSPECIFIED": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4,
            "IONIC": 5, "HYDROGEN": 6, "THREECENTER": 7, "DATIVEONE": 8,
            "DATIVE": 9, "DATIVEL": 10, "DATIVER": 11, "OTHER": 12, "ZERO": 13
        }

        processed_features = []
        for key, val in features.items():
            if key == 'bond_type':
                one_hot_list = [0] * len(bond_type_encoder_dict)
                one_hot_list[bond_type_encoder_dict[val]] = 1
                processed_features.extend(one_hot_list)
            else:
                processed_features.append(val)
        return processed_features


class DrugProteinDataset(Dataset):
    def __init__(self, data_folder: str) -> None:
        super().__init__()

        data_df = pd.read_csv(f'{data_folder}/filtered_cancer_all.csv')
        # Filter out rows that don't have a SMILES string for the drug.
        data_df = data_df.loc[~data_df['smiles'].isna(), :]
        self.size = data_df.shape[0]

        self.pchembl_scores = data_df['pChEMBL_Value'].tolist()
        self.protein_ids = data_df['Target_ID'].tolist()
        self.smiles_strs = data_df['smiles'].tolist()

        protein_embeddings_df = pd.read_csv(f'{data_folder}/protein_embeddings.csv', index_col=0)
        self.protein_embeddings_dict = {x: torch.tensor(protein_embeddings_df.loc[x, :].tolist())
                                        for x in protein_embeddings_df.index}

        # Lazyload the drug graphs.
        self.drug_graphs = [None] * self.size

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        drug_graph = self.drug_graphs[idx]
        if drug_graph is None:
            drug_graph = DrugMolecule(self.smiles_strs[idx])
            self.drug_graphs[idx] = drug_graph

        protein_id = self.protein_ids[idx]
        protein_embedding = self.protein_embeddings_dict[protein_id]
        pchembl_score = self.pchembl_scores[idx]

        node_features = drug_graph.node_tensor
        edge_features = drug_graph.edge_tensor
        adjacency_matrix = drug_graph.adjacency_tensor

        node_features = torch.cat(
            (node_features, protein_embedding.unsqueeze(0).repeat(node_features.shape[0], 1)), dim=-1)

        return node_features, edge_features, adjacency_matrix, pchembl_score
