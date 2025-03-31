"""
This module contains the necessary datasets for the project.
More specifically, it has the main DrugProtein dataset, which contains specific DrugMolecules.
Note that DrugMolecules has additional helper functions to aid in both pre-training and for graph visualizations.
"""

import rdkit.Chem
from rdkit import Chem
from typing import Any
import pandas as pd
from itertools import product

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .functional_groups import *


class DrugMolecule:
    """
    Represents a drug molecule and its molecular graph structure.

    Instance Attributes:
        mol: The RDKit molecule object representing the drug.
        node_features: A list containing feature dictionaries for each atom in the molecule.
        edge_features: A dictionary mapping bond pairs to their respective feature dictionaries.
        adjacency_list: A list of tuples representing undirected adjacency relationships between atoms.
        neighbours: A list where each index corresponds to an atom and contains a list of its neighboring atom indices.
        num_nodes: The total number of atoms in the molecule.
        node_tensor: A tensor representation of the node features.
        edge_tensor: A tensor representation of the edge features.
        adjacency_tensor: A tensor representation of the adjacency matrix.
    """
    mol: rdkit.Chem.Mol
    node_features: list[Any]
    edge_features: dict[Any, Any]
    adjacency_list: list[Any]
    neighbours: list[list[int]]
    num_nodes: int
    node_tensor: torch.Tensor
    edge_tensor: torch.Tensor
    adjacency_tensor: torch.Tensor

    def __init__(self, smiles_str: str) -> None:
        """Initializes a DrugMolecule by constructing its molecular graph."""
        self.mol, self.node_features, self.edge_features, self.adjacency_list, self.neighbours = (
            self._construct_molecular_graph(smiles_str))
        self.num_nodes = len(self.node_features)

        self.node_tensor, self.edge_tensor, self.adjacency_tensor = self._tensor_preprocess()

    def find_functional_group(self, functional_group: FunctionalGroup) -> list[dict[int, int]]:
        """Finds occurrences of a specified functional group within the molecule."""
        root_node_specs = functional_group.get_root_node_specs()
        final_found_dicts = []
        for node_num, node_specs in enumerate(self.node_features):
            if self._check_features_match(root_node_specs, node_specs):
                # If this node matches the root node, create a mapping between this node and the root node.
                found_dict = {functional_group.root_node: node_num}
                new_found_dicts = self._functional_group_helper(functional_group, functional_group.root_node, found_dict)
                if len(new_found_dicts) > 0:
                    final_found_dicts.extend(new_found_dicts)
        return final_found_dicts

    def _functional_group_helper(self, functional_group: FunctionalGroup, node: int,
                                 found_dict: dict[int, int]) -> list[dict[int, int]]:
        """Recursively finds matching subgraphs corresponding to a functional group."""
        drug_node = found_dict[node]
        functional_group_neighbours = functional_group.neighbours[node]
        # Create a dictionary that tracks the options for drug nodes that can be used for a given
        # functional group node.
        functional_group_node_to_drug_nodes = {}

        for x in functional_group_neighbours:
            expected_node_features = functional_group.node_features[x]
            expected_edge_features = functional_group.edge_features[(x, node)] \
                if (x, node) in functional_group.edge_features \
                else functional_group.edge_features[(node, x)]

            # If this node has already been used, check that its edge matches the function group specification.
            if x in found_dict:
                # Check that the edge features match the expected edge features.
                actual_edge_features = self.edge_features[(drug_node, found_dict[x])] \
                    if (drug_node, found_dict[x]) in self.edge_features \
                    else self.edge_features[(found_dict[x], drug_node)]

                if not self._check_features_match(expected_edge_features, actual_edge_features):
                    # If an edge features conflict occurs, no matching subgraph can be found.
                    return []
            else:
                # Get all the options from the drug molecule that can be used as this node.
                curr_options = []
                for possible_drug_node in self.neighbours[drug_node]:
                    if self._check_features_match(expected_node_features, self.node_features[possible_drug_node]):
                        actual_edge_features = self.edge_features[(drug_node, possible_drug_node)] \
                            if (drug_node, possible_drug_node) in self.edge_features \
                            else self.edge_features[(possible_drug_node, drug_node)]
                        if self._check_features_match(expected_edge_features, actual_edge_features):
                            curr_options.append(possible_drug_node)

                if len(curr_options) == 0:
                    # If no options were found, a matching subgraph can't be found.
                    return []
                else:
                    functional_group_node_to_drug_nodes[x] = curr_options

        if len(functional_group_node_to_drug_nodes) == 0:
            # No more nodes need to be added to the subgraph, so a complete subgraph has been found.
            return [found_dict]

        final_found_dicts = []
        # Iterate over every possible combination of choices to iterate over next.
        for combination in self._get_unique_combinations(functional_group_node_to_drug_nodes):
            functional_group_nodes = list(combination.keys())
            drug_nodes = list(combination.values())

            # Add the current combination being tried to found_dict.
            curr_found_dicts = [found_dict.copy() | combination]
            old_found_dicts = []

            for curr_functional_group_node in functional_group_nodes:
                old_found_dicts = curr_found_dicts
                curr_found_dicts = []
                for found_dict in old_found_dicts:
                    curr_found_dicts.extend(self._functional_group_helper(
                        functional_group, curr_functional_group_node, found_dict))

                if len(curr_found_dicts) == 0:
                    return []

            final_found_dicts.extend(curr_found_dicts)

        return final_found_dicts

    def _get_unique_combinations(self, options_dict: dict[Any, list]) -> list[dict[Any, Any]]:
        """Given a dict, generate all possible combinations while filtering out duplicates."""
        keys, values = zip(*sorted(options_dict.items()))  # Extract keys and ordered value lists
        all_combinations = product(*values)  # Generate all possible combinations

        # Convert to dictionaries and filter out those with duplicate values.
        unique_dicts = [
            dict(zip(keys, combo)) for combo in all_combinations if len(set(combo)) == len(combo)
        ]

        return unique_dicts

    def _check_features_match(self, features_to_check: dict[Any, Any], curr_features: dict[Any, Any]) -> bool:
        """Given current features and the features to check, check if the nodes matches all specifications"""
        # Check if the current node matches all the specifications.
        for key, val in features_to_check.items():
            if curr_features[key] != val:
                return False
        return True

    def _construct_molecular_graph(self, smiles_str: str) \
            -> tuple[rdkit.Chem.Mol, list[Any], dict[Any, Any], list[Any], list[list[int]]]:
        """Construct the molecular graph given atoms and bonds."""
        mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles_str))  # remove explicit H atoms

        node_features = []
        adjacency_list = []
        edge_features = {}
        neighbours = []

        for atom in mol.GetAtoms():
            feats = {
                "atomic_num": atom.GetAtomicNum(),
                "formal_charge": atom.GetFormalCharge(),
                "degree": atom.GetDegree(),
                "hybridization": str(atom.GetHybridization()),
                "aromatic": int(atom.GetIsAromatic())
            }
            node_features.append(feats)
            neighbours.append([])

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_feat = {
                "bond_type": str(bond.GetBondType()),
                "conjugated": int(bond.GetIsConjugated()),
                "ring": int(bond.IsInRing())
            }
            edge_features[(i, j)] =  bond_feat
            # Undirected adjacency
            adjacency_list.append((i, j))
            adjacency_list.append((j, i))
            neighbours[i].append(j)
            neighbours[j].append(i)

        return mol, node_features, edge_features, adjacency_list, neighbours

    def _tensor_preprocess(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pre-process PyTorch tensors to be eventually passed as input to GAT"""
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
        for (node_1, node_2), features in self.edge_features.items():
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
        """Pad the tensors so that every dimension passed in has a size of size_after_padding."""
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
        """Process node features and encode necessary attributes."""
        hybridization_encoder_dict = {
            "UNSPECIFIED": 0, "S": 1, "SP": 2, "SP2": 3, "SP3": 4,
            "SP2D": 5, "SP3D": 6, "SP3D2": 7, "OTHER": 8
        }
        atomic_nums = [1, 3, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 19, 30, 34, 35, 53]

        processed_features = []
        for key, val in features.items():
            if key == 'hybridization':
                one_hot_list = [0] * len(hybridization_encoder_dict)
                one_hot_list[hybridization_encoder_dict[val]] = 1
                processed_features.extend(one_hot_list)
            elif key == 'atomic_num':
                one_hot_list = [0] * len(atomic_nums)
                try:
                    idx = atomic_nums.index(val)
                    # The element in the one-hot list corresponding to the current atom is set as 1.
                    one_hot_list[idx] = 1
                except ValueError:
                    # If the element is not commonly found in organic molecules (i.e. wasn't encountered
                    # during training), then ignore the atomic number.
                    pass
                processed_features.extend(one_hot_list)
            else:
                processed_features.append(val)

        return processed_features

    def _process_edge_features(self, features: dict[str, int | str]) -> list[int]:
        """Process edge features and encode necessary attributes."""
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
    """
    A PyTorch Dataset representing drug-protein interaction data.

    Instance Attributes:
        size: The total number of samples in the dataset.
        pchembl_scores: A list of pChEMBL activity values for each sample.
        protein_ids: A list of protein target identifiers for each sample.
        smiles_strs: A list of SMILES representations of drug molecules.
        protein_embeddings_dict: A dictionary mapping protein target IDs to their corresponding tensor embeddings.
        drug_graphs: A lazy-loaded list of DrugMolecule objects corresponding to the drugs in the dataset.
    """
    size: int
    pcheml_scores: list
    protein_ids: list
    smiles_strs: list
    protein_embeddings_dict: dict
    drug_graphs: list

    def __init__(self, data_df: pd.DataFrame, protein_embeddings_df: pd.DataFrame) -> None:
        """Initializes the dataset by processing input data frames."""
        super().__init__()

        self.size = data_df.shape[0]

        self.pchembl_scores = data_df['pChEMBL_Value'].tolist()
        self.protein_ids = data_df['Target_ID'].tolist()
        self.smiles_strs = data_df['smiles'].tolist()

        self.protein_embeddings_dict = {x: torch.tensor(protein_embeddings_df.loc[x, :].tolist())
                                        for x in protein_embeddings_df.index}

        # Lazyload the drug graphs.
        self.drug_graphs = [None] * self.size

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Retrieves a single data sample, including drug and protein representations."""
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

        reshaped_protein_embedding = protein_embedding.unsqueeze(0).repeat(node_features.shape[0], 1)
        node_features = torch.cat(
            (node_features, reshaped_protein_embedding), dim=-1)

        return node_features, edge_features, adjacency_matrix, pchembl_score


if __name__ == '__main__':
    import python_ta

    # Did not work for us (maybe PythonTA has a bug)
    # AttributeError: 'ClassDef' object has no attribute 'value'. Did you mean: 'values'?
    python_ta.check_all(config={
        'extra-imports': [
            'rdkit.Chem',
            'rdkit',
            'typing',
            'pandas',
            'itertools',
            'torch',
            'torch.utils.data',
            'torch.nn.functional',
            'src.utils.functional_groups',
        ],
        'disable': ['R0914', 'E1101'],  # R0914 for local variable, E1101 for attributes for imported modules
        'allowed-io': [],
        'max-line-length': 120,
    })
