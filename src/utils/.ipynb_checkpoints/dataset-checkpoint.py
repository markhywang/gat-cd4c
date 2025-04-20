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
from utils.embed_proteins import ProteinGraphBuilder


class DrugMolecule:
    """
    Represents a drug molecule and its molecular graph structure.

    Instance Attributes:
        mol: The RDKit molecule object representing the drug.
        node_tensor: A tensor representation of the node features.
        edge_tensor: A tensor representation of the edge features.
        adjacency_tensor: A tensor representation of the adjacency matrix.
    """
    def __init__(self, smiles_str: str, max_nodes: int = 50):
        self.mol, self.node_feats, self.edge_feats, self.adjacency_list, self.neighbours = \
            self._construct_molecular_graph(smiles_str)
        self.num_nodes = len(self.node_feats)
        self.max_nodes = max_nodes
        self.node_tensor, self.edge_tensor, self.adjacency_tensor = self._tensor_preprocess()

    def _construct_molecular_graph(self, smiles_str: str):
        mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles_str))
        node_feats, adjacency_list, edge_feats, neighbours = [], [], {}, []
        for atom in mol.GetAtoms():
            feats = {
                "atomic_num": atom.GetAtomicNum(),
                "formal_charge": atom.GetFormalCharge(),
                "degree": atom.GetDegree(),
                "hybridization": str(atom.GetHybridization()),
                "aromatic": int(atom.GetIsAromatic())
            }
            node_feats.append(feats)
            neighbours.append([])
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = {"bond_type": str(bond.GetBondType()),
                  "conjugated": int(bond.GetIsConjugated()),
                  "ring": int(bond.IsInRing())}
            edge_feats[(i, j)], edge_feats[(j, i)] = bf, bf
            adjacency_list += [(i, j), (j, i)]
            neighbours[i].append(j)
            neighbours[j].append(i)
        return mol, node_feats, edge_feats, adjacency_list, neighbours

    def _tensor_preprocess(self):
        # Node feature processing
        proc_nodes = [self._process_node_features(f) for f in self.node_feats]
        x = torch.tensor(proc_nodes, dtype=torch.float32)
        x = self._pad(x, (self.max_nodes, x.size(1)))
        # Edge feature and adjacency
        num_edge_feats = 16
        e = torch.zeros((self.max_nodes, self.max_nodes, num_edge_feats), dtype=torch.float32)
        a = torch.zeros((self.max_nodes, self.max_nodes), dtype=torch.float32)
        for (i, j), bf in self.edge_feats.items():
            if i < self.max_nodes and j < self.max_nodes:
                e[i, j] = torch.tensor(self._process_edge_features(bf), dtype=torch.float32)
                a[i, j] = 1
        return x, e, a

    def _pad(self, t: torch.Tensor, shape: tuple):
        pad = []
        for cur, tgt in zip(reversed(t.shape), reversed(shape)):
            pad += [0, tgt - cur]
        return F.pad(t, pad, mode='constant', value=0)

    def _process_node_features(self, features: dict[str, Any]) -> list[int]:
        hyb = {"UNSPECIFIED":0,"S":1,"SP":2,"SP2":3,"SP3":4,"SP2D":5,"SP3D":6,"SP3D2":7,"OTHER":8}
        atoms = [1,3,5,6,7,8,9,11,14,15,16,17,19,30,34,35,53]
        out = []
        for k, v in features.items():
            if k == 'hybridization':
                one = [0]*len(hyb); one[hyb.get(v,0)] = 1; out += one
            elif k == 'atomic_num':
                one = [0]*len(atoms)
                if v in atoms: one[atoms.index(v)] = 1
                out += one
            else:
                out.append(v)
        return out

    def _process_edge_features(self, features: dict[str, Any]) -> list[int]:
        bond_enc = {"UNSPECIFIED":0,"SINGLE":1,"DOUBLE":2,"TRIPLE":3,"AROMATIC":4,
                    "IONIC":5,"HYDROGEN":6,"THREECENTER":7,"DATIVEONE":8,"DATIVE":9,
                    "DATIVEL":10,"DATIVER":11,"OTHER":12,"ZERO":13}
        out = []
        for k, v in features.items():
            if k == 'bond_type':
                one = [0]*len(bond_enc); one[bond_enc.get(v,0)] = 1; out += one
            else:
                out.append(v)
        return out

    def to_tensors(self):
        return self.node_tensor, self.edge_tensor, self.adjacency_tensor


class DrugProteinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, prot_emb: pd.DataFrame, graph_dir: str,
                 max_nodes: int = 256, use_half: bool = False):
        self.max_nodes = max_nodes
        self.use_half = use_half
        self.pchembl = df['pChEMBL_Value'].tolist()
        self.smiles = df['smiles'].tolist()
        self.prot_ids = df['Target_ID'].tolist()
        self.prot_emb_df = prot_emb
        self.graph_dir = graph_dir

    def __len__(self):
        return len(self.pchembl)

    def __getitem__(self, i):
        # drug
        dm = DrugMolecule(self.smiles[i], self.max_nodes)
        d_n, d_e, d_a = dm.to_tensors()
        # protein graph, lazy-load per sample
        builder = ProteinGraphBuilder(self.graph_dir)
        pg = builder.load(self.prot_ids[i])
        # map to CPU, optionally half
        p_n = pg.x.cpu().half() if self.use_half else pg.x.cpu()
        p_e = pg.edge_attr.cpu().half() if self.use_half else pg.edge_attr.cpu()
        p_i = pg.edge_index.cpu()
        # label
        lbl = torch.tensor(self.pchembl[i], dtype=torch.float32)
        return d_n, d_e, d_a, p_n, p_e, p_i, lbl


if __name__ == '__main__':
    import doctest
    doctest.testmod()

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
