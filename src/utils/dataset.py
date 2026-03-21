"""
This module contains the necessary datasets for the project.
More specifically, it has the main DrugProtein dataset, which contains specific DrugMolecules.
Note that DrugMolecules now include optional 3D atomic coordinates as node features.
"""

import rdkit.Chem
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Any
import pandas as pd
from functools import lru_cache

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .functional_groups import *
from utils.embed_proteins import ProteinGraphBuilder


class DrugMolecule:
    """
    Represents a drug molecule and its molecular graph structure, optionally with 3D coords.

    Node features: one-hot atom type, formal charge, degree, hybridization, aromatic, plus (x,y,z) coords if available.
    Edge features: one-hot bond type, conjugation, ring flags.
    """
    def __init__(self, smiles_str: str, max_nodes: int = 50, include_3d: bool = False):
        self.include_3d = include_3d
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles_str}")

        # Always embed a 3-D conformer so that pos_tensor is available for
        # cross-graph edge construction in collate_drug_prot.  H atoms are
        # added for geometry quality, then stripped; the resulting heavy-atom
        # mol is used for both the coordinate array and the graph.
        # NOTE: these are RDKit ETKDGv3 coords in a local frame.  Replace
        # atom_coords with Boltz-2 complex coordinates (same frame as the
        # protein PDB) once Boltz-2 integration is complete.
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        embed_ok = AllChem.EmbedMolecule(mol_h, params) != -1
        mol = Chem.RemoveHs(mol_h)          # heavy-atom mol, conformer intact when embed_ok
        if embed_ok and mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            self.atom_coords = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
        else:
            self.atom_coords = [[0.0, 0.0, 0.0]] * mol.GetNumAtoms()

        self.mol, self.node_feats, self.edge_feats, self.adjacency_list, self.neighbours = \
            self._construct_molecular_graph(mol)
        self.num_nodes = len(self.node_feats)
        self.max_nodes = max_nodes
        self.node_tensor, self.edge_tensor, self.adjacency_tensor, self.pos_tensor = \
            self._tensor_preprocess()

    def _construct_molecular_graph(self, mol: Chem.Mol):
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
        # process node features
        proc = []
        for idx, feats in enumerate(self.node_feats):
            vec = self._process_node_features(feats)
            if self.include_3d:
                # append xyz coords for this atom
                coords = self.atom_coords[idx]
                vec += coords
            proc.append(vec)
        x = torch.tensor(proc, dtype=torch.float32)
        x = self._pad(x, (self.max_nodes, x.size(1)))
        # edge and adjacency
        # determine edge feature size, even if no edges
        if len(self.edge_feats) > 0:
            num_edge_feats = len(self._process_edge_features(next(iter(self.edge_feats.values()))))
        else:
            # no bonds present: default to unspecified bond encoding + flags
            num_edge_feats = len(self._process_edge_features({"bond_type":"UNSPECIFIED","conjugated":0,"ring":0}))
        e = torch.zeros((self.max_nodes, self.max_nodes, num_edge_feats), dtype=torch.float32)
        a = torch.zeros((self.max_nodes, self.max_nodes), dtype=torch.float32)
        for (i, j), bf in self.edge_feats.items():
            if i < self.max_nodes and j < self.max_nodes:
                e[i, j] = torch.tensor(self._process_edge_features(bf), dtype=torch.float32)
                a[i, j] = 1

        # pos_tensor: heavy-atom 3-D coordinates padded/cropped to max_nodes.
        # Kept separate from node features so that include_3d=False does not
        # change the node feature dimension seen by the model.
        pad_coords = (self.atom_coords + [[0.0, 0.0, 0.0]] * self.max_nodes)[:self.max_nodes]
        pos = torch.tensor(pad_coords, dtype=torch.float32)   # [max_nodes, 3]

        return x, e, a, pos

    def _pad(self, t: torch.Tensor, shape: tuple):
        pad = []
        for cur, tgt in zip(reversed(t.shape), reversed(shape)):
            pad += [0, tgt - cur]
        return F.pad(t, pad, mode='constant', value=0)

    def _process_node_features(self, features: dict[str, Any]) -> list[float]:
        hyb = {"UNSPECIFIED":0,"S":1,"SP":2,"SP2":3,"SP3":4,"SP2D":5,"SP3D":6,"SP3D2":7,"OTHER":8}
        atoms = [1,3,5,6,7,8,9,11,14,15,16,17,19,30,34,35,53]
        out: list[float] = []
        for k, v in features.items():
            if k == 'hybridization':
                one = [0.0]*len(hyb); one[hyb.get(v,0)] = 1.0; out += one
            elif k == 'atomic_num':
                one = [0.0]*len(atoms)
                if v in atoms: one[atoms.index(v)] = 1.0
                out += one
            else:
                out.append(float(v))
        return out

    def _process_edge_features(self, features: dict[str, Any]) -> list[float]:
        bond_enc = {"UNSPECIFIED":0,"SINGLE":1,"DOUBLE":2,"TRIPLE":3,"AROMATIC":4,
                    "IONIC":5,"HYDROGEN":6,"THREECENTER":7,"DATIVEONE":8,"DATIVE":9,
                    "DATIVEL":10,"DATIVER":11,"OTHER":12,"ZERO":13}
        out: list[float] = []
        for k, v in features.items():
            if k == 'bond_type':
                one = [0.0]*len(bond_enc); one[bond_enc.get(v,0)] = 1.0; out += one
            else:
                out.append(float(v))
        return out

    def to_tensors(self):
        return self.node_tensor, self.edge_tensor, self.adjacency_tensor, self.pos_tensor


class DrugProteinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, prot_emb: pd.DataFrame, graph_dir: str,
                 max_nodes: int = 256, use_half: bool = False, include_3d: bool = False):
        self.max_nodes = max_nodes
        self.use_half = use_half
        self.include_3d = include_3d
        self.pchembl = df['pChEMBL_Value'].tolist()
        self.smiles = df['smiles'].tolist()
        self.prot_ids = df['Target_ID'].tolist()
        self.prot_emb_df = prot_emb
        self.graph_dir = graph_dir
        self.builder = ProteinGraphBuilder(self.graph_dir)

    def __len__(self):
        return len(self.pchembl)

    @lru_cache(maxsize=512)
    def load_drug(self, smiles: str):
        return DrugMolecule(smiles, self.max_nodes, self.include_3d).to_tensors()

    @lru_cache(maxsize=128)
    def load_protein(self, pid: str):
        return self.builder.load(pid)

    def __getitem__(self, i):
        d_n, d_e, d_a, d_pos = self.load_drug(self.smiles[i])
        pg = self.load_protein(self.prot_ids[i])
        p_n   = pg.x.cpu().half()        if self.use_half else pg.x.cpu()
        p_e   = pg.edge_attr.cpu().half() if self.use_half else pg.edge_attr.cpu()
        p_i   = pg.edge_index.cpu()
        # Protein Cα positions from PDB / Boltz-2; used for cross-graph edge construction.
        p_pos = pg.pos.cpu() if (hasattr(pg, 'pos') and pg.pos is not None) \
                else torch.zeros((pg.x.size(0), 3), dtype=torch.float32)
        lbl   = torch.tensor(self.pchembl[i], dtype=torch.float32)
        return d_n, d_e, d_a, d_pos, p_n, p_e, p_i, p_pos, lbl


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'extra-imports': [
            'rdkit.Chem', 'Chem', 'AllChem', 'typing', 'pandas', 'torch',
            'torch.utils.data', 'torch.nn.functional', 'src.utils.functional_groups'
        ],
        'disable': ['R0914', 'E1101'],
        'allowed-io': [],
        'max-line-length': 120,
    })
