"""
This module contains the necessary datasets for the project.
More specifically, it has the main DrugProtein dataset, which contains specific DrugMolecules.
Note that DrugMolecules now include optional 3D atomic coordinates as node features.
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Any
import pandas as pd
from functools import lru_cache

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

try:
    from .functional_groups import *
    from .embed_proteins import ProteinGraphBuilder
except ImportError:
    # For when the module is imported from outside
    from src.utils.functional_groups import *
    from src.utils.embed_proteins import ProteinGraphBuilder

ATOM_PAD_ID = 0          # 0 = "dummy / padded node"

from sklearn.model_selection import train_test_split
from tdc.multi_pred import DTI


# ----------------------------------------------------------------------------
# Collation helper
# ----------------------------------------------------------------------------
def pad_to(x: torch.Tensor, shape: tuple):
    pad = []
    for cur, tgt in zip(reversed(x.shape), reversed(shape)):
        pad += [0, tgt - cur]
    return F.pad(x, pad, mode='constant', value=0)


def collate_drug_prot(
        batch,
        prot_graph_dir,
        hard_limit=64,
        drug_edge_feats=17,
        prot_edge_feats=1):

    drug_zs, drug_xs, drug_es, drug_as = [], [], [], []
    prot_zs, prot_xs, prot_es, prot_as = [], [], [], []
    labels = []

    H = hard_limit

    for d_x, d_z, d_e, d_a, p_x, p_e, p_i, label in batch:
        # ───────────── drug ─────────────
        drug_zs.append(pad_to(d_z, (H,)))
        drug_xs.append(pad_to(d_x, (H, d_x.size(-1))))
        drug_es.append(pad_to(d_e, (H, H, drug_edge_feats)))
        drug_as.append(pad_to(d_a, (H, H)))

        # ─────────── protein ────────────
        N = p_x.size(0)
        if N > H:                                   # truncate to hard_limit
            mask = (p_i[0] < H) & (p_i[1] < H)
            p_i  = p_i[:, mask]
            p_e  = p_e[mask]
            p_x  = p_x[:H]
            N    = H

        # (1) residue-type integer IDs  (derive from first 20 dims of one-hot)
        res_ids = torch.argmax(p_x[:, :20], dim=1).long() + 1   # 1…20, 0=pad
        prot_zs.append(pad_to(res_ids, (H,)))

        # (2) node features
        prot_xs.append(pad_to(p_x, (H, p_x.size(1))))

        # (3) dense edge-attr & adjacency
        adj     = torch.zeros((H, H), dtype=torch.float32)
        edge_t  = torch.zeros((H, H, prot_edge_feats), dtype=torch.float32)
        for j in range(p_i.size(1)):
            i0, i1 = int(p_i[0, j]), int(p_i[1, j])
            adj[i0, i1]     = 1
            edge_t[i0, i1]  = p_e[j]

        prot_as.append(adj)
        prot_es.append(edge_t)

        labels.append(label)

    # final stacked batch
    return (
        torch.stack(drug_zs),    # [B, H]          long
        torch.stack(drug_xs),    # [B, H, F_d]
        torch.stack(drug_es),    # [B, H, H, F_e_d]
        torch.stack(drug_as),    # [B, H, H]
        torch.stack(prot_zs),    # [B, H]          long
        torch.stack(prot_xs),    # [B, H, F_p]
        torch.stack(prot_es),    # [B, H, H, F_e_p]
        torch.stack(prot_as),    # [B, H, H]
        torch.tensor(labels, dtype=torch.float32)
    )



class DrugMolecule:
    """
    Represents a drug molecule and its molecular graph structure, optionally with 3D coords.

    Node features: one-hot atom type, formal charge, degree, hybridization, aromatic, plus (x,y,z) coords if available.
    Edge features: one-hot bond type, conjugation, ring flags.
    """
    def __init__(self, smiles_str: str, max_nodes: int = 50, include_3d: bool = True):
        self.include_3d = include_3d

        # Parse base molecule
        mol0 = Chem.MolFromSmiles(smiles_str)
        if mol0 is None:
            raise ValueError(f"Invalid SMILES: {smiles_str}")

        # Prepare hydrogenated mol for optional 3D embedding
        mol_h = Chem.AddHs(mol0)

        # Identify heavy atom indices for coordinate fallback
        heavy_atom_indices = [atom.GetIdx() for atom in mol_h.GetAtoms() if atom.GetAtomicNum() != 1]

        if include_3d:
            # Embed with ETKDG, fixed seed
            params = AllChem.ETKDGv3()
            params.randomSeed = 0
            res = AllChem.EmbedMolecule(mol_h)
            # 2) then optimize with MMFF94
            if res == 0 and mol_h.GetNumConformers() > 0:
                opt = AllChem.MMFFOptimizeMolecule(mol_h)   # <— run MMFF94
                conf = mol_h.GetConformer()
                heavy_coords = [[*conf.GetAtomPosition(idx)] for idx in heavy_atom_indices]
            else:
                heavy_coords = [[0.0, 0.0, 0.0] for _ in heavy_atom_indices]
            self.atom_coords = heavy_coords

        # Remove hydrogens to build heavy-only graph
        mol = Chem.RemoveHs(mol_h)

        # Extract graph structure
        self.mol, self.node_feats, self.edge_feats, self.adjacency_list, self.neighbours = \
            self._construct_molecular_graph(mol)
        self.num_nodes = len(self.node_feats)
        self.max_nodes = max_nodes
        self.node_tensor, self.edge_tensor, self.adjacency_tensor, self.atomic_id_tensor = self._tensor_preprocess()

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
        proc_x   = []
        proc_z   = []                       # <— NEW
        
        for idx, feats in enumerate(self.node_feats):
            proc_x.append(self._process_node_features(feats))
            proc_z.append(feats["atomic_num"])          # keep raw Z

        # --- pad (node-feature matrix) ---
        x = torch.tensor(proc_x, dtype=torch.float32)
        x = self._pad(x, (self.max_nodes, x.size(1)))

        # --- pad (atomic-ID vector) ------
        z = torch.tensor(proc_z, dtype=torch.long)
        z = F.pad(z, (0, self.max_nodes - z.size(0)), value=ATOM_PAD_ID)

        # edge and adjacency
        if len(self.edge_feats) > 0:
            num_edge_feats = len(self._process_edge_features(next(iter(self.edge_feats.values()))))
        else:
            num_edge_feats = len(self._process_edge_features({"bond_type":"UNSPECIFIED","conjugated":0,"ring":0}))
        e = torch.zeros((self.max_nodes, self.max_nodes, num_edge_feats), dtype=torch.float32)
        a = torch.zeros((self.max_nodes, self.max_nodes), dtype=torch.float32)
        for (i, j), bf in self.edge_feats.items():
            if i < self.max_nodes and j < self.max_nodes:
                e[i, j] = torch.tensor(self._process_edge_features(bf), dtype=torch.float32)
                a[i, j] = 1
            
        return x, e, a, z

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
        return self.node_tensor, self.edge_tensor, self.adjacency_tensor, self.atomic_id_tensor


class DrugProteinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, prot_emb: pd.DataFrame, graph_dir: str,
                 max_nodes: int = 64, use_half: bool = False, include_3d: bool = False):
        from rdkit import RDLogger

        # Retrieve the singleton logger
        lg = RDLogger.logger()
        # Suppress everything below CRITICAL (i.e., warnings and infos are silenced)
        lg.setLevel(RDLogger.CRITICAL)

        
        self.max_nodes = max_nodes
        self.use_half = use_half
        self.include_3d = include_3d
        self.pchembl = df['pChEMBL_Value'].values.tolist()
        self.smiles = df['smiles'].values.tolist()
        self.prot_ids = df['Target_ID'].values.tolist()
        self.prot_emb_df = prot_emb
        self.graph_dir = graph_dir
        self.builder = ProteinGraphBuilder(self.graph_dir)

    def __len__(self):
        return len(self.pchembl)

    @lru_cache(maxsize=8192)
    def load_drug(self, smiles: str):
        return DrugMolecule(smiles, self.max_nodes, self.include_3d).to_tensors()

    @lru_cache(maxsize=256)
    def load_protein(self, pid: str):
        return self.builder.load(pid)

    def __getitem__(self, i):
        # ── drug tensors (order matters!) ───────────────────────────────
        d_x, d_e, d_a, d_z = self.load_drug(self.smiles[i])   # <- 4-tuple
    
        # ── raw protein graph from builder ─────────────────────────────
        pg   = self.load_protein(self.prot_ids[i])
        p_x  = pg.x.cpu().half()        if self.use_half else pg.x.cpu()        # node-features
        p_e  = pg.edge_attr.cpu().half() if self.use_half else pg.edge_attr.cpu()  # edge-attr
        p_i  = pg.edge_index.cpu()                                            # edge index (sparse)
    
        lbl  = torch.tensor(self.pchembl[i], dtype=torch.float32)
    
        # NOTE: we *do not* build dense adjacency here; that's done in the collate_fn
        return d_x, d_z, d_e, d_a, p_x, p_e, p_i, lbl


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