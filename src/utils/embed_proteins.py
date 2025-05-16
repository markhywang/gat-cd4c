# === embed_proteins.py (complete) ============================================
"""Protein graph builder & bulk graph generator for CD4C, KIBA and DAVIS
===========================================================================

Usage
-----
1. **Single build (library use)**

   ```python
   from embed_proteins import ProteinGraphBuilder
   builder = ProteinGraphBuilder(graph_dir="../data/protein_graphs")
   data = builder.build("AF‑P00533‑F1‑model_v4.pdb")
   builder.save("CHEMBL612545", data)
   ```

2. **Dataset‑wide pre‑compute**

   ```bash
   python embed_proteins.py --dataset KIBA --out-dir ../data/protein_graphs \
                            --num-workers 8
   ```

   The script pulls the desired **TDC** dataset, resolves each *Target_ID*
   (which is already a UniProt accession for KIBA/DAVIS), downloads the AlphaFold
   structure *(.pdb)* if not cached, builds the residue graph (one‑hot + charge +
   Cα coords) and saves it as `<Target_ID>.pt`.

Node features
-------------
* **one‑hot(20)** standard amino‑acid alphabet
* **charge (1)** integer {‑1,0,+1} at pH≈7
* **coords (3)** Cα x,y,z in Å
Total: **24** dims.

Edges
-----
* Peptide‑bond neighbours
* Any residue pair with Cα‑Cα distance < *cut‑off* (default **10 Å**).
Edge feature = single scalar distance (Å).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import re
import sys
from typing import Dict, Optional

import requests
import torch
from Bio.PDB import PDBParser
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Amino‑acid constants
# ---------------------------------------------------------------------------
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX: Dict[str, int] = {aa: i for i, aa in enumerate(AA_ORDER)}
AA_CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 1}
NUM_AA = len(AA_ORDER)

_AF_URL = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb"

# ---------------------------------------------------------------------------
class ProteinGraphBuilder:
    """Construct, cache and load residue‑level graphs."""

    def __init__(self, graph_dir: str = "../data/protein_graphs", cutoff: float = 10.0):
        self.cutoff = cutoff
        self.graph_dir = pathlib.Path(graph_dir)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.parser = PDBParser(QUIET=True)

    # ----------------------------- helpers ---------------------------------
    def _three_to_one(self, resname: str) -> str:
        tbl = {
            "ALA": "A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I","LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y",
        }
        return tbl.get(resname.upper(), "X")

    def _fetch_pdb(self, acc: str) -> pathlib.Path:
        out = self.graph_dir / f"AF-{acc}-F1-model_v4.pdb"
        if out.exists():
            return out
        url = _AF_URL.format(acc=acc)
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            raise RuntimeError(f"AlphaFold PDB not found for {acc}")
        out.write_bytes(r.content)
        return out

    def _extract_ca(self, pdb_path: pathlib.Path):
        coords, aa3 = [], []
        struct = self.parser.get_structure("af", str(pdb_path))
        for model in struct:
            for chain in model:
                for res in chain:
                    if "CA" in res:
                        coords.append(torch.tensor(res["CA"].coord, dtype=torch.float32))
                        aa3.append(res.get_resname())
        return torch.stack(coords), aa3

    def _node_feats(self, aa3: list[str], coords: torch.Tensor):
        one_hot = torch.zeros((len(aa3), NUM_AA), dtype=torch.float32)
        charges = torch.zeros((len(aa3), 1), dtype=torch.float32)
        for i, r3 in enumerate(aa3):
            r1 = self._three_to_one(r3)
            idx = AA_TO_IDX.get(r1)
            if idx is not None:
                one_hot[i, idx] = 1.0
            charges[i, 0] = AA_CHARGE.get(r1, 0)
        return torch.cat([one_hot, charges, coords], dim=1)  # 24‑D

    # ------------------------- public API ----------------------------------
    def build(self, pdb_path: pathlib.Path | str) -> Data:
        pdb_path = pathlib.Path(pdb_path)
        coords, aa3 = self._extract_ca(pdb_path)
        x = self._node_feats(aa3, coords)

        dist = torch.cdist(coords, coords)"""
embed_proteins.py  — revised
---------------------------
Build per‑protein *graphs* instead of fixed 1 283‑D ESM embeddings.

Each residue becomes a node with:
    • 20‑D one‑hot amino‑acid vector
    • 1‑D integer charge at pH ≈ 7.0  {‑1, 0, +1}
    • 3‑D Cα coordinates (x,y,z)
=> node feature dim = 24.

Edges connect residues that are either:
    • sequential neighbours (peptide bond)  OR
    • spatial neighbours within a distance < cut‑off (default 10 Å)
Edge feature = Euclidean distance [Å].

Graphs are saved in ``protein_graphs/<CHEMBL_ID>.pt``
and consumed later by ``utils.dataset.DrugProteinDataset``.

The loader still supports the old “ESM + coords” pathway:  
pass an ESM tensor to *build* and it will concatenate instead of one‑hotting.
"""

from __future__ import annotations

import os
import pathlib
from typing import Dict, Optional

import torch
from Bio.PDB import PDBParser
from torch_geometric.data import Data

__all__ = [
    "AA_TO_IDX",
    "AA_CHARGE",
    "NUM_AA",
    "ProteinGraphBuilder",
]

# ---------------------------------------------------------------------------
# Amino‑acid lookup tables
# ---------------------------------------------------------------------------
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX: Dict[str, int] = {aa: i for i, aa in enumerate(AA_ORDER)}
NUM_AA = len(AA_ORDER)                                                  # 20

# integer charge at pH≈7 (simplified)
AA_CHARGE: Dict[str, int] = {
    "D": -1,
    "E": -1,
    "K": +1,
    "R": +1,
    "H": +1,  # histidine is ~ +0.1 but bin to +1
}

# ---------------------------------------------------------------------------
class ProteinGraphBuilder:
    """Build *either* whole‑protein graphs (旧 behaviour) *or* pocket‑only graphs*.

    New args
    -----
    pocket_radius : float | None
        If given, residues whose C‑α atom is farther than this radius (Å) from **any** ligand
        heavy‑atom will be *discarded*, giving a smaller, task‑focused graph.
    """

    def __init__(self,
                 graph_dir: str = "../../data/protein_graphs",
                 cutoff: float = 10.0,
                 pocket_radius: float | None = None):
        self.cutoff = cutoff              # edge build distance threshold (Å)
        self.pocket_radius = pocket_radius
        self.graph_dir = pathlib.Path(graph_dir)
        self.parser = PDBParser(QUIET=True)

    # ---------------------------------------------------------------------
    # Low‑level helpers
    # ---------------------------------------------------------------------
    def _extract_ca(self, pdb_path: pathlib.Path):
        """Return (**coords**, **aa_list**) for all Cα atoms in the structure."""
        coords, aas = [], []
        structure = self.parser.get_structure("af", str(pdb_path))
        for model in structure:
            for chain in model:
                for res in chain:
                    if "CA" not in res:
                        continue
                    coords.append(torch.tensor(res["CA"].coord, dtype=torch.float32))
                    aas.append(res.get_resname().upper())
        return torch.stack(coords), aas

    def _aa_one_hot(self, aa_list):
        one_hot = torch.zeros((len(aa_list), NUM_AA), dtype=torch.float32)
        for i, aa3 in enumerate(aa_list):
            aa1 = self._three_to_one(aa3)
            idx = AA_TO_IDX.get(aa1, None)
            if idx is not None:
                one_hot[i, idx] = 1.0
        return one_hot

    @staticmethod
    def _three_to_one(resname: str) -> str:
        """Convert 3‑letter residue code to 1‑letter, fallback to 'X'."""
        mapping = {
            "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
            "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
            "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
            "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
        }
        return mapping.get(resname, "X")

    def _charges(self, aa_list):
        charges = torch.zeros((len(aa_list), 1), dtype=torch.float32)
        for i, aa3 in enumerate(aa_list):
            aa1 = self._three_to_one(aa3)
            charges[i, 0] = AA_CHARGE.get(aa1, 0)
        return charges

    def _crop_to_pocket(self,
                        coords: torch.Tensor,
                        ligand_coords: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the subset of `coords` within `pocket_radius` of *any* `ligand_coords`."""
        if ligand_coords is None or self.pocket_radius is None:
            m = torch.ones(coords.size(0), dtype=torch.bool)
            return coords, m
        dist = torch.cdist(coords, ligand_coords)          # [N_res, N_lig]
        m = (dist.min(dim=1).values < self.pocket_radius)   # [N_res]
        return coords[m], m
    
    def build(
        self,
        pdb_path: str | pathlib.Path,
        seq: Optional[str] = None,
        esm: Optional[torch.Tensor] = None,
        ligand_coords: Optional[torch.Tensor] = None,
    ) -> Data:
        """Construct a **torch_geometric** graph for the given PDB file.

        If *esm* is provided, it is concatenated to Cα coordinates (re‑creates
        the old 1 283‑D representation).  Otherwise the node feature is:

            one‑hot(20) + charge(1) + coords(3) = 24‑D
        """
        pdb_path = pathlib.Path(pdb_path)
        coords, aa3 = self._extract_ca(pdb_path)

        # --- pocket cropping ----------------------------------------------
        coords, mask = self._crop_to_pocket(coords, ligand_coords)
        aa3 = [a for a, keep in zip(aa3, mask) if keep]
        if esm is not None:
            if coords.shape[0] != esm[mask].shape[0]:
                raise ValueError("Seq/structure length mismatch for ESM embedding")
            node_x = torch.cat([esm[mask], coords], dim=1)
        else:
            if seq is not None and len(seq) != coords.shape[0]:
                raise ValueError("Seq/structure length mismatch for one‑hot pathway")
            one_hot = self._aa_one_hot(aa3)
            charges = self._charges(aa3)
            node_x = torch.cat([one_hot, charges, coords], dim=1)

        dist = torch.cdist(coords, coords)
        within = (dist < self.cutoff) & (dist > 0)
        idx = torch.arange(coords.shape[0] - 1, dtype=torch.long)
        within[idx, idx + 1] = True
        within[idx + 1, idx] = True

        edge_index = within.nonzero(as_tuple=False).t()
        edge_attr = dist[within].unsqueeze(-1)

        return Data(x=node_x, pos=coords, edge_index=edge_index, edge_attr=edge_attr)

    # ------------------------------------------------------------------
    # Caching helpers (optional)
    # ------------------------------------------------------------------
    def load(self, chembl_id: str) -> Data:
        path = self.graph_dir / f"{chembl_id}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Protein graph not found: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def save(self, chembl_id: str, data: Data):
        path = self.graph_dir / f"{chembl_id}.pt"
        torch.save(data, path)

        n = coords.size(0)
        mask = (dist < self.cutoff) & (dist > 0)
        idx = torch.arange(n - 1)
        mask[idx, idx + 1] = True  # peptide bond
        mask[idx + 1, idx] = True

        edge_index = mask.nonzero(as_tuple=False).t()
        edge_attr = dist[mask].unsqueeze(-1)
        return Data(x=x, pos=coords, edge_index=edge_index, edge_attr=edge_attr)

    def save(self, chembl_id: str, data: Data):
        torch.save(data, self.graph_dir / f"{chembl_id}.pt")

    def load(self, chembl_id: str) -> Data:
        return torch.load(self.graph_dir / f"{chembl_id}.pt", map_location="cpu")


# ---------------------------------------------------------------------------
# Bulk generation CLI
# ---------------------------------------------------------------------------

def _unique_targets(dataset: str):
    from tdc.multi_pred import DTI
    data = DTI(name=dataset.upper())
    df = data.get_data()
    return sorted(df["Target"].unique())


def _process_one(acc: str, builder: ProteinGraphBuilder):
    try:
        pdb_path = builder._fetch_pdb(acc)
        G = builder.build(pdb_path)
        builder.save(acc, G)
        return True, acc
    except Exception as e:
        return False, f"{acc}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["KIBA", "DAVIS", "CD4C"], required=True)
    ap.add_argument("--out-dir", default="../data/protein_graphs")
    ap.add_argument("--cutoff", type=float, default=10.0)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    builder = ProteinGraphBuilder(graph_dir=args.out_dir, cutoff=args.cutoff)

    if args.dataset.upper() in {"KIBA", "DAVIS"}:
        targets = _unique_targets(args.dataset)
    else:  # CD4C – read CSV
        csv = pathlib.Path("../data/filtered_cancer_all.csv")
        if not csv.exists():
            print("CD4C csv not found – abort", file=sys.stderr)
            sys.exit(1)
        targets = sorted(pd.read_csv(csv)["Target_ID"].unique())

    print(f"Building graphs for {len(targets)} proteins → {args.out_dir}")

    with mp.Pool(processes=args.num_workers) as pool:
        for ok, msg in pool.imap_unordered(lambda t: _process_one(t, builder), targets):
            status = "✔" if ok else "✗"
            print(f"[{status}] {msg}")


if __name__ == "__main__":
    main()
