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
                            --num-workers 8 [--use-local-colabfold]
   ```

   The script pulls the desired **TDC** dataset, resolves each *Target_ID* (which
   is already a UniProt accession for KIBA/DAVIS) or raw sequence, obtains a PDB
   via the **ColabFold** local pipeline (if `--use-local-colabfold` is set) or
   remote AlphaFold DB, builds the residue graph (one‑hot + charge + Cα coords) and
   saves it as `<Target_ID>.pt`.

### Install ColabFold Locally

ColabFold dramatically reduces disk requirements (~50 GB vs 2.6 TB) by using MMseqs2.

```bash
# Create a dedicated conda env
conda create -n colabfold -c conda-forge python=3.9
conda activate colabfold
# Install core dependencies
conda install -c conda-forge openmm biopython jax jaxlib
# Install ColabFold batch script
pip install colabfold
# Make sure colabfold_batch is on your PATH
```

### Example Run of ColabFold Batch

```bash
# single FASTA to structure
colabfold_batch --fasta your_protein.fasta --output_dir ./cf_out --amber
```

Node features
-------------
* **one‑hot(20)** standard amino‑acid alphabet
* **charge (1)** integer {‑1,0,+1} at pH≈7
* **coords (3)** Cα x,y,z in Å
Total: **24** dims.

Edges
-----
* Peptide‑bond neighbours
* Any residue pair with Cα‑Cα distance < *cut‑off* (default **10 Å**).
Edge feature = single scalar distance (Å).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import sys
import hashlib
import re
import time
from typing import Dict, Optional
import subprocess

import requests
import torch
import pandas as pd
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
_UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"

# Regex for UniProt accession
UNIPROT_REGEX = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]{5}|[A-N][0-9][A-Z0-9]{3}[0-9]{4}|[A-Z]{3}[0-9]{7})$"
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def looks_like_uniprot(acc: str) -> bool:
    return bool(UNIPROT_REGEX.match(acc))


def uniprot_exists(acc: str) -> bool:
    url = _UNIPROT_FASTA.format(acc=acc)
    try:
        r = requests.head(url, timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def run_colabfold(fasta_name: str, seq: str, output_dir: pathlib.Path) -> pathlib.Path:
    """
    Run ColabFold locally via the `colabfold_batch` CLI.
    Expects `colabfold_batch` on PATH.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_file = output_dir / f"{fasta_name}.fasta"
    with open(fasta_file, 'w') as f:
        f.write(f">{fasta_name}\n{seq}\n")
    # call ColabFold batch
    cmd = [
        "colabfold_batch",
        "--fasta", str(fasta_file),
        "--output_dir", str(output_dir),
        "--amber"
    ]
    subprocess.run(cmd, check=True)
    pdb_path = output_dir / f"{fasta_name}_unrelaxed_rank_1_model_1.pdb"
    if not pdb_path.exists():
        raise RuntimeError(f"ColabFold did not produce PDB for {fasta_name}")
    return pdb_path

# ---------------------------------------------------------------------------
class ProteinGraphBuilder:
    """Construct, cache and load residue‑level graphs."""

    def __init__(
        self,
        graph_dir: str = "../data/protein_graphs",
        cutoff: float = 10.0,
        use_colabfold: bool = False
    ):
        self.cutoff = cutoff
        self.graph_dir = pathlib.Path(graph_dir)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.parser = PDBParser(QUIET=True)
        self.use_colabfold = use_colabfold

    def _three_to_one(self, resname: str) -> str:
        tbl = {
            "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G",
            "HIS":"H","ILE":"I","LYS":"K","LEU":"L","MET":"M","ASN":"N",
            "PRO":"P","GLN":"Q","ARG":"R","SER":"S","THR":"T","VAL":"V",
            "TRP":"W","TYR":"Y"
        }
        return tbl.get(resname.upper(), "X")

    def _fetch_pdb(self, acc: str) -> pathlib.Path:
        # Determine sequence or UniProt accession
        is_uniprot = looks_like_uniprot(acc) and uniprot_exists(acc)
        seq = acc
        name = acc
        # fetch remote if not using ColabFold
        if not self.use_colabfold:
            if is_uniprot:
                pdb_url = _AF_URL.format(acc=acc)
                out = self.graph_dir / f"AF-{acc}-F1-model_v4.pdb"
                if out.exists():
                    return out
                r = requests.get(pdb_url, timeout=20)
                if r.status_code != 200:
                    raise RuntimeError(f"AlphaFold PDB not found for {acc}")
                out.write_bytes(r.content)
                return out
            else:
                raise RuntimeError(f"Cannot fetch non-UniProt ID {acc} remotely")
        # use ColabFold for any
        return run_colabfold(name, seq, self.graph_dir)

    def _extract_ca(self, pdb_path: pathlib.Path):
        coords, aa3 = [], []
        struct = self.parser.get_structure("cf", str(pdb_path))
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
        return torch.cat([one_hot, charges, coords], dim=1)

    def build(self, pdb_path: pathlib.Path | str) -> Data:
        pdb_path = pathlib.Path(pdb_path)
        coords, aa3 = self._extract_ca(pdb_path)
        x = self._node_feats(aa3, coords)
        dist = torch.cdist(coords, coords)
        n = coords.size(0)
        mask = (dist < self.cutoff) & (dist > 0)
        idx = torch.arange(n - 1)
        mask[idx, idx + 1] = True
        mask[idx + 1, idx] = True
        edge_index = mask.nonzero(as_tuple=False).t()
        edge_attr = dist[mask].unsqueeze(-1)
        return Data(x=x, pos=coords, edge_index=edge_index, edge_attr=edge_attr)

    def save(self, chembl_id: str, data: Data):
        filename = f"{chembl_id}.pt" if len(chembl_id)<=50 else f"seq-{hashlib.md5(chembl_id.encode()).hexdigest()}.pt"
        torch.save(data, self.graph_dir/filename)

    def load(self, chembl_id: str) -> Data:
        filename = f"{chembl_id}.pt" if len(chembl_id)<=50 else f"seq-{hashlib.md5(chembl_id.encode()).hexdigest()}.pt"
        return torch.load(self.graph_dir/filename, map_location="cpu")

# ---------------------------------------------------------------------------
def _unique_targets(dataset: str):
    from tdc.multi_pred import DTI
    data=DTI(name=dataset.upper())
    targets=list(data.get_data()["Target"].unique())
    print(f"Found {len(targets)} total targets in {dataset}")
    return sorted(targets)


def _process_one(acc: str, builder: ProteinGraphBuilder):
    try:
        pdb=builder._fetch_pdb(acc)
        G=builder.build(pdb)
        builder.save(acc,G)
        return True,acc
    except Exception as e:
        return False,f"{acc}: {e}"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=["KIBA","DAVIS","CD4C"],required=True)
    p.add_argument("--out-dir",default="../data/protein_graphs")
    p.add_argument("--cutoff",type=float,default=10.0)
    p.add_argument("--num-workers",type=int,default=8)
    p.add_argument("--use-local-colabfold",action="store_true",
                   help="Use ColabFold locally instead of remote AlphaFold DB")
    args=p.parse_args()
    builder=ProteinGraphBuilder(
        graph_dir=args.out_dir,
        cutoff=args.cutoff,
        use_colabfold=args.use_local_colabfold
    )
    if args.dataset.upper() in {"KIBA","DAVIS"}:
        targets=_unique_targets(args.dataset)
    else:
        f=pathlib.Path("../data/filtered_cancer_all.csv")
        if not f.exists():sys.exit("CD4C csv not found")
        targets=sorted(pd.read_csv(f)["Target_ID"].unique())
    print(f"Building {len(targets)} proteins → {args.out_dir}")
    with mp.Pool(args.num_workers) as pool:
        for ok,msg in pool.imap_unordered(lambda x:_process_one(x,builder),targets):
            print(f"[{'✔' if ok else '✗'}] {msg}")

if __name__=="__main__":main()
