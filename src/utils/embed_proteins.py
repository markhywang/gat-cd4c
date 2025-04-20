### Updated `embed_proteins.py`
#!/usr/bin/env python
"""
embed_proteins.py
-----------------
Create one *fixed‑width* 1 283‑D vector per protein, loading graphs on demand and using CPU-only loads.

Outputs
-------
- protein_embeddings.csv
- protein_graphs/<ID>.pt   (only if --save-graphs)
"""

import argparse
import pathlib
import requests
import pandas as pd
import torch
from urllib.parse import quote
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
from Bio.PDB import PDBParser
from torch_geometric.data import Data
from chembl_webresource_client.new_client import new_client
from Bio import SeqIO
from io import StringIO
import os

hf_logging.set_verbosity_error()

CANONICAL_OVERRIDES = {
    "O00688": "P00533",
    "A6NCG0": "P09104",
}

def canonical_uniprot(acc: str) -> str:
    if acc in CANONICAL_OVERRIDES:
        return CANONICAL_OVERRIDES[acc]
    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{quote(acc)}.json", timeout=15)
    r.raise_for_status()
    return r.json()["primaryAccession"]


def download_af_pdb(acc: str, cache_dir="af_cache", model_tags=("v4","v3","v2","v1")) -> pathlib.Path | None:
    canon = canonical_uniprot(acc)
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    for tag in model_tags:
        fname = f"AF-{canon}-F1-model_{tag}.pdb"
        out = cache_dir / fname
        if out.exists():
            return out
        url = f"https://alphafold.ebi.ac.uk/files/{fname}"
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            out.write_bytes(r.content)
            return out
        if r.status_code not in (404,):
            r.raise_for_status()
    return None

class ESMResidueEmbedder:
    def __init__(self, model="facebook/esm2_t33_650M_UR50D"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, seq: str) -> torch.Tensor:
        toks = self.tokenizer(seq, add_special_tokens=False, return_tensors="pt").to(self.device)
        return self.model(**toks).last_hidden_state.squeeze(0).cpu()

GLOBAL_ESM = ESMResidueEmbedder()

class ProteinGraphBuilder:
    def __init__(self, graph_dir="../../data/protein_graphs", cutoff=10.0):
        self.cutoff = cutoff
        self.graph_dir = graph_dir
        self.parser = PDBParser(QUIET=True)

    def _ca_atoms(self, pdb_path):
        ca = []
        s = self.parser.get_structure("af", str(pdb_path))
        for model in s:
            for chain in model:
                for res in chain:
                    if "CA" in res:
                        ca.append(torch.tensor(res["CA"].coord, dtype=torch.float32))
        return torch.stack(ca)

    def build(self, pdb_path, esm):
        coords = self._ca_atoms(pdb_path)
        if coords.shape[0] != esm.shape[0]:
            raise ValueError("Seq/structure length mismatch")
        node_x = torch.cat([esm, coords], dim=1)
        dist = torch.cdist(coords, coords)
        mask = (dist < self.cutoff) & (dist > 0)
        edge_index = mask.nonzero(as_tuple=False).t()
        edge_attr = dist[mask].unsqueeze(-1)
        return Data(x=node_x, pos=coords, edge_index=edge_index, edge_attr=edge_attr)

    def load(self, chembl_id: str) -> Data:
        path = os.path.join(self.graph_dir, f"{chembl_id}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Protein graph not found: {path}")
        # load only weights, map to CPU to avoid GPU memory
        return torch.load(path, map_location='cpu', weights_only=False)

EMB_DIM = 1283

def main(csv_in: pathlib.Path, out_csv: pathlib.Path, save_graphs: bool):
    df = pd.read_csv(csv_in)
    df = df.drop_duplicates(subset="Target_ID", keep="first")
    rows, gdir = [], pathlib.Path("../../data/protein_graphs")
    if save_graphs:
        gdir.mkdir(exist_ok=True)
    for pid, seq in df[["Target_ID","protein"]].itertuples(index=False):
        uniprot = chembl_to_uniprot(pid)
        if not uniprot:
            continue
        vec = embed_one(seq, pid, uniprot, gdir if save_graphs else None)
        rows.append([pid, *vec.numpy()])
    cols = ["Target_ID"] + [f"emb_{i}" for i in range(EMB_DIM)]
    pd.DataFrame(rows, columns=cols).to_csv(out_csv, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../../data/filtered_cancer_all.csv", type=pathlib.Path)
    ap.add_argument("--output", default="../../data/protein_embeddings.csv", type=pathlib.Path)
    ap.add_argument("--save-graphs", action="store_true")
    args = ap.parse_args()
    main(args.input, args.output, args.save_graphs)
