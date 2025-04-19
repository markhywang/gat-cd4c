#!/usr/bin/env python
"""
embed_proteins.py
-----------------
Create one *fixed‑width* 1 283‑D vector per protein:

    • 1 280 dims  = mean‑pooled ESM‑2 residue embeddings
    •    3 dims  = mean xyz (0,0,0 when no AlphaFold structure)

Outputs
-------
- protein_embeddings.csv
- protein_graphs/<ID>.pt   (only if --save-graphs)

Requires
--------
pip install pandas torch requests biopython torch-geometric \
            transformers chembl_webresource_client
"""

# ------------------------------------------------------------------ #
# 0.  Imports & one‑time objects                                     #
# ------------------------------------------------------------------ #
import argparse, pathlib, requests, pandas as pd, torch
from urllib.parse import quote
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
from Bio.PDB import PDBParser
from torch_geometric.data import Data
from chembl_webresource_client.new_client import new_client

hf_logging.set_verbosity_error()          # silence pooler warning

# ------------------------------------------------------------------ #
# 1.  Helpers                                                        #
# ------------------------------------------------------------------ #

def canonical_uniprot(acc: str) -> str:
    """Map secondary / isoform accession → primary canonical accession."""
    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{quote(acc)}.json",
                     timeout=15)
    r.raise_for_status()
    return r.json()["primaryAccession"]


def download_af_pdb(acc: str, cache_dir="af_cache") -> pathlib.Path | None:
    """Return local path of AlphaFold PDB or None if it does not exist."""
    canon = canonical_uniprot(acc)
    url   = f"https://alphafold.ebi.ac.uk/files/AF-{canon}-F1-model_v4.pdb"
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    out = cache_dir / f"{canon}.pdb"

    if out.exists():
        return out
    r = requests.get(url, timeout=20)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    out.write_bytes(r.content)
    return out


class ESMResidueEmbedder:
    """Return (L,1280) tensor of per‑residue embeddings (ESM‑2)."""
    def __init__(self, model="facebook/esm2_t33_650M_UR50D"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model     = AutoModel.from_pretrained(model).to(device).eval()
        self.device    = device

    @torch.no_grad()
    def __call__(self, seq: str) -> torch.Tensor:
        toks = self.tokenizer(seq, add_special_tokens=False,
                              return_tensors="pt").to(self.device)
        return self.model(**toks).last_hidden_state.squeeze(0).cpu()


GLOBAL_ESM = ESMResidueEmbedder()        # instantiate once


class ProteinGraphBuilder:
    """Build a residue‑level radius graph (PyTorch Geometric `Data`)."""
    def __init__(self, cutoff=10.0):
        self.cutoff = cutoff
        self.parser = PDBParser(QUIET=True)

    def _ca_atoms(self, pdb_path):
        ca = []
        s = self.parser.get_structure("af", str(pdb_path))
        for model in s:
            for chain in model:
                for res in chain:
                    if "CA" in res:
                        ca.append(torch.tensor(res["CA"].coord, dtype=torch.float32))
        return torch.stack(ca)   # (L,3)

    def build(self, pdb_path, esm):
        coords = self._ca_atoms(pdb_path)
        if coords.shape[0] != esm.shape[0]:
            raise ValueError("Seq/structure length mismatch")
        node_x = torch.cat([esm, coords], dim=1)      # (L,1283)
        dist = torch.cdist(coords, coords)
        mask = (dist < self.cutoff) & (dist > 0)
        edge_index = mask.nonzero(as_tuple=False).t()
        edge_attr  = dist[mask].unsqueeze(-1)
        return Data(x=node_x, pos=coords,
                    edge_index=edge_index, edge_attr=edge_attr)


def chembl_to_uniprot(chembl_id: str) -> str | None:
    """Query ChEMBL for UniProt accession; return None if not found."""
    tgt = new_client.target.get(chembl_id)
    for comp in tgt["target_components"]:
        for xr in comp["target_component_xrefs"]:
            if xr["xref_src_db"] == "UniProt":
                return xr["xref_id"]
    return None


def embed_one(seq: str, uniprot: str,
              save_graph_dir: pathlib.Path | None = None) -> torch.Tensor:
    """
    Return a 1 283‑D vector.
    If AlphaFold structure exists → ESM+xyz; otherwise pad xyz with zeros.
    """
    esm = GLOBAL_ESM(seq)                         # (L,1280)
    pdb_path = download_af_pdb(uniprot)

    if pdb_path is None:                          # fallback: zeros for xyz
        return torch.cat([esm.mean(0), torch.zeros(3)], dim=0)

    graph = ProteinGraphBuilder().build(pdb_path, esm)
    if save_graph_dir is not None:
        torch.save(graph, save_graph_dir / f"{uniprot}.pt")
    return graph.x.mean(dim=0)                    # (1283,)


# ------------------------------------------------------------------ #
# 2.  CSV driver                                                     #
# ------------------------------------------------------------------ #

EMB_DIM = 1283

def main(csv_in: pathlib.Path,
         out_csv: pathlib.Path,
         save_graphs: bool):

    df = pd.read_csv(csv_in)
    req_cols = {"Target_ID", "protein"}
    if not req_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {req_cols}")
    
    df = df.drop_duplicates(subset="Target_ID", keep="first")

    rows, gdir = [], pathlib.Path("protein_graphs")
    if save_graphs:
        gdir.mkdir(exist_ok=True)

    for pid, seq in df[["Target_ID", "protein"]].itertuples(index=False):
        uniprot = chembl_to_uniprot(pid)
        if not uniprot:
            print(f"{pid}: no UniProt – skipped")
            continue
        try:
            vec = embed_one(seq, uniprot, gdir if save_graphs else None)
            rows.append([pid, *vec.numpy()])
            print("✓", pid)
        except Exception as e:
            print(f"{pid}: {e}")

    cols = ["Target_ID"] + [f"emb_{i}" for i in range(EMB_DIM)]
    pd.DataFrame(rows, columns=cols).to_csv(out_csv, index=False)
    print("Wrote", out_csv)


# ------------------------------------------------------------------ #
# 3.  CLI                                                             #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  type=pathlib.Path,
                    default="../../data/filtered_cancer_all.csv")
    ap.add_argument("--output", type=pathlib.Path,
                    default="protein_embeddings.csv")
    ap.add_argument("--save-graphs", action="store_true")
    args = ap.parse_args()

    main(args.input, args.output, args.save_graphs)
