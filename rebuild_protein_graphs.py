"""
rebuild_protein_graphs.py
--------------------------
Converts existing protein graph .pt files from 1283-D (ESM-2 + coords)
to 24-D (one-hot AA + charge + coords) node features.

Sequence data is taken from filtered_cancer_all.csv (the 'protein' column).
3-D Cα coordinates and graph topology (edge_index, edge_attr) are preserved
from the existing .pt files — no PDB files needed.

Usage:
    python rebuild_protein_graphs.py
    python rebuild_protein_graphs.py --data_path data --graph_dir data/protein_graphs --dry_run
"""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd
import torch
from torch_geometric.data import Data

from src.utils.embed_proteins import AA_TO_IDX, AA_CHARGE, NUM_AA


def aa_one_hot(seq: str) -> torch.Tensor:
    """One-hot encode a protein sequence string. Unknown residues → all zeros."""
    t = torch.zeros((len(seq), NUM_AA), dtype=torch.float32)
    for i, aa in enumerate(seq):
        idx = AA_TO_IDX.get(aa, None)
        if idx is not None:
            t[i, idx] = 1.0
    return t


def aa_charges(seq: str) -> torch.Tensor:
    t = torch.zeros((len(seq), 1), dtype=torch.float32)
    for i, aa in enumerate(seq):
        t[i, 0] = AA_CHARGE.get(aa, 0)
    return t


def rebuild(graph_dir: pathlib.Path, sequences: dict[str, str], dry_run: bool) -> None:
    pt_files = sorted(graph_dir.glob("*.pt"))
    print(f"Found {len(pt_files)} protein graph files.")

    converted, skipped = 0, 0
    for pt_path in pt_files:
        chembl_id = pt_path.stem
        if chembl_id not in sequences:
            print(f"  [SKIP] {chembl_id} — no sequence in CSV")
            skipped += 1
            continue

        seq = sequences[chembl_id]
        old_graph = torch.load(pt_path, map_location="cpu", weights_only=False)

        if old_graph.x.shape[1] == 24:
            print(f"  [OK]   {chembl_id} already 24-D, skipping")
            skipped += 1
            continue

        coords = old_graph.pos  # [N, 3]  — preserved from original build
        N = old_graph.x.shape[0]

        if N != len(seq):
            # Graph was likely pocket-cropped; can't map sequence back.
            # Use zero one-hot + zero charge, keep coordinates so the graph
            # remains structurally valid.
            print(
                f"  [WARN] {chembl_id}: graph has {N} nodes but seq length is "
                f"{len(seq)} — using zero AA features (coords preserved)"
            )
            one_hot = torch.zeros((N, NUM_AA), dtype=torch.float32)
            charges = torch.zeros((N, 1),      dtype=torch.float32)
        else:
            one_hot = aa_one_hot(seq)          # [N, 20]
            charges = aa_charges(seq)          # [N, 1]

        new_x = torch.cat([one_hot, charges, coords], dim=1)  # [N, 24]

        new_graph = Data(
            x=new_x,
            pos=coords,
            edge_index=old_graph.edge_index,
            edge_attr=old_graph.edge_attr,
        )

        if not dry_run:
            torch.save(new_graph, pt_path)

        print(
            f"  [{'DRY' if dry_run else 'OK'}]  {chembl_id}: "
            f"{old_graph.x.shape[1]}-D → {new_x.shape[1]}-D  ({old_graph.x.shape[0]} residues)"
        )
        converted += 1

    print(
        f"\nDone. Converted: {converted}  Skipped: {skipped}"
        + ("  (dry run — no files written)" if dry_run else "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild protein graphs with 24-D features")
    parser.add_argument("--data_path",  default="data",                  help="Path to data directory")
    parser.add_argument("--graph_dir",  default="data/protein_graphs",   help="Path to protein_graphs directory")
    parser.add_argument("--dry_run",    action="store_true",              help="Preview changes without writing files")
    args = parser.parse_args()

    data_path  = pathlib.Path(args.data_path)
    graph_dir  = pathlib.Path(args.graph_dir)
    csv_path   = data_path / "filtered_cancer_all.csv"

    df = pd.read_csv(csv_path)
    sequences: dict[str, str] = df.groupby("Target_ID")["protein"].first().to_dict()
    print(f"Loaded sequences for {len(sequences)} proteins from {csv_path}")

    rebuild(graph_dir, sequences, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
