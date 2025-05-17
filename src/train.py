# === train.py (updated) =========================================================
import math
import argparse
import os
import multiprocessing as mp
from functools import partial
import csv
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

try:
    from model import DualGraphAttentionNetwork
    from utils.dataset import DrugProteinDataset, collate_drug_prot
    from utils.helper_functions import (
        set_seeds,
        count_params,
        try_run
    )
except ImportError:
    # For when the module is imported from outside
    from src.model import DualGraphAttentionNetwork
    from src.utils.dataset import DrugProteinDataset, collate_drug_prot
    from src.utils.helper_functions import (
        set_seeds,
        count_params,
        try_run
    )

torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# TRAIN LOOP
# ---------------------------------------------------------------------------

def train_model(args: argparse.Namespace, m_device: torch.device) -> None:
    """Full training routine that works for *CD4C*, **DAVIS**, or **KIBA**.

    All splits are served through :class:`DrugProteinDataset`, which pulls the
    pre‑built protein *graphs* (``protein_graphs/<ID>.pt``) created by the
    revised **embed_proteins.py** utility.
    """
    set_seeds(args.seed)
    device = m_device

    model = DualGraphAttentionNetwork(
        drug_in_features=29,
        prot_in_features=24,          # one‑hot(20) + charge + coords(3)
        hidden_size=args.hidden_size,
        emb_size=args.emb_size or args.hidden_size,
        drug_edge_features=17,
        prot_edge_features=1,
        num_layers=args.num_layers,
        num_heads=args.num_attn_heads,
        dropout=args.dropout,
        mlp_dropout=args.mlp_dropout,
        pooling_dim=args.pooling_dim,
        mlp_hidden=args.mlp_hidden,
        device=device,
    ).to(device)

    print(f"Model parameters: {count_params(model):,}")

    # ------------------------------------------------------------------
    # dataset + dataloader
    # ------------------------------------------------------------------
    train_ds, val_ds, _ = load_data(
        data_path=args.data_path,
        seed=args.seed,
        frac_train=args.frac_train,
        frac_val=args.frac_validation,
        frac_test=args.frac_test,
        use_small=args.use_small_dataset,
        protein_graph_dir=args.protein_graph_dir,
        dataset=args.dataset,
    )

    ctx = mp.get_context("spawn")
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
        multiprocessing_context=ctx,
        prefetch_factor=4,
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        collate_fn=partial(
            collate_drug_prot,
            prot_graph_dir=args.protein_graph_dir,
            hard_limit=args.max_nodes,
            drug_edge_feats=17,
            prot_edge_feats=1,
        ),
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        collate_fn=partial(
            collate_drug_prot,
            prot_graph_dir=args.protein_graph_dir,
            hard_limit=args.max_nodes,
            drug_edge_feats=17,
            prot_edge_feats=1,
        ),
        **loader_kwargs,
    )

    # ------------------------------------------------------------------
    # optimisation setup
    # ------------------------------------------------------------------
    loss_func = nn.SmoothL1Loss(beta=args.huber_beta)
    optimiser = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=args.scheduler_factor, patience=args.scheduler_patience
    )

    best_val = float("inf")
    no_imp = 0

    csv_path = f"{args.model_path}.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)

    columns = [
        "train_loss",
        "validation_loss",
        "train_acc",
        "validation_acc",
        "train_mse",
        "validation_mse",
        "train_mae",
        "validation_mae",
    ]
    pd.DataFrame(columns=columns).to_csv(csv_path, index=False)

    # ------------------------------------------------------------------
    # epochs
    # ------------------------------------------------------------------
    for epoch in range(args.max_epochs):
        model.train()
        total = dict(loss=0.0, acc=0.0, mse=0.0, mae=0.0)
        seen = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.max_epochs}"):
            (d_z, d_x, d_e, d_a, p_z, p_x, p_e, p_a, labels) = [x.to(device) for x in batch]

            optimiser.zero_grad(set_to_none=True)
            preds = model(d_z, d_x, d_e, d_a, p_z, p_x, p_e, p_a).squeeze(-1)
            loss = loss_func(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            n = labels.size(0)
            seen += n
            total["loss"] += loss.item() * n
            total["acc"] += accuracy_func(preds, labels, 1.0)
            total["mse"] += mse_func(preds, labels) * n
            total["mae"] += mae_func(preds, labels) * n

        # --- aggregate train metrics ---
        tl = total["loss"] / seen
        ta = total["acc"] / seen
        tm = total["mse"] / seen
        tma = total["mae"] / seen

        # --- validation ------------------------------------------------
        v_loss, v_acc, v_mse, v_mae = get_validation_metrics(val_loader, model, loss_func, device)
        scheduler.step(v_loss)

        print(
            f"Epoch {epoch+1}/{args.max_epochs} | "
            f"Train L={tl:.4f} MSE={tm:.4f} MAE={tma:.4f} Acc={ta:.4f}  ||  "
            f"Val L={v_loss:.4f} MSE={v_mse:.4f} MAE={v_mae:.4f} Acc={v_acc:.4f}"
        )

        if v_loss < best_val:
            best_val = v_loss
            no_imp = 0
            torch.save(model.state_dict(), args.model_path)
        else:
            no_imp += 1
            if no_imp >= args.stoppage_epochs:
                print("\nEarly‑stopping – no improvement\n")
                break

        pd.DataFrame(
            [[tl, v_loss, ta, v_acc, tm, v_mse, tma, v_mae]], columns=columns
        ).to_csv(csv_path, mode="a", header=False, index=False)


# ---------------------------------------------------------------------------
# EVALUATION HELPERS
# ---------------------------------------------------------------------------

def get_validation_metrics(loader, model, loss_func, device):
    model.eval()
    totals = dict(loss=0.0, acc=0.0, mse=0.0, mae=0.0)
    seen = 0
    with torch.no_grad():
        for batch in loader:
            (d_z, d_x, d_e, d_a, p_z, p_x, p_e, p_a, labels) = [x.to(device) for x in batch]
            preds = model(d_z, d_x, d_e, d_a, p_z, p_x, p_e, p_a).squeeze(-1)
            loss = loss_func(preds, labels).item()
            n = labels.size(0)
            seen += n
            totals["loss"] += loss * n
            totals["acc"] += accuracy_func(preds, labels, 1.0)
            totals["mse"] += mse_func(preds, labels) * n
            totals["mae"] += mae_func(preds, labels) * n

    return (
        totals["loss"] / seen,
        totals["acc"] / seen,
        totals["mse"] / seen,
        totals["mae"] / seen,
    )


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_data(
    data_path: str,
    seed: int,
    frac_train: float,
    frac_val: float,
    frac_test: float,
    use_small: bool,
    protein_graph_dir: str,
    dataset: str,
):
    """Return three :class:`DrugProteinDataset` objects (train/val/test)."""

    assert math.isclose(frac_train + frac_val + frac_test, 1.0), "Fractions must sum to 1"

    def _keep_cols(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in df.columns if c in ("smiles", "Target_ID", "pChEMBL_Value")]
        return df.loc[:, cols].copy()

    if dataset.upper() in {"KIBA", "DAVIS"}:
        # --- pull from TDC --------------------------------------------------
        from tdc.multi_pred import DTI

        data = DTI(name=dataset.upper())
        split = data.get_split("random", seed=seed, frac=[frac_train, frac_val, frac_test])
        tr, val, te = split["train"], split["valid"], split["test"]
        for df in (tr, val, te):
            df.rename(columns={"Drug": "smiles", "Target": "Target_ID", "Y": "pChEMBL_Value"}, inplace=True)

        tr, val, te = map(_keep_cols, (tr, val, te))
        prot_emb = pd.DataFrame()  # <- not used with graph‑only pathway

    else:
        # --- legacy CD4C csv -----------------------------------------------
        fname = "filtered_cancer_small.csv" if use_small else "filtered_cancer_all.csv"
        df = pd.read_csv(f"{data_path}/{fname}")
        df["strat"] = pd.qcut(df["pChEMBL_Value"], 10, duplicates="drop", labels=False)

        tr, rem = train_test_split(df, test_size=frac_val + frac_test, stratify=df["strat"], random_state=seed)
        val, te = train_test_split(rem, test_size=frac_test / (frac_val + frac_test), stratify=rem["strat"], random_state=seed)
        tr, val, te = map(lambda d: d.drop(columns="strat"), (tr, val, te))
        tr, val, te = map(_keep_cols, (tr, val, te))

        # although *not* used, we keep reading to satisfy constructor sign.↵
        prot_emb = pd.read_csv(f"{data_path}/protein_embeddings.csv", index_col=0)

    train_ds = DrugProteinDataset(tr, prot_emb, protein_graph_dir)
    val_ds = DrugProteinDataset(val, prot_emb, protein_graph_dir)
    test_ds = DrugProteinDataset(te, prot_emb, protein_graph_dir)
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# CLI ARGUMENTS
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # dataset / paths
    p.add_argument("--dataset", type=str, default="CD4C", choices=["CD4C", "KIBA", "DAVIS"], help="Which dataset to train on")
    p.add_argument("--data_path", type=str, default="../data", help="Root data directory")
    p.add_argument("--protein_graph_dir", type=str, default="../data/protein_graphs", help="Directory with *.pt protein graphs")

    # training splits
    p.add_argument("--frac_train", type=float, default=0.7)
    p.add_argument("--frac_validation", type=float, default=0.15)
    p.add_argument("--frac_test", type=float, default=0.15)

    # optimisation
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--scheduler_patience", type=int, default=10)
    p.add_argument("--scheduler_factor", type=float, default=0.5)
    p.add_argument("--huber_beta", type=float, default=1.0)
    p.add_argument("--stoppage_epochs", type=int, default=10)

    # architecture
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--emb_size", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--num_attn_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--mlp_dropout", type=float, default=0.2)
    p.add_argument("--pooling_dim", type=int, default=128)
    p.add_argument("--mlp_hidden", type=int, default=128)
    p.add_argument("--max_nodes", type=int, default=256)

    # misc
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_path", type=str, default="model.pt")
    p.add_argument("--use_small_dataset", action="store_true")
    return p


if __name__ == "__main__":
    parser = get_parser()
    cli_args = parser.parse_args()

    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    train_model(cli_args, dev)