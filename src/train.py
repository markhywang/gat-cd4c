import math
import argparse
import pandas as pd
import contextlib
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim, nn
from functools import partial
import torch.nn.functional as F
import multiprocessing as mp

from model import DualGraphAttentionNetwork
from utils.embed_proteins import ProteinGraphBuilder
from utils.dataset import DrugProteinDataset
from utils.helper_functions import set_seeds, count_model_params, plot_loss_curves, accuracy_func, mse_func, mae_func

torch.set_float32_matmul_precision('high')

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
        hard_limit: int = 80,
        drug_edge_feats: int = 17,
        cross_cutoff: float = 5.0):
    """Collate a list of DrugProteinDataset items into a batch.

    Drug graphs remain **dense** (padded/cropped to hard_limit × hard_limit).
    Protein graphs remain **sparse** (PyG COO format, concatenated across the batch).
    Cross-graph edges connect drug atoms to protein residues within *cross_cutoff* Å,
    yielding a sparse [2, E_cross] index for biologically focused cross-attention.

    Batch item format (from DrugProteinDataset.__getitem__):
        d_n   [max_nodes, F_n_drug]  — drug node features (pre-padded)
        d_e   [max_nodes, max_nodes, F_e_drug]  — drug edge features
        d_a   [max_nodes, max_nodes]  — drug adjacency
        d_pos [max_nodes, 3]          — drug atom 3-D positions (RDKit ETKDGv3 placeholder)
        p_n   [N_prot, F_n_prot]      — protein node features (sparse, no padding)
        p_e   [E_prot, F_e_prot]      — protein edge attributes (COO)
        p_i   [2, E_prot]             — protein edge index (COO)
        p_pos [N_prot, 3]             — protein Cα positions
        label scalar

    Returns (10-tuple):
        drug_ns   [B, H, F_n_drug]
        drug_es   [B, H, H, F_e_drug]
        drug_as   [B, H, H]
        prot_ns   [N_total, F_n_prot]   sparse — concatenated protein nodes
        prot_eis  [2, E_prot_total]     sparse — offset-corrected edge index
        prot_eas  [E_prot_total, F_e]   sparse — protein edge attributes
        prot_batch[N_total]             batch assignment vector (0 … B-1)
        cross_ei  [2, E_cross]          sparse — drug global idx → prot global idx
        cross_ea  [E_cross, 1]          Euclidean distance per cross-edge (Å)
        labels    [B]
    """
    H = hard_limit
    drug_ns, drug_es, drug_as = [], [], []
    prot_ns_list, prot_ei_list, prot_ea_list, prot_batch_parts = [], [], [], []
    cross_ei_list, cross_ea_list = [], []
    labels = []

    prot_node_offset = 0  # cumulative protein node count for COO index offsetting

    for b_idx, (d_n, d_e, d_a, d_pos, p_n, p_e, p_i, p_pos, label) in enumerate(batch):

        # ── Drug: dense, pad/crop to H ─────────────────────────────────────
        drug_ns.append(pad_to(d_n,   (H, d_n.size(-1))))
        drug_es.append(pad_to(d_e,   (H, H, drug_edge_feats)))
        drug_as.append(pad_to(d_a,   (H, H)))
        drug_pos_b = pad_to(d_pos,   (H, 3))              # [H, 3]

        # ── Protein: sparse, truncate to H if necessary ────────────────────
        N = p_n.size(0)
        if N > H:
            keep = (p_i[0] < H) & (p_i[1] < H)
            p_i   = p_i[:, keep]
            p_e   = p_e[keep]
            p_pos = p_pos[:H]
            p_n   = p_n[:H]
            N     = H

        prot_ns_list.append(p_n)
        prot_ei_list.append(p_i + prot_node_offset)       # offset into global node space
        prot_ea_list.append(p_e)
        prot_batch_parts.append(
            torch.full((N,), b_idx, dtype=torch.long)
        )

        # ── Cross-graph edges within cross_cutoff Å ────────────────────────
        # n_real_drug: non-padded rows have at least one non-zero feature
        n_real_drug = int((d_n.abs().sum(dim=-1) > 0).sum().item())
        n_real_drug = min(n_real_drug, H)

        c_ei, c_ea = ProteinGraphBuilder.build_cross_edges(
            drug_pos    = drug_pos_b,
            prot_pos    = p_pos,
            drug_offset = b_idx * H,
            prot_offset = prot_node_offset,
            n_real_drug = n_real_drug,
            cutoff      = cross_cutoff,
        )
        if c_ei.size(1) > 0:
            cross_ei_list.append(c_ei)
            cross_ea_list.append(c_ea)

        prot_node_offset += N
        labels.append(label)

    # ── Assemble sparse protein tensors ────────────────────────────────────
    prot_ns   = torch.cat(prot_ns_list, dim=0)            # [N_total, F_n_prot]
    prot_eis  = torch.cat(prot_ei_list, dim=1)            # [2, E_prot_total]
    prot_eas  = torch.cat(prot_ea_list, dim=0)            # [E_prot_total, F_e]
    prot_batch= torch.cat(prot_batch_parts, dim=0)        # [N_total]

    if cross_ei_list:
        cross_ei = torch.cat(cross_ei_list, dim=1)        # [2, E_cross]
        cross_ea = torch.cat(cross_ea_list, dim=0)        # [E_cross, 1]
    else:
        cross_ei = torch.zeros((2, 0), dtype=torch.long)
        cross_ea = torch.zeros((0, 1), dtype=torch.float32)

    return (
        torch.stack(drug_ns),                             # [B, H, F_n_drug]
        torch.stack(drug_es),                             # [B, H, H, F_e_drug]
        torch.stack(drug_as),                             # [B, H, H]
        prot_ns,                                          # [N_total, F_n_prot]
        prot_eis,                                         # [2, E_prot_total]
        prot_eas,                                         # [E_prot_total, F_e]
        prot_batch,                                       # [N_total]
        cross_ei,                                         # [2, E_cross]
        cross_ea,                                         # [E_cross, 1]
        torch.tensor(labels, dtype=torch.float32),        # [B]
    )


def train_model(args: argparse.Namespace, m_device: torch.device) -> None:
    set_seeds()
    device = m_device

    model = DualGraphAttentionNetwork(
        drug_in_features=29,
        prot_in_features=1283,
        hidden_size=args.hidden_size,
        emb_size=getattr(args, "emb_size", args.hidden_size),
        drug_edge_features=17,
        prot_edge_features=1,
        num_layers=args.num_layers,
        num_heads=args.num_attn_heads,
        dropout=args.dropout,
        mlp_dropout=args.mlp_dropout,
        pooling_dim=args.pooling_dim,
        mlp_hidden=getattr(args, "mlp_hidden", 128),
        device=device
    ).to(device)

    # model = torch.compile(model, fullgraph=True)      # fuse the whole graph
    # torch._dynamo.config.dynamic_shapes = True       
    
    print(f'Model parameters: {count_model_params(model)}')

    train_ds, val_ds, _ = load_data(
        args.data_path, args.seed,
        args.frac_train, args.frac_validation, args.frac_test,
        args.use_small_dataset, args.protein_graph_dir
    )

    ctx = mp.get_context('spawn')

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=6,  # increase if CPU cores available
        pin_memory=True,  # beneficial for CUDA
        persistent_workers=True,
        prefetch_factor=4, # prefetch more batches
        multiprocessing_context=ctx,
    )

    collate_fn = partial(
        collate_drug_prot,
        hard_limit=args.max_nodes,
        drug_edge_feats=17,
        cross_cutoff=5.0,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  collate_fn=collate_fn, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, collate_fn=collate_fn, **loader_kwargs)

    loss_func = nn.SmoothL1Loss(beta=args.huber_beta)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.scheduler_factor, patience=args.scheduler_patience
    )

    best_val = float('inf')
    no_imp = 0
    metrics = pd.DataFrame(
        columns=[
            'train_loss',     'validation_loss',
            'train_acc',      'validation_acc',
            'train_mse',      'validation_mse',
            'train_mae',      'validation_mae'
        ],
        index=range(args.max_epochs)
    )

    
    for epoch in range(args.max_epochs):
        model.train()
        total = dict(loss=0, acc=0, mse=0, mae=0)
        samples = 0
    
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.max_epochs}"):
            d_n, d_e, d_a, p_n, p_ei, p_ea, p_batch, cross_ei, cross_ea, labels = batch
            d_n, d_e, d_a   = d_n.to(device),   d_e.to(device),   d_a.to(device)
            p_n, p_ei, p_ea = p_n.to(device),   p_ei.to(device),  p_ea.to(device)
            p_batch         = p_batch.to(device)
            cross_ei, cross_ea = cross_ei.to(device), cross_ea.to(device)
            labels          = labels.to(device)
            optimizer.zero_grad()

            # TODO: DualGraphAttentionNetwork.forward must be updated to accept
            # sparse protein tensors (p_n, p_ei, p_ea, p_batch) and cross-graph
            # edges (cross_ei, cross_ea) before training is functional.
            preds = model(d_n, d_e, d_a, p_n, p_ei, p_ea, p_batch, cross_ei, cross_ea).squeeze(-1)
            loss = loss_func(preds, labels)
    
            # backward + clip + step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
            # accumulate metrics
            n = labels.size(0)
            samples += n
            total['loss'] += loss.item() * n
            total['acc']  += accuracy_func(preds, labels, 1.0)
            total['mse']  += mse_func(preds, labels) * n
            total['mae']  += mae_func(preds, labels) * n
    
        # compute epoch‐level metrics…
        tl = total['loss'] / samples
        ta = total['acc'] / samples
        tm = total['mse'] / samples
        tma = total['mae'] / samples

        # validation
        v_loss,v_acc,v_mse,v_mae = get_validation_metrics(val_loader, model, loss_func, device)

        lr_scheduler.step(v_loss)

        row = [
            tl, v_loss,
            ta,  v_acc,
            tm,  v_mse,
            tma, v_mae,
        ]
        # unwrap any torch.Tensor into float
        row = [x.item() if isinstance(x, torch.Tensor) else float(x) for x in row]
        
        metrics.loc[epoch] = row     
        print(f"Epoch {epoch+1}/{args.max_epochs}: "
              f"Train Loss={tl:.5f}, MSE={tm:.5f}, MAE={tma:.5f}, Acc={ta:.5f} | "
              f"Val Loss={v_loss:.5f}, MSE={v_mse:.5f}, MAE={v_mae:.5f}, Acc={v_acc:.5f}"
        )

        if v_loss < best_val:
            best_val = v_loss
            no_imp = 0
            torch.save(model.state_dict(), '../models/model.pth')
        else:
            no_imp += 1
            if no_imp >= args.stoppage_epochs:
                break

    plot_loss_curves(metrics)


def get_validation_metrics(loader, model, loss_func, device):
    model.eval()
    total_samples = 0
    total_loss = 0
    total_acc = 0
    total_mse = 0
    total_mae = 0

    with torch.no_grad():
        for batch in loader:
            d_n, d_e, d_a, p_n, p_ei, p_ea, p_batch, cross_ei, cross_ea, labels = batch
            d_n, d_e, d_a   = d_n.to(device, non_blocking=True), d_e.to(device, non_blocking=True), d_a.to(device, non_blocking=True)
            p_n, p_ei, p_ea = p_n.to(device, non_blocking=True), p_ei.to(device, non_blocking=True), p_ea.to(device, non_blocking=True)
            p_batch         = p_batch.to(device, non_blocking=True)
            cross_ei        = cross_ei.to(device, non_blocking=True)
            cross_ea        = cross_ea.to(device, non_blocking=True)
            labels          = labels.to(device, non_blocking=True)
            # TODO: update model.forward for sparse protein + cross-edges.
            preds = model(d_n, d_e, d_a, p_n, p_ei, p_ea, p_batch, cross_ei, cross_ea).squeeze(-1)
            loss = loss_func(preds, labels).item()
            acc = accuracy_func(preds, labels, threshold=1.0)
            mse = mse_func(preds, labels)
            mae = mae_func(preds, labels)

            n = labels.shape[0]
            total_samples += n
            total_loss += loss * n
            total_acc += acc
            total_mse += mse * n
            total_mae += mae * n

    return (
        total_loss / total_samples,
        total_acc / total_samples,
        total_mse / total_samples,
        total_mae / total_samples
    )


def load_data(data_path, seed, frac_train, frac_val, frac_test, use_small, protein_graph_dir):
    assert math.isclose(frac_train + frac_val + frac_test, 1), \
        "Train/val/test fractions must sum to 1"

    file = 'filtered_cancer_small.csv' if use_small else 'filtered_cancer_all.csv'
    df = pd.read_csv(f'{data_path}/{file}')
    prot_emb = pd.read_csv(f'{data_path}/protein_embeddings.csv', index_col=0)

    df['stratify_col'] = df['Target_ID'] + '_' + df['label'].astype(str)
    tr, rem = train_test_split(df, test_size=frac_val+frac_test,
                               stratify=df['stratify_col'],
                               random_state=seed)
    val, te = train_test_split(rem, test_size=frac_test/(frac_val+frac_test),
                                stratify=rem['stratify_col'],
                                random_state=seed)

    tr = tr.drop(columns='stratify_col')
    val = val.drop(columns='stratify_col')
    te = te.drop(columns='stratify_col')

    train_ds = DrugProteinDataset(tr, prot_emb, protein_graph_dir)
    val_ds = DrugProteinDataset(val, prot_emb, protein_graph_dir)
    test_ds = DrugProteinDataset(te, prot_emb, protein_graph_dir)
    return train_ds, val_ds, test_ds


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_small_dataset", action="store_true",
                        help="Whether to use the small dataset")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for data loader")
    parser.add_argument("--stoppage_epochs", type=int, default=10,
                        help="Patience for early stopping")
    parser.add_argument("--max_epochs", type=int, default=128,
                        help="Maximum number of epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    parser.add_argument("--data_path", type=str, default='../data',
                        help="Path to interaction CSVs and embeddings")
    parser.add_argument("--protein_graph_dir", type=str, default='../../data/protein_graphs',
                        help="Directory containing saved protein .pt graphs")
    parser.add_argument("--frac_train", type=float, default=0.7,
                        help="Fraction of data for training")
    parser.add_argument("--frac_validation", type=float, default=0.15,
                        help="Fraction of data for validation")
    parser.add_argument("--frac_test", type=float, default=0.15,
                        help="Fraction of data for testing")

    parser.add_argument("--huber_beta", type=float, default=1.0,
                        help="Beta for Huber loss")
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help="Weight decay")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--scheduler_patience", type=int, default=10,
                        help="LR scheduler patience")
    parser.add_argument("--scheduler_factor", type=float, default=0.5,
                        help="LR scheduler factor")

    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Hidden dimension size")
    parser.add_argument("--emb_size", type=int, default=None,
                        help="Final GAT embedding size (defaults to hidden_size)")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of GAT layers")
    parser.add_argument("--num_attn_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate for GAT layers")
    parser.add_argument("--mlp_dropout", type=float, default=0.2,
                        help="Dropout rate for GAT layers")
    parser.add_argument("--pooling_dim", type=int, default=128,
                        help="Hidden dim for pooling MLP")
    parser.add_argument("--mlp_hidden", type=int, default=128,
                        help="Hidden size for final MLP")
    parser.add_argument("--max_nodes", type=int, default=256,
                        help="Cap node count per graph to reduce memory")

    return parser


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'extra-imports': [
            'argparse', 'pandas', 'sklearn.model_selection', 'tqdm',
            'math', 'torch', 'torch.utils.data', 'torch.optim',
            'torch.nn', 'model', 'embed_proteins', 'utils.dataset',
            'utils.helper_functions'
        ],
        'disable': ['C9103', 'R0913', 'R0914', 'E9997', 'E1101', 'E9992'],
        'allowed-io': ['train_model'],
        'max-line-length': 120,
    })

    args = get_parser().parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    train_model(args, device)
