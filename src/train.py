import math
import argparse
import pandas as pd
import contextlib
import os
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
from utils.dataset import DrugProteinDataset, collate_drug_prot
from utils.helper_functions import set_seeds, count_model_params, plot_loss_curves, accuracy_func, mse_func, mae_func

torch.set_float32_matmul_precision('high')


def train_model(args: argparse.Namespace, m_device: torch.device) -> None:
    set_seeds()
    device = m_device

    model = DualGraphAttentionNetwork(
        drug_in_features=29,
        prot_in_features=1283,
        hidden_size=args.hidden_size,
        emb_size=args.emb_size,
        drug_edge_features=17,
        prot_edge_features=1,
        num_layers=args.num_layers,
        num_heads=args.num_attn_heads,
        dropout=args.dropout,
        mlp_dropout=args.mlp_dropout,
        pooling_dim=args.pooling_dim,
        mlp_hidden=args.mlp_hidden,
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
        num_workers=8,  # increase if CPU cores available
        pin_memory=True,  # beneficial for CUDA
        persistent_workers=True,
        prefetch_factor=4, # prefetch more batches
        multiprocessing_context=ctx,
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        collate_fn=partial(
            collate_drug_prot,
            prot_graph_dir=args.protein_graph_dir,
            hard_limit=args.max_nodes,
            drug_edge_feats=17,
            prot_edge_feats=1
        ),
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        collate_fn=partial(
            collate_drug_prot,
            prot_graph_dir=args.protein_graph_dir,
            hard_limit=args.max_nodes,
            drug_edge_feats=17,
            prot_edge_feats=1
        ),
        **loader_kwargs
    )

    loss_func = nn.SmoothL1Loss(beta=args.huber_beta)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.scheduler_factor, patience=args.scheduler_patience
    )

    best_val = float('inf')
    no_imp = 0

    columns = [
        'train_loss',     'validation_loss',
        'train_acc',      'validation_acc',
        'train_mse',      'validation_mse',
        'train_mae',      'validation_mae'
    ]
    
    metrics = pd.DataFrame(
        columns=columns,
        index=range(args.max_epochs)
    )

    # Clear metrics csv
    csv_path = f"{args.model_path}.csv"
    
    # 1) Remove any old file
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # Write correct headers
    pd.DataFrame(columns=columns).to_csv(csv_path, index=False)
    
    for epoch in range(args.max_epochs):
        model.train()
        total = dict(loss=0, acc=0, mse=0, mae=0)
        samples = 0
    
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.max_epochs}"):
            d_z, d_x, d_e, d_a, p_z, p_x, p_e, p_a, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
    
            # forward
            preds = model(d_z, d_x, d_e, d_a, p_z, p_x, p_e, p_a).squeeze(-1)
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
            torch.save(model.state_dict(), args.model_path)
        else:
            no_imp += 1
            if no_imp >= args.stoppage_epochs:
                break
    
        row_df = pd.DataFrame([row])  
        # append without header, without index
        row_df.to_csv(csv_path, mode='a', header=False, index=False)


def get_validation_metrics(loader, model, loss_func, device):
    model.eval()
    total_samples = 0
    total_loss = 0
    total_acc = 0
    total_mse = 0
    total_mae = 0

    with torch.no_grad():
        for batch in loader:
            d_z, d_x, d_e, d_a, p_z, p_x, p_e, p_a, labels = [x.to(device, non_blocking=True) for x in batch]
            preds = model(d_z, d_x, d_e, d_a, p_z, p_x, p_e, p_a).squeeze(-1)
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
