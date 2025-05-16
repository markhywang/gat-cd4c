import math
import argparse
import pandas as pd
import contextlib
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from functools import partial
import torch.nn.functional as F
import multiprocessing as mp

from model import DualGraphAttentionNetwork
from utils.dataset import DrugProteinDataset, collate_drug_prot
from utils.helper_functions import set_seeds, count_model_params, accuracy_func, mse_func, mae_func

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

    print(f'Model parameters: {count_model_params(model)}')

    train_ds, val_ds, _ = load_data(
        args.data_path, args.seed,
        args.frac_train, args.frac_validation, args.frac_test,
        args.use_small_dataset, args.protein_graph_dir, args.dataset
    )

    ctx = mp.get_context('spawn')

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
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
    
    if os.path.exists(csv_path):
        os.remove(csv_path)

    pd.DataFrame(columns=columns).to_csv(csv_path, index=False)
    
    for epoch in range(args.max_epochs):
        model.train()
        total = dict(loss=0, acc=0, mse=0, mae=0)
        samples = 0
    
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.max_epochs}"):
            d_z, d_x, d_e, d_a, p_z, p_x, p_e, p_a, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
    
            preds = model(d_z, d_x, d_e, d_a, p_z, p_x, p_e, p_a).squeeze(-1)
            loss = loss_func(preds, labels)
    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
            n = labels.size(0)
            samples += n
            total['loss'] += loss.item() * n
            total['acc']  += accuracy_func(preds, labels, 1.0)
            total['mse']  += mse_func(preds, labels) * n
            total['mae']  += mae_func(preds, labels) * n
    
        tl = total['loss'] / samples
        ta = total['acc'] / samples
        tm = total['mse'] / samples
        tma = total['mae'] / samples

        v_loss, v_acc, v_mse, v_mae = get_validation_metrics(val_loader, model, loss_func, device)

        lr_scheduler.step(v_loss)

        row = [
            tl, v_loss,
            ta,  v_acc,
            tm,  v_mse,
            tma, v_mae,
        ]
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
    
        pd.DataFrame([row]).to_csv(csv_path, mode='a', header=False, index=False)


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


def load_data(data_path, seed, frac_train, frac_val, frac_test, use_small, protein_graph_dir, dataset):
    """Load train/val/test splits for either the original CD4C, KIBA, or DAVIS dataset.

    Ensures that **each split only contains the three canonical columns** expected by
    ``DrugProteinDataset`` (``smiles``, ``Target_ID``, ``pChEMBL_Value``) so that
    each slice is a *Series* and therefore has the ``.tolist`` method.
    """
    assert math.isclose(frac_train + frac_val + frac_
