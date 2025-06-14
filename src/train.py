# === train.py (updated) =========================================================
import math
import argparse
import os
import multiprocessing as mp
from functools import partial
import csv
# from dataclasses import dataclass # Not used

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

try:
    # Attempt relative imports first, assuming src is a package
    from .model import DualGraphAttentionNetwork
    from .utils.dataset import DrugProteinDataset, collate_drug_prot, pad_to
    from .utils.unpaired_dataset import DrugOnlyDataset, ProtOnlyDataset
    from .utils.helper_functions import (
        set_seeds,
        count_params,
        # try_run, # Not used in this file
        mse_func,
        mae_func,
        accuracy_func, # Assuming accuracy_func is the CI or similar for this model context
        concordance_index, # More specific CI function
        transform_davis_score # Specific to DAVIS dataset label transformation
    )
except ImportError:
    # Fallback for when src is not treated as a package (e.g., running script directly in src)
    from model import DualGraphAttentionNetwork
    from utils.dataset import DrugProteinDataset, collate_drug_prot, pad_to
    from utils.unpaired_dataset import DrugOnlyDataset, ProtOnlyDataset
    from utils.helper_functions import (
        set_seeds,
        count_params,
        mse_func,
        mae_func,
        accuracy_func,
        concordance_index,
        transform_davis_score
    )

torch.set_float32_matmul_precision("high")


def safe_nan_to_num(t: torch.Tensor) -> torch.Tensor:
    if torch.isnan(t).any() or torch.isinf(t).any():
        return torch.nan_to_num(t, nan=0.0, posinf=1e4, neginf=-1e4)
    return t


def collate_unpaired_drug(batch, max_nodes: int):
    """Collate function for the unpaired drug dataset."""
    # Batch is a list of (d_x, d_z, d_e, d_a) tuples from DrugMolecule.to_tensors()
    H = max_nodes
    drug_zs, drug_xs, drug_es, drug_as = [], [], [], []

    for d_x, d_z, d_e, d_a in batch:
        drug_zs.append(pad_to(safe_nan_to_num(d_z), (H,)))
        drug_xs.append(pad_to(safe_nan_to_num(d_x), (H, d_x.size(-1))))
        drug_es.append(pad_to(safe_nan_to_num(d_e), (H, H, d_e.size(-1))))
        drug_as.append(pad_to(safe_nan_to_num(d_a), (H, H)))

    return (
        torch.stack(drug_zs),
        torch.stack(drug_xs),
        torch.stack(drug_es),
        torch.stack(drug_as)
    )

def collate_unpaired_prot(batch, max_nodes: int):
    """Collate function for the unpaired protein dataset."""
    # Batch is a list of torch_geometric.data.Data objects
    H = max_nodes
    prot_zs, prot_xs, prot_es, prot_as = [], [], [], []
    for protein_graph_data in batch:
        p_x = safe_nan_to_num(protein_graph_data.x)
        p_edge_index = protein_graph_data.edge_index
        p_edge_attr = protein_graph_data.edge_attr
        N_prot_nodes = p_x.size(0)

        # Scale coordinates
        coords = p_x[:, -3:] / 100.0
        p_x[:, -3:] = coords

        if N_prot_nodes > H: # truncate
            node_mask = torch.arange(H)
            p_x = p_x[node_mask]
            edge_mask = (p_edge_index[0] < H) & (p_edge_index[1] < H)
            p_edge_index = p_edge_index[:, edge_mask]
            p_edge_attr = p_edge_attr[edge_mask]

        res_ids = torch.argmax(p_x[:, :20], dim=1).long() + 1
        prot_zs.append(pad_to(res_ids, (H,)))
        prot_node_dense_feats = p_x[:, 20:]
        prot_xs.append(pad_to(prot_node_dense_feats, (H, prot_node_dense_feats.size(1))))

        adj_prot = torch.zeros((H, H), dtype=torch.float32)
        edge_attr_prot_dense = torch.zeros((H, H, p_edge_attr.size(-1)), dtype=torch.float32)

        if p_edge_index.numel() > 0:
            row, col = p_edge_index[0], p_edge_index[1]
            adj_prot[row, col] = 1
            adj_prot[col, row] = 1
            edge_attr_prot_dense[row, col] = safe_nan_to_num(p_edge_attr)
            edge_attr_prot_dense[col, row] = safe_nan_to_num(p_edge_attr)
        
        prot_as.append(adj_prot)
        prot_es.append(edge_attr_prot_dense)
    
    return (
        torch.stack(prot_zs),
        torch.stack(prot_xs),
        torch.stack(prot_es),
        torch.stack(prot_as)
    )

# ---------------------------------------------------------------------------
# TRAIN LOOP
# ---------------------------------------------------------------------------

def train_model(args: argparse.Namespace, m_device: torch.device) -> None:
    """Full training routine for DTI prediction using graph-based protein representation.

    All splits are served through :class:`DrugProteinDataset`, which pulls the
    pre‑built protein *graphs* (``protein_graphs/<ID>.pt``) created by the
    revised **embed_proteins.py** utility.
    """
    # Disable costly anomaly detection hook
    torch.autograd.set_detect_anomaly(False)

    set_seeds(args.seed)
    device = m_device

    # prot_in_features should be 4: (charge(1) + coords(3)) from .pt files,
    # as FeaturePrep handles the 20 AA one-hot encodings.
    # prot_edge_features should be 1 (distance) from .pt files.
    model = DualGraphAttentionNetwork(
        drug_in_features=29, # From DrugMolecule: 17(atom_type) + 1(charge) + 1(degree) + 9(hybridization) + 1(aromatic)
        prot_in_features=4,  # charge (1) + Calpha_coords (3) = 4 dense features for protein after FeaturePrep
        hidden_size=args.hidden_size,
        emb_size=args.emb_size or args.hidden_size, # emb_size for FeaturePrep output and GAT output dim
        drug_edge_features=17, # From DrugMolecule edge processing
        prot_edge_features=1,  # Distance between Calpha atoms
        num_layers=args.num_layers,
        num_heads=args.num_attn_heads,
        dropout=args.dropout,
        mlp_dropout=args.mlp_dropout,
        pooling_dim=args.pooling_dim,
        mlp_hidden=args.mlp_hidden,
        device=device,
        # z_emb_dim, num_atom_types, num_res_types will use defaults in DualGraphAttentionNetwork
        # use_prot_feature_prep=True by default in model, which is what we want now.
        # protein_embedding_dim_passed is no longer needed.
        use_cross=args.use_cross # Pass use_cross from args
    ).to(device)

    # Enable gradient hooks for NaN filtering
    for p in model.parameters():
        p.register_hook(
            lambda grad: torch.nan_to_num(grad, nan=0.0, posinf=1e4, neginf=-1e4)
        )

    print(f"Model parameters: {count_params(model):,}")

    # Setup mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    # ------------------------------------------------------------------
    # dataset + dataloader
    # ------------------------------------------------------------------
    train_ds, val_ds, test_ds = load_data( # test_ds is returned but not used in this train_model func
        data_path=args.data_path,
        seed=args.seed,
        frac_train=args.frac_train,
        frac_val=args.frac_validation,
        frac_test=args.frac_test,
        use_small=args.use_small_dataset,
        dataset_name=args.dataset, # Renamed to avoid conflict with utils.dataset module
        protein_graph_dir=args.protein_graph_dir, # Restored
        max_nodes=args.max_nodes,
        include_3d_drug=getattr(args, 'include_3d_drug', False) # Add if drug 3D coords are used
    )
    
    num_workers = getattr(args, 'num_workers', min(os.cpu_count() or 1, 8))
    # Persistent workers and spawn context can be problematic on some systems, especially Windows.
    # Using default mp context if spawn causes issues, or making it configurable.
    mp_context = "spawn" if os.name != 'nt' else None # "spawn" is good for Linux/MacOS, None for Windows default
    if hasattr(args, 'mp_context'): mp_context = args.mp_context

    base_loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0), # Only if using workers
        multiprocessing_context=mp_context,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    # For collate_drug_prot, prot_edge_feats should match what .pt files provide (1 for distance)
    collate_fn_configured = partial(
        collate_drug_prot,
        hard_limit=args.max_nodes,
        drug_edge_feats=17, # As expected by DrugMolecule and model
        prot_edge_feats=1   # As expected by protein graphs and model
    )

    collate_unpaired_drug_configured = partial(collate_unpaired_drug, max_nodes=args.max_nodes)
    collate_unpaired_prot_configured = partial(collate_unpaired_prot, max_nodes=args.max_nodes)

    # ---- Semi-supervised data loaders (trick #2) ----------------------------
    paired_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_configured,
        **base_loader_kwargs
    )

    unl_drug_loader = DataLoader(
        DrugOnlyDataset(train_ds),
        batch_size=int(args.batch_size * args.unlabeled_ratio),
        shuffle=True,
        collate_fn=collate_unpaired_drug_configured,
        **base_loader_kwargs
    )

    unl_prot_loader = DataLoader(
        ProtOnlyDataset(train_ds),
        batch_size=int(args.batch_size * args.unlabeled_ratio),
        shuffle=True,
        collate_fn=collate_unpaired_prot_configured,
        **base_loader_kwargs
    )
    
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_configured, **base_loader_kwargs)
    # test_loader = DataLoader(test_ds, shuffle=False, collate_fn=collate_fn_configured, **loader_kwargs) # If needed

    # ------------------------------------------------------------------
    # optimisation setup
    # ------------------------------------------------------------------
    optimiser = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = 2_000
    steps_per_epoch = len(paired_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.max_epochs * steps_per_epoch, eta_min=1e-6
    )

    best_val_loss = float("inf")
    no_improvement_epochs = 0

    csv_log_path = f"{args.model_path}.csv"
    if os.path.exists(csv_log_path):
        print(f"Removing existing log file: {csv_log_path}")
        os.remove(csv_log_path)

    log_columns = [
        "epoch", "train_loss", "validation_loss", "train_ci", "validation_ci",
        "train_mse", "validation_mse", "train_mae", "validation_mae", "lr"
    ]
    pd.DataFrame(columns=log_columns).to_csv(csv_log_path, index=False)
    print(f"Logging training progress to: {csv_log_path}")

    # ------------------------------------------------------------------
    # epochs
    # ------------------------------------------------------------------
    from itertools import cycle

    for epoch in tqdm(range(args.max_epochs), desc="Training"):
        model.train()
        train_epoch_loss = 0.0
        train_epoch_preds = []
        train_epoch_labels = []
        train_seen_samples = 0
        global_step = epoch * len(paired_loader)

        # Create cycled iterators for unpaired data
        drug_loader_cycle = cycle(unl_drug_loader)
        prot_loader_cycle = cycle(unl_prot_loader)

        for batch_idx, paired_batch in enumerate(paired_loader):
            global_step = epoch * len(paired_loader) + batch_idx

            # Warm-up learning rate
            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / warmup_steps
                for pg in optimiser.param_groups:
                    pg["lr"] = args.lr * lr_scale

            # Get next batches from cycled loaders
            drug_batch = next(drug_loader_cycle)
            prot_batch = next(prot_loader_cycle)

            # ---------- Paired data ----------
            (d_z, d_x, d_e, d_a, p_z, p_x_dense, p_e, p_a, y) = [t.to(device) for t in paired_batch]

            # ---------- Create MLM masks & masked copies for paired ----------
            def _mask(z):
                m = (torch.rand_like(z.float()) < args.mlm_mask_prob) & (z != 0)
                z_ = z.clone()
                z_[m] = 0
                return z_, m

            d_z_masked, m_d = _mask(d_z)
            p_z_masked, m_p = _mask(p_z)

            optimiser.zero_grad(set_to_none=True)

            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                # Forward pass on paired (regression + paired-MLM)
                y_pred, drug_logits, prot_logits = model(
                    d_z_masked, d_x, d_e, d_a,
                    p_z_masked, p_x_dense, p_e, p_a,
                    mlm_mask_drug=m_d, mlm_mask_prot=m_p
                )

                reg_loss = F.mse_loss(y_pred, y)
                mlm_loss_paired = 0.0
                if drug_logits is not None:
                    mlm_loss_paired += F.cross_entropy(drug_logits, d_z[m_d], ignore_index=0)
                if prot_logits is not None:
                    mlm_loss_paired += F.cross_entropy(prot_logits, p_z[m_p], ignore_index=0)

                # ---------- Unlabeled-drug batch (only drug-MLM) ----------
                (u_d_z, u_d_x, u_d_e, u_d_a) = [t.to(device) for t in drug_batch]
                u_d_z_masked, m_d_un = _mask(u_d_z)
                batch_un_d = u_d_z.size(0)
                # Create dummy protein inputs (all zeros)
                dummy_p_z = torch.zeros((batch_un_d, args.max_nodes), dtype=torch.long, device=device)
                dummy_p_x = torch.zeros((batch_un_d, args.max_nodes, p_x_dense.size(-1)), device=device)
                dummy_p_e = torch.zeros((batch_un_d, args.max_nodes, args.max_nodes, p_e.size(-1)), device=device)
                dummy_p_a = torch.zeros((batch_un_d, args.max_nodes, args.max_nodes), device=device)

                # Disable cross-attention for dummy partner
                orig_flag = model.use_cross
                model.use_cross = False # no cross-attn!
                _, drug_logits_un, _ = model(
                    u_d_z_masked, u_d_x, u_d_e, u_d_a,
                    dummy_p_z, dummy_p_x, dummy_p_e, dummy_p_a,
                    mlm_mask_drug=m_d_un, mlm_mask_prot=None
                )
                model.use_cross = orig_flag

                mlm_loss_un_drug = 0.0
                if drug_logits_un is not None:
                    mlm_loss_un_drug = F.cross_entropy(drug_logits_un, u_d_z[m_d_un], ignore_index=0)

                # ---------- Unlabeled-protein batch (only prot-MLM) ----------
                (u_p_z, u_p_x, u_p_e, u_p_a) = [t.to(device) for t in prot_batch]
                u_p_z_masked, m_p_un = _mask(u_p_z)
                batch_un_p = u_p_z.size(0)
                # Create dummy drug inputs (all zeros)
                dummy_d_z = torch.zeros((batch_un_p, args.max_nodes), dtype=torch.long, device=device)
                dummy_d_x = torch.zeros((batch_un_p, args.max_nodes, d_x.size(-1)), device=device)
                dummy_d_e = torch.zeros((batch_un_p, args.max_nodes, args.max_nodes, d_e.size(-1)), device=device)
                dummy_d_a = torch.zeros((batch_un_p, args.max_nodes, args.max_nodes), device=device)

                # Disable cross-attention for dummy partner
                model.use_cross = False # no cross-attn!
                _, _, prot_logits_un = model(
                    dummy_d_z, dummy_d_x, dummy_d_e, dummy_d_a,
                    u_p_z_masked, u_p_x, u_p_e, u_p_a,
                    mlm_mask_drug=None, mlm_mask_prot=m_p_un
                )
                model.use_cross = orig_flag

                mlm_loss_un_prot = 0.0
                if prot_logits_un is not None:
                    mlm_loss_un_prot = F.cross_entropy(prot_logits_un, u_p_z[m_p_un], ignore_index=0)

                # ---------- Combine paired + unlabeled MLM losses ----------
                total_mlm_loss = mlm_loss_paired + mlm_loss_un_drug + mlm_loss_un_prot
                alpha_now = args.alpha_mlm * (0.1 if epoch > 5 else 1.0)
                loss = reg_loss + alpha_now * total_mlm_loss
            
            if not torch.isfinite(loss):
                print(f"[ERR] non-finite loss at step {global_step}; y_pred stats:",
                      y_pred.min().item(), y_pred.max().item())
                optimiser.zero_grad(set_to_none=True)
                continue

            # Mixed precision backward pass
            scaler.scale(loss).backward()
            
            # Unscale before gradient clipping
            scaler.unscale_(optimiser)

            # --- Grad clipping and sanity check ---
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-10.0, 10.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            bad_grad = False
            for p in model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    bad_grad = True
                    break
            
            if bad_grad:
                optimiser.zero_grad(set_to_none=True)
                print(f"[warn] skipped update at global-step {global_step} because of NaN/Inf gradients")
                continue
            
            # Update with scaler
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            optimiser.zero_grad(set_to_none=True)

            # --- Weight sanity check ---
            bad_w = False
            for p in model.parameters():
                if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                    bad_w = True
                    break
            
            if bad_w:
                print(f"[ERR] param overflow at step {global_step}; rolling back")
                optimiser.zero_grad(set_to_none=True)
                # Naive rollback - could be replaced with state_dict loading
                for p in model.parameters():
                    if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                        p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1e4, neginf=-1e4)
                continue

            batch_size = y.size(0)
            train_epoch_loss += loss.item() * batch_size
            train_epoch_preds.extend(y_pred.detach().cpu().tolist())
            train_epoch_labels.extend(y.detach().cpu().tolist())
            train_seen_samples += batch_size

        if train_seen_samples == 0: 
            print("Warning: No samples processed in training epoch. Check data loader and dataset.")
            # Decide how to handle this - e.g. break, or log and continue
            tl, t_ci, t_mse, t_ma = float('nan'), float('nan'), float('nan'), float('nan')
        else:
            tl = train_epoch_loss / train_seen_samples
            t_ci = concordance_index(torch.tensor(train_epoch_preds), torch.tensor(train_epoch_labels))
            # For MSE/MAE, ensure they are calculated correctly if not done per batch
            # Assuming helper functions mse_func, mae_func work on full epoch tensors
            t_mse = mse_func(torch.tensor(train_epoch_preds), torch.tensor(train_epoch_labels))
            t_ma = mae_func(torch.tensor(train_epoch_preds), torch.tensor(train_epoch_labels))

        # --- validation ------------------------------------------------
        v_loss, v_ci, v_mse, v_mae = get_validation_metrics(val_loader, model, device)

        print(
            f"Epoch {epoch+1}/{args.max_epochs} | LR: {optimiser.param_groups[0]['lr']:.2e} | "
            f"Train L={tl:.4f} CI={t_ci:.4f} MSE={t_mse:.2f} MAE={t_ma:.2f} | "
            f"Val L={v_loss:.4f} CI={v_ci:.4f} MSE={v_mse:.2f} MAE={v_mae:.2f}"
        )

        epoch_log_data = pd.DataFrame([
            [epoch+1, tl, v_loss, t_ci, v_ci, t_mse, v_mse, t_ma, v_mae, optimiser.param_groups[0]['lr']]
        ], columns=log_columns)
        epoch_log_data.to_csv(csv_log_path, mode="a", header=False, index=False)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            no_improvement_epochs = 0
            torch.save(model.state_dict(), args.model_path)
            print(f"  Best validation loss improved to {best_val_loss:.4f}. Model saved to {args.model_path}")
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= args.stoppage_epochs:
                print(f"\nEarly stopping triggered after {args.stoppage_epochs} epochs without improvement.")
                break


# ---------------------------------------------------------------------------
# EVALUATION HELPERS (modified for CI calculation at epoch level)
# ---------------------------------------------------------------------------

def get_validation_metrics(loader, model, device):
    model.eval()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []
    seen_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            try:
                (d_z, d_x, d_e, d_a, p_z, p_x_dense, p_e, p_a, labels) = [item.to(device) for item in batch]
                preds, _, _ = model(d_z, d_x, d_e, d_a, p_z, p_x_dense, p_e, p_a)
                
                if torch.isnan(preds).any(): continue
                loss = F.mse_loss(preds, labels)
                if torch.isnan(loss): continue
                
                batch_size = labels.size(0)
                epoch_loss += loss.item() * batch_size
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                seen_samples += batch_size
            except RuntimeError as e:
                print(f"Runtime error in validation batch: {str(e)}. Skipping.")
                continue

    if seen_samples == 0: return float('inf'), 0.0, float('inf'), float('inf')

    avg_loss = epoch_loss / seen_samples
    # Calculate CI, MSE, MAE using all predictions and labels for the epoch
    preds_tensor = torch.tensor(all_preds)
    labels_tensor = torch.tensor(all_labels)
    ci = concordance_index(preds_tensor, labels_tensor)
    mse = mse_func(preds_tensor, labels_tensor)
    mae = mae_func(preds_tensor, labels_tensor)
    return avg_loss, ci, mse, mae


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

# from tdc.multi_pred import DTI # Moved import inside function where it's used

def load_data(
    data_path: str,
    seed: int,
    frac_train: float,
    frac_val: float,
    frac_test: float,
    use_small: bool,
    dataset_name: str, # Renamed from 'dataset' to avoid module name conflict
    protein_graph_dir: str, # Path to precomputed .pt protein graph files
    max_nodes: int, # Max nodes for padding
    include_3d_drug: bool # Whether to use 3D coordinates for drugs
):
    """Load and split interaction data, then create datasets.

    Handles loading from TDC (BindingDB, DAVIS, KIBA) or a custom CSV file.
    """
    # Debug: Print the received dataset_name and its type
    print(f"DEBUG load_data: Received dataset_name = '{dataset_name}' (type: {type(dataset_name)})")

    from tdc.multi_pred import DTI # Import DTI here

    # Construct dataset CSV path
    main_dataset_csv_path = os.path.join(data_path, f"{dataset_name.upper()}_dataset.csv")
    print(f"Attempting to load main interaction data from: {main_dataset_csv_path}")

    if not os.path.exists(main_dataset_csv_path):
        if dataset_name.upper() in ["DAVIS", "KIBA"]:
            print(f"Cache file {main_dataset_csv_path} not found. Attempting to load {dataset_name.upper()} via TDC...")
            # TDC DTI expects lowercase names for these specific datasets if version matters
            tdc_dti_name = dataset_name.lower() # e.g., 'davis', 'kiba'
            data_source = DTI(name=tdc_dti_name, path=data_path)
            df_interactions = data_source.get_data()
            # Rename columns to match expected format: 'Drug', 'Target_ID', 'Label'
            # Assuming TDC DTI standard column names: 'Drug_ID' (SMILES), 'Target_ID' (Protein ID), 'Y' (Label)
            df_interactions.rename(columns={
                'Drug_ID': 'Drug',       # Assuming Drug_ID contains SMILES
                # 'Target_ID' is often already correct
                'Y': 'Label' 
            }, inplace=True, errors='ignore') # errors='ignore' if some columns might not exist
            if 'Drug' not in df_interactions.columns and 'Drug_ID' in df_interactions.columns:
                 df_interactions.rename(columns={'Drug_ID': 'Drug'}, inplace=True)
            
            # Apply transformations based on dataset type
            if dataset_name.upper() == "DAVIS":
                print("INFO: Applying DAVIS-specific Kd transformation to labels.")
                df_interactions['Label'] = df_interactions['Label'].apply(transform_davis_score)

                # Keep only ONE 'Drug' column ― the SMILES we just picked
                df_interactions = df_interactions.loc[:, ~df_interactions.columns.duplicated(keep='last')]
            
                # Downstream code always expects string SMILES
                df_interactions['Drug'] = df_interactions['Drug'].astype(str)
            
            df_interactions.to_csv(main_dataset_csv_path, index=False)
            print(f"Loaded and cached {dataset_name.upper()} from TDC to {main_dataset_csv_path}")
        elif dataset_name.upper() == "CD4C": # Example for a custom local CSV like filtered_cancer_all.csv
            # This path was for the legacy CD4C csv. Adapt if CD4C also comes from TDC or another source.
            cd4c_csv_name = "filtered_cancer_small.csv" if use_small else "filtered_cancer_all.csv"
            cd4c_path = os.path.join(data_path, cd4c_csv_name)
            if not os.path.exists(cd4c_path):
                raise FileNotFoundError(f"CD4C dataset file not found: {cd4c_path}. Specific handling needed.")
            print(f"Loading CD4C data from: {cd4c_path}")
            df_interactions = pd.read_csv(cd4c_path)
            # Ensure columns are 'Drug', 'Target_ID', 'Label' (e.g. pChEMBL_Value for CD4C)
            if 'smiles' in df_interactions.columns and 'Drug' not in df_interactions.columns:
                 df_interactions.rename(columns={'smiles': 'Drug'}, inplace=True)
            if 'pChEMBL_Value' in df_interactions.columns and 'Label' not in df_interactions.columns:
                 df_interactions.rename(columns={'pChEMBL_Value': 'Label'}, inplace=True)
        else:
            raise FileNotFoundError(f"Dataset file for '{dataset_name}' not found at {main_dataset_csv_path} and no specific loading logic defined.")
    else:
        print(f"Loading main interaction data from cached file: {main_dataset_csv_path}")
        df_interactions = pd.read_csv(main_dataset_csv_path)

    # ------------------------------------------------------------------
    # DAVIS: verify that the column used as SMILES is really SMILES
    # ------------------------------------------------------------------
    def _looks_like_smiles(s: pd.Series) -> bool:
        """Heuristic: at least one entry contains a typical SMILES character."""
        return s.astype(str).str.contains(r"[B-DF-HJ-NP-TV-Zb-df-hj-np-tv-z@\[\]\(\)=#]").any()

    if dataset_name.upper() == "DAVIS":
        # a) prefer an explicit `smiles` column if it exists
        if 'smiles' in df_interactions.columns:
            df_interactions.rename(columns={'smiles': 'Drug'}, inplace=True)

        # b) if `Drug` is present but all-numeric while `Drug_ID` looks like SMILES, swap them
        if ('Drug' in df_interactions.columns
                and pd.api.types.is_numeric_dtype(df_interactions['Drug'])
                and 'Drug_ID' in df_interactions.columns
                and _looks_like_smiles(df_interactions['Drug_ID'])):
            print("[INFO] Cached DAVIS file contained numeric Drug indices – "
                  "using Drug_ID column as the real SMILES.")
            df_interactions['Drug'] = df_interactions['Drug_ID']

        # c) if the column we ended up with still does *not* look like SMILES,
        #    drop the stale cache and rebuild from TDC.
        if not _looks_like_smiles(df_interactions['Drug']):
            print("[WARN] Cached file still lacks valid SMILES. Re-downloading DAVIS …")
            os.remove(main_dataset_csv_path)
            return load_data(   # ─ rerun but this time the cache is gone
                data_path, seed, frac_train, frac_val, frac_test, use_small,
                dataset_name, protein_graph_dir, max_nodes, include_3d_drug)

    # make sure downstream code always sees strings
    df_interactions['Drug'] = df_interactions['Drug'].astype(str)

    # Standardize column names after loading, before validation
    # Handles cases where CSV might have 'Y' or 'pChEMBL_Value' for labels,
    # or 'Drug_ID' / 'smiles' for drug SMILES.
    rename_map = {}
    if 'Y' in df_interactions.columns and 'Label' not in df_interactions.columns:
        rename_map['Y'] = 'Label'
    elif 'pChEMBL_Value' in df_interactions.columns and 'Label' not in df_interactions.columns: # For CD4C
        rename_map['pChEMBL_Value'] = 'Label'

    if 'Drug_ID' in df_interactions.columns and 'Drug' not in df_interactions.columns:
        rename_map['Drug_ID'] = 'Drug'
    elif 'smiles' in df_interactions.columns and 'Drug' not in df_interactions.columns: # For CD4C
        rename_map['smiles'] = 'Drug'
    
    # If 'Drug' is present but 'Drug_ID' is the actual SMILES column and 'Drug' is something else (e.g. drug name)
    # This can happen if TDC data was saved with both. We prefer 'Drug' to be SMILES.
    # If 'Drug_ID' exists and contains SMILES, and 'Drug' also exists but might not be SMILES,
    # ensure 'Drug' becomes the SMILES column.
    # The current error shows both 'Drug_ID' and 'Drug'. We need to ensure 'Drug' is the one used for SMILES.
    # If 'Drug_ID' is the SMILES and 'Drug' is something else, we should use 'Drug_ID' as 'Drug'.
    # If 'Drug' is already SMILES, this is fine.
    # The error `Found: Index(['Drug_ID', 'Drug', 'Target_ID', 'Target', 'Y']` suggests 'Drug' might already be SMILES.
    # The initial TDC rename `{'Drug_ID': 'Drug'}` might be sufficient if 'Drug' wasn't originally present or was to be overwritten.

    # Let's simplify: ensure 'Y' becomes 'Label'. If 'Drug' is not present, try 'Drug_ID' then 'smiles'.
    # The error indicates 'Drug' IS present. So the main issue is likely 'Y' vs 'Label'.

    if rename_map:
        df_interactions.rename(columns=rename_map, inplace=True)
        print(f"Renamed columns: {rename_map}")


    # Ensure essential columns exist
    required_cols = ['Drug', 'Target_ID', 'Label']
    if not all(col in df_interactions.columns for col in required_cols):
        raise ValueError(f"Interaction data CSV must contain columns: {required_cols}. Found: {df_interactions.columns}")

    if use_small:
        df_interactions = df_interactions.sample(n=min(1000, len(df_interactions)), random_state=seed)
        print(f"Using small dataset with {len(df_interactions)} samples.")

    # Split data - ensure stratification if dataset is imbalanced or regression values are skewed.
    # Using simple random split for now, can be enhanced with stratified split on 'Label' if needed.
    train_df, temp_df = train_test_split(df_interactions, train_size=frac_train, random_state=seed)
    # Adjust split for validation and test to sum correctly with frac_train
    test_frac_of_remainder = frac_test / (frac_val + frac_test) if (frac_val + frac_test) > 0 else 0.5
    val_df, test_df = train_test_split(temp_df, test_size=test_frac_of_remainder, random_state=seed)

    print(f"Dataset split: Train {len(train_df)}, Validation {len(val_df)}, Test {len(test_df)}")

    common_dataset_args = {
        'graph_dir': protein_graph_dir,
        'max_nodes': max_nodes,
        'include_3d_drug': include_3d_drug
    }

    train_dataset = DrugProteinDataset(df=train_df, **common_dataset_args)
    val_dataset = DrugProteinDataset(df=val_df, **common_dataset_args)
    test_dataset = DrugProteinDataset(df=test_df, **common_dataset_args)

    return train_dataset, val_dataset, test_dataset


# ---------------------------------------------------------------------------
# CLI ARGUMENTS
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DualGraphAttentionNetwork for DTI with graph protein features.")
    # Basic settings
    parser.add_argument("--dataset", type=str, default="DAVIS", choices=["CD4C", "DAVIS", "KIBA"], help="Dataset to use.")
    parser.add_argument("--data_path", type=str, default="../data", help="Path to data directory (for dataset CSVs). CS.")
    parser.add_argument("--protein_graph_dir", type=str, default="../data/protein_graphs", help="Directory for precomputed .pt protein graph files.")
    parser.add_argument("--model_path", type=str, default='../models/gat_cd4c_model.pth', help="Path to save/load the trained model.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"], help="Device to use.")

    # Training loop
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--max_epochs", type=int, default=512, help="Max training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=2e-4, help="Weight decay.")
    parser.add_argument("--stoppage_epochs", type=int, default=32, help="Early stopping patience.")
    parser.add_argument("--scheduler_patience", type=int, default=10, help="LR scheduler patience.")
    parser.add_argument("--scheduler_factor", type=float, default=0.5, help="LR scheduler reduction factor.")
    parser.add_argument("--huber_beta", type=float, default=0.5, help="Beta for Huber loss.")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm value.")
    
    # Data handling
    parser.add_argument("--use_small_dataset", action="store_true", help="Use a small subset of data.")
    parser.add_argument("--frac_train", type=float, default=0.8, help="Fraction for training.")
    parser.add_argument("--frac_validation", type=float, default=0.1, help="Fraction for validation.")
    parser.add_argument("--frac_test", type=float, default=0.1, help="Fraction for testing.")
    parser.add_argument("--max_nodes", type=int, default=72, help="Max nodes for padding drug/protein graphs.")
    parser.add_argument("--num_workers", type=int, default=None, help="Num workers for DataLoader. Defaults to min(CPU_count, 8). Set 0 for no multiprocessing.")
    parser.add_argument("--mp_context", type=str, default=None, help="Multiprocessing context (e.g., 'spawn', 'fork'). Default chosen based on OS.")

    # Model architecture (GAT-CD4C specific)
    # drug_in_features=29, prot_in_features=4 (dense part), prot_edge_features=1 are somewhat fixed by data processing
    parser.add_argument("--hidden_size", type=int, default=192, help="Hidden size in GAT layers.")
    parser.add_argument("--emb_size", type=int, default=96, help="Embedding size (FeaturePrep output, GAT output).")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of GAT layers.")
    parser.add_argument("--num_attn_heads", type=int, default=6, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout in GAT attention/FFN.")
    parser.add_argument("--mlp_dropout", type=float, default=0.2, help="Dropout in readout MLP.")
    parser.add_argument("--pooling_dim", type=int, default=96, help="Hidden dim for GlobalAttentionPooling projection.")
    parser.add_argument("--mlp_hidden", type=int, default=192, help="Hidden dim in readout MLP.")
    parser.add_argument("--use_cross", action="store_true", help="Enable cross-attention between drug/protein branches.")
    parser.add_argument("--no_cross", action="store_false", dest="use_cross", help="Disable cross-attention.")
    parser.set_defaults(use_cross=True) # Default to using cross-attention
    parser.add_argument('--alpha_mlm', type=float, default=0.1,
                        help='Weight of MLM loss.')
    parser.add_argument('--unlabeled_ratio', type=float, default=0.5,
                        help='Fraction of each mini‑batch made of unlabeled graphs.')
    parser.add_argument('--mlm_mask_prob', type=float, default=0.15,
                        help='Probability of masking a node for MLM.')
    # parser.add_argument("--include_3d_drug", action="store_true", help="Include 3D coordinates for drug molecules if featurizer supports it.") # This is implicitly handled by DrugMolecule

    return parser


if __name__ == "__main__":
    cli_parser = get_parser()
    cli_args = cli_parser.parse_args()

    if cli_args.num_workers is None:
        cli_args.num_workers = min(os.cpu_count() or 1, 8)
        print(f"Defaulting num_workers to: {cli_args.num_workers}")

    selected_device = None
    if cli_args.device == "cuda" and torch.cuda.is_available():
        selected_device = torch.device("cuda")
    elif cli_args.device == "mps" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("MPS device requested and available.")
        selected_device = torch.device("mps")
    else:
        if cli_args.device != "cpu":
            print(f"Device '{cli_args.device}' not available. Falling back to CPU.")
        selected_device = torch.device("cpu")
        
    print(f"Using device: {selected_device}")
    train_model(cli_args, selected_device)
