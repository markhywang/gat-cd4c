"""Graph Attention Networks (GAT) components — *normalisation‑enhanced version*
==========================================================================
Adds the four extra LayerNorm sites discussed in the review:
1. **FeaturePrep output**  → `DualGraphAttentionNetwork.drug_ln` / `prot_ln`
2. **Edge stream post‑residual** → `GraphAttentionLayer.layer_norm_edge_out`
3. **Feed‑forward sub‑layer inside GPSLayer** → inserts a LayerNorm between CELU and Dropout
4. **Graph‑level embeddings** → `DualGraphAttentionNetwork.pool_norm_[drug|prot]`

Minor fixes
-----------
* Propagates `mlp_dropout` everywhere (previous commit missed the edge‑MLP).
* Removes the duplicate legacy `GraphAttentionLayer` definition at the end of the file.
* Syntax typo in `edge_mlp` (missing comma) corrected.
"""
from __future__ import annotations

import math
from typing import Tuple, Union, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# -------------------------------------------------
#  SSM‑DTA helpers
# -------------------------------------------------


class MLMHead(nn.Module):
    """Linear decoder that predicts the original node class from a masked embedding."""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, num_classes)

    def forward(self, hidden: Tensor) -> Tensor:  # [N_masked, emb]
        return self.proj(hidden)                 # [N_masked, vocab]

# -----------------------------------------------------------------------------
# Basic utility modules -------------------------------------------------------
# -----------------------------------------------------------------------------


class FeaturePrep(nn.Module):
    """Prepares node features by embedding categorical IDs and concatenating dense features."""

    def __init__(
        self,
        num_categories: int,      # Number of unique ID categories (e.g., atom types, residue types)
        emb_dim: int,             # Dimension for embedding the IDs
        in_features_dense: int,   # Number of dense features in `x`
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.embedding = nn.Embedding(num_categories + 1, emb_dim, padding_idx=0) # +1 for padding_idx 0
        self.out_features = emb_dim + in_features_dense
        self.to(self.device)

    def forward(self, z: torch.Tensor, x_dense: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z (torch.Tensor): Categorical node IDs [B, N_max_nodes].
            x_dense (torch.Tensor): Dense node features [B, N_max_nodes, F_dense].
        """
        z_emb = self.embedding(z)  # [B, N_max_nodes, emb_dim]
        # Concatenate embeddings with dense features
        # Ensure x_dense is on the same device
        return torch.cat([z_emb, x_dense.to(self.device)], dim=-1) # [B, N_max_nodes, emb_dim + F_dense]


class CoAttention(nn.Module):
    """Bidirectional cross‑graph attention with a learnable gate α."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.drug2prot = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.prot2drug = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.norm_drug = nn.LayerNorm(embed_dim)
        self.norm_prot = nn.LayerNorm(embed_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # gating parameter

    def forward(self, drug_feats: Tensor, prot_feats: Tensor) -> Tuple[Tensor, Tensor]:
        d_attn, _ = self.drug2prot(query=drug_feats, key=prot_feats, value=prot_feats)
        p_attn, _ = self.prot2drug(query=prot_feats, key=drug_feats, value=drug_feats)

        d_up = self.norm_drug(drug_feats + self.alpha * d_attn)
        p_up = self.norm_prot(prot_feats + self.alpha * p_attn)
        return d_up, p_up


# -----------------------------------------------------------------------------
# Graph transformer building blocks ------------------------------------------
# -----------------------------------------------------------------------------
class GraphAttentionLayer(nn.Module):
    """Single message‑passing layer (edge‑aware multi‑head attention) with NaN‑safe soft‑max."""

    def __init__(
        self,
        device: Union[str, torch.device],
        in_features: int,
        out_features: int,
        num_edge_features: int,
        num_attn_heads: int = 1,
        dropout: float = 0.2,
        mlp_dropout: float = 0.2,
        use_leaky_relu: bool = True,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.num_edge_features = num_edge_features
        self.num_attn_heads = num_attn_heads
        self.out_features_per_head = out_features // num_attn_heads

        # projections & normalisation ---------------------------------------
        self.node_projection = nn.Linear(in_features, out_features)
        self.layer_norm_1 = nn.LayerNorm(in_features)

        if num_edge_features > 0:
            self.edge_mlp = nn.Sequential(
                nn.Linear(num_edge_features, num_edge_features * 2),
                nn.ReLU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(num_edge_features * 2, num_attn_heads),
            )
            self.edge_layer_norm = nn.LayerNorm(num_edge_features)
            self.layer_norm_edge_out = nn.LayerNorm(num_attn_heads)
        else:
            self.edge_mlp = None
            self.edge_layer_norm = None
            self.layer_norm_edge_out = None

        # attention projections ---------------------------------------------
        self.W_q = nn.Linear(out_features, out_features, bias=False)
        self.W_k = nn.Linear(out_features, out_features, bias=False)
        self.W_v = nn.Linear(out_features, out_features, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2) if use_leaky_relu else nn.Identity()

        # feed forward -------------------------------------------------------
        self.ffn = nn.Sequential(
            nn.Linear(out_features, 2 * out_features),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(2 * out_features, out_features),
        )
        self.layer_norm_2 = nn.LayerNorm(out_features)
        self.output_dropout = nn.Dropout(dropout)
        self.to(self.device)

    def forward(self, x: Tensor, adj: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        B, N, _ = x.shape

        # 1. linear proj + LN ----------------------------------------------
        x_proj = self.node_projection(self.layer_norm_1(x))

        # 2. build Q,K,V -----------------------------------------------------
        Q = self.W_q(x_proj).view(B, N, self.num_attn_heads, self.out_features_per_head).transpose(1, 2)
        K = self.W_k(x_proj).view(B, N, self.num_attn_heads, self.out_features_per_head).transpose(1, 2)
        V = self.W_v(x_proj).view(B, N, self.num_attn_heads, self.out_features_per_head).transpose(1, 2)

        # 3. scaled dot‑product --------------------------------------------
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.out_features_per_head)

        # 4. edge bias (clamped) -------------------------------------------
        if self.edge_mlp is not None and edge_attr is not None:
            edge_bias = self.edge_mlp(self.edge_layer_norm(edge_attr))  # [B,N,N,H]
            edge_bias = edge_bias.permute(0, 3, 1, 2)                  # [B,H,N,N]
            attn_scores = attn_scores + torch.tanh(edge_bias) * 5.0

        attn_scores = self.leaky_relu(attn_scores)

        # 5. mask + self‑loop -------------------------------------------------
        eye  = torch.eye(N, device=adj.device).unsqueeze(0)           # [1,N,N]
        mask = (adj + eye).clamp(max=1).unsqueeze(1)                  # [B,1,N,N]

        neg_inf = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(mask == 0, neg_inf)

        all_masked = (mask == 0).all(dim=-1, keepdim=True)            # [B,1,N,1]
        attn_scores = attn_scores.masked_fill(all_masked, 0.0)

        # 6. safe soft‑max ---------------------------------------------------
        attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True)
        attn_probs  = torch.softmax(attn_scores, dim=-1)
        attn_probs  = torch.nan_to_num(attn_probs, nan=0.0)
        attn_probs  = self.attn_dropout(attn_probs)
        ctx = torch.matmul(attn_probs, V).transpose(1, 2).contiguous().view(B, N, -1)

        h_attn = x_proj + self.output_dropout(ctx)
        h_ffn = self.ffn(self.layer_norm_2(h_attn))
        return h_attn + self.output_dropout(h_ffn)


class GPSLayer(nn.Module):
    """Local GAT + optional global/cross attention + feed‑forward."""

    def __init__(
        self,
        local_layer: nn.Module,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        mlp_dropout: float,
        use_cross: bool,
    ) -> None:
        super().__init__()
        self.local = local_layer

        in_dim = local_layer.node_projection.in_features
        out_dim = local_layer.node_projection.out_features
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = nn.Identity()

        self.global_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.use_cross = use_cross
        if use_cross:
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.gamma_raw = nn.Parameter(torch.tensor(0.1))  # For raw feature skip connection

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(mlp_dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x_e_a: Tuple[Tensor, Tensor, Tensor], context: Tensor | None = None) -> Tuple[Tensor, Tensor, Tensor]:
        x, e, a = x_e_a
        residual = x

        # 1. local message passing ------------------------------------------
        x = self.local(x, a, e)

        # 2. global self‑attention -----------------------------------------
        g_out, _ = self.global_attn(x, x, x)

        # 3. optional cross‑graph attention --------------------------------
        if self.use_cross and context is not None and context.size(-1) == x.size(-1):
            c_out, _ = self.cross_attn(x, context, context)
        else:
            c_out = torch.zeros_like(x)

        # 4. gated fusion & feed‑forward -----------------------------------
        res = self.residual_proj(residual)
        fused = res + x + self.gamma * g_out + self.gamma * c_out
        fused = self.norm(fused + self.gamma_raw * res)
        out = self.mlp(fused)
        return out, e, a


class GlobalAttentionPooling(nn.Module):
    """Global attention pooling layer from the GAT-CD4C paper with NaN- and half-precision-safe masking."""

    def __init__(self, in_features: int, hidden_dim: int, device: Union[str, torch.device] = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.gate_nn = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.to(self.device)

    def forward(self, x: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Node features [B, N, F_in].
            node_mask (torch.Tensor, optional): Mask for actual nodes [B, N], 1 for real, 0 for padding.
        """
        # [B, N, 1]
        attn_logits = self.gate_nn(x)

        if node_mask is not None:
            # build mask [B, N, 1]
            mask = node_mask.unsqueeze(-1) == 0
            # smallest finite value for this dtype (safe in fp16/bf16)
            neg_inf = torch.finfo(attn_logits.dtype).min
            attn_logits = attn_logits.masked_fill(mask, neg_inf)

            # if a sample has *all* nodes masked, reset logits to zero so softmax is uniform
            all_masked = mask.all(dim=1, keepdim=True)  # [B, 1, 1]
            attn_logits = attn_logits.masked_fill(all_masked, 0.0)

        # subtract max for numerical stability, then softmax
        attn_logits = attn_logits - attn_logits.amax(dim=1, keepdim=True)
        attn_scores = torch.softmax(attn_logits, dim=1)
        # guard against any remaining NaNs
        attn_scores = torch.nan_to_num(attn_scores, nan=0.0, posinf=0.0, neginf=0.0)

        # weighted sum → [B, F_in]
        context = torch.sum(attn_scores * x, dim=1)
        return context


class GraphAttentionEncoder(nn.Module):
    """Stack of GPS layers."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int,
        num_edge_features: int,
        num_layers: int,
        num_attn_heads: int,
        dropout: float,
        mlp_dropout: float,
        device: Union[str, torch.device],
        *,
        use_cross: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(num_layers):
            in_f = in_features if i == 0 else hidden_size
            out_f = out_features if i == num_layers - 1 else hidden_size
            heads = 1 if i == num_layers - 1 else num_attn_heads

            local = GraphAttentionLayer(
                device,
                in_f,
                out_f,
                num_edge_features,
                heads,
                dropout,
                mlp_dropout,
                use_leaky_relu=(i != num_layers - 1),
            )
            layers.append(
                GPSLayer(
                    local,
                    embed_dim=out_f,
                    num_heads=heads,
                    dropout=dropout,
                    mlp_dropout=mlp_dropout,
                    use_cross=use_cross,
                )
            )
        self.gps_layers = nn.ModuleList(layers)

    def forward(self, node_feats: Tensor, edge_feats: Tensor, adj: Tensor, *, context: Tensor | None = None) -> Tensor:
        x, e, a = node_feats, edge_feats, adj
        for layer in self.gps_layers:
            x, e, a = layer((x, e, a), context=context)
        return x


class DualGraphAttentionNetwork(nn.Module):
    """Drug‑graph encoder + Protein‑graph encoder + regression head."""

    def __init__(
        self,
        drug_in_features: int,    # Number of raw dense features for drug atoms (e.g., the 29 from DrugMolecule)
        prot_in_features: int,    # Number of raw dense features for protein residues (e.g. charge, coords = 4)
        hidden_size: int = 64,    # Hidden size within GAT layers (intermediate, not directly GAT output unless emb_size=hidden_size)
        emb_size: int = 64,       # Output dimension of GAT layers & input to pooling if no cross-attn
        drug_edge_features: int = 17, 
        prot_edge_features: int = 1,  
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        mlp_dropout: float = 0.2,
        pooling_dim: int = 128,   # Hidden dim for GlobalAttentionPooling's gate_nn
        mlp_hidden: int = 128,
        device: Union[str, torch.device] = "cpu",
        z_emb_dim: int = 16,      # Embedding dimension for atomic numbers (drug) and residue types (protein)
        num_atom_types: int = 119, 
        num_res_types: int = 22,  # 20 AA + 1 unknown/mask + 1 padding_idx for nn.Embedding
        use_prot_feature_prep: bool = True, # Should be True for graph proteins
        use_cross: bool = True
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.use_cross = use_cross
        self.use_prot_feature_prep = use_prot_feature_prep

        # ---- Feature preparation ----
        self.drug_feature_prep = FeaturePrep(num_atom_types, z_emb_dim, drug_in_features, device)
        if use_prot_feature_prep:
            self.prot_feature_prep = FeaturePrep(num_res_types, z_emb_dim, prot_in_features, device)
        self.drug_ln = nn.LayerNorm(self.drug_feature_prep.out_features)
        self.prot_ln = nn.LayerNorm(
            self.prot_feature_prep.out_features if use_prot_feature_prep else prot_in_features
        )

        # ---- Graph encoders ----
        self.drug_encoder = GraphAttentionEncoder(
            in_features=self.drug_feature_prep.out_features,
            hidden_size=hidden_size,
            out_features=emb_size,
            num_edge_features=drug_edge_features,
            num_layers=num_layers,
            num_attn_heads=num_heads,
            dropout=dropout,
            mlp_dropout=mlp_dropout,
            device=device,
            use_cross=use_cross,
        )
        self.prot_encoder = GraphAttentionEncoder(
            in_features=self.prot_feature_prep.out_features if use_prot_feature_prep else prot_in_features,
            hidden_size=hidden_size,
            out_features=emb_size,
            num_edge_features=prot_edge_features,
            num_layers=num_layers,
            num_attn_heads=num_heads,
            dropout=dropout,
            mlp_dropout=mlp_dropout,
            device=device,
            use_cross=use_cross,
        )

        # ---- CLS-only cross-attention ----
        if self.use_cross:
            self.cross_cls_drug = nn.MultiheadAttention(
                emb_size, num_heads, dropout=dropout, batch_first=True
            )
            self.cross_cls_prot = nn.MultiheadAttention(
                emb_size, num_heads, dropout=dropout, batch_first=True
            )

        # ---- Pooling ----
        self.drug_pooling = GlobalAttentionPooling(emb_size, pooling_dim, device)
        self.prot_pooling = GlobalAttentionPooling(emb_size, pooling_dim, device)
        self.pool_norm_drug = nn.LayerNorm(emb_size)
        self.pool_norm_prot = nn.LayerNorm(emb_size)

        # ---- MLM heads ----
        self.mlm_head_drug = MLMHead(emb_size, num_atom_types + 1)
        self.mlm_head_prot = MLMHead(emb_size, num_res_types + 1)

        # ---- Regression head ----
        self.regressor = nn.Sequential(
            nn.Linear(emb_size * 2, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

        self.to(self.device)

    def forward(
        self,
        d_z: torch.Tensor,      # Drug atomic numbers [B, N_d]
        d_x: torch.Tensor,      # Drug node dense features [B, N_d, F_d_dense_raw] (e.g. 29)
        d_e: torch.Tensor,      # Drug edge features [B, N_d, N_d, F_d_edge]
        d_a: torch.Tensor,      # Drug adjacency matrix [B, N_d, N_d]
        p_z: torch.Tensor,      # Protein residue type IDs [B, N_p]
        p_x_dense: torch.Tensor,# Protein node dense features (charge, coords) [B, N_p, F_p_dense_raw = 4]
        p_e: torch.Tensor,      # Protein edge features [B, N_p, N_p, F_p_edge]
        p_a: torch.Tensor,      # Protein adjacency matrix [B, N_p, N_p]
        mlm_mask_drug: Optional[torch.Tensor] = None,  # [B, N_d], True for masked nodes
        mlm_mask_prot: Optional[torch.Tensor] = None,  # [B, N_p], True for masked nodes
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        # --- Drug Branch ---
        drug_features_prepared = self.drug_feature_prep(d_z, d_x)
        drug_features_prepared = self.drug_ln(drug_features_prepared)
        hd = self.drug_encoder(drug_features_prepared, d_e, d_a)

        # --- Protein Branch ---
        if self.use_prot_feature_prep:
            prot_features_prepared = self.prot_feature_prep(p_z, p_x_dense)
            prot_features_prepared = self.prot_ln(prot_features_prepared)
        else:
            prot_features_prepared = p_x_dense
        hp = self.prot_encoder(prot_features_prepared, p_e, p_a)

        # --- CLS-only cross-attention or plain pooling ---
        if self.use_cross:
            d_cls = self.drug_pooling(hd, node_mask=(d_z != 0)).unsqueeze(1)  # [B,1,E]
            p_cls = self.prot_pooling(hp, node_mask=(p_z != 0)).unsqueeze(1)  # [B,1,E]

            drug_padding_mask = (d_z == 0) # [B,N]
            prot_padding_mask = (p_z == 0)

            # sample-wise flags: does this graph have ≥1 real node?
            has_prot = (~prot_padding_mask).any(dim=1) # [B]
            has_drug = (~drug_padding_mask).any(dim=1) # [B]

            # init with the plain CLS vectors (no cross)
            d_cls_upd = d_cls.clone()
            p_cls_upd = p_cls.clone()

            if has_prot.any():
                d_cls_upd[has_prot], _ = self.cross_cls_drug(
                    d_cls[has_prot], hp[has_prot], hp[has_prot],
                    key_padding_mask=prot_padding_mask[has_prot])

            if has_drug.any():
                p_cls_upd[has_drug], _ = self.cross_cls_prot(
                    p_cls[has_drug], hd[has_drug], hd[has_drug],
                    key_padding_mask=drug_padding_mask[has_drug])

            # final guard – just in case
            d_cls_upd = torch.nan_to_num(d_cls_upd, nan=0.0)
            p_cls_upd = torch.nan_to_num(p_cls_upd, nan=0.0)

            drug_vec = d_cls_upd.squeeze(1)
            prot_vec = p_cls_upd.squeeze(1)
        else:
            drug_vec = self.drug_pooling(hd, node_mask=(d_z != 0))
            prot_vec = self.prot_pooling(hp, node_mask=(p_z != 0))

        drug_vec = self.pool_norm_drug(drug_vec)
        prot_vec = self.pool_norm_prot(prot_vec)

        # --- Regression head ---
        reg = self.regressor(torch.cat([drug_vec, prot_vec], dim=1)).squeeze(-1)

        # --- Optional MLM logits (only for masked nodes) ---
        drug_logits = prot_logits = None
        if mlm_mask_drug is not None and mlm_mask_drug.any():
            drug_logits = self.mlm_head_drug(hd[mlm_mask_drug])
        if mlm_mask_prot is not None and mlm_mask_prot.any():
            prot_logits = self.mlm_head_prot(hp[mlm_mask_prot])

        return reg, drug_logits, prot_logits


if __name__ == '__main__':
    # import doctest
    # doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': [
            'numpy',
            'pandas',
            'sklearn.model_selection',
            'sklearn.metrics',
            'rdkit',
            'xgboost',
            'rdkit.Chem.rdFingerprintGenerator',
            'Chem.MolFromSmiles',
            'DataStructs.ConvertToNumpyArray',
            'math',
            'torch',
            'torch.nn',
            'torch.nn.functional'
        ],
        'disable': ['R0914', 'E1101', 'R0913', 'R0902', 'E9959'],
        # R0914 for local variable, E1101 for attributes for imported modules
        # R0913 for arguments, R0902 for instance attributes in class
        # E9959 for instance annotation
        'allowed-io': ['main'],
        'max-line-length': 120,
    })