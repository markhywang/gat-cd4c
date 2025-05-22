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
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# -----------------------------------------------------------------------------
# Basic utility modules -------------------------------------------------------
# -----------------------------------------------------------------------------


class FeaturePrep(nn.Module):
    """Prepend a learned categorical embedding in front of dense node features."""

    def __init__(self, num_types: int, emb_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_types, emb_dim)

    def forward(self, ids: Tensor, feats: Tensor) -> Tensor:  # ids [B,N], feats [B,N,F]
        emb = self.embed(ids)  # [B, N, emb_dim]
        return torch.cat([emb, feats], dim=-1)


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
    """Single message‑passing layer (edge‑aware multi‑head attention)."""

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

        # node & edge preprocessing --------------------------------------------
        self.node_projection = nn.Linear(in_features, out_features)
        self.layer_norm_1 = nn.LayerNorm(in_features)

        self.edge_mlp = nn.Sequential(
            nn.Linear(num_edge_features, 2 * num_edge_features),
            nn.CELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(2 * num_edge_features, num_edge_features),
        )
        self.layer_norm_2 = nn.LayerNorm(num_edge_features)
        # **NEW**
        self.layer_norm_edge_out = nn.LayerNorm(num_edge_features)

        # attention machinery ---------------------------------------------------
        self.num_attn_heads = num_attn_heads
        self.head_size = out_features // num_attn_heads
        self.attn_matrix = nn.Parameter(
            torch.empty(num_attn_heads, 2 * self.head_size + num_edge_features)
        )
        self.attn_leaky_relu = nn.LeakyReLU(0.2)

        # output & regularisation ----------------------------------------------
        self.layer_norm_out = nn.LayerNorm(out_features)
        self.out_node_projection = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)

        self.use_leaky_relu = use_leaky_relu
        if use_leaky_relu:
            self.leaky_relu = nn.LeakyReLU(0.2)

        # Xavier init -----------------------------------------------------------
        nn.init.xavier_uniform_(self.node_projection.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.attn_matrix, gain=math.sqrt(2))

        # residual projector if needed -----------------------------------------
        self.residual_proj = (
            nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        )

    # ---------------------------------------------------------------------
    # forward helpers -------------------------------------------------------
    # ---------------------------------------------------------------------
    def _compute_attn_coeffs(
        self,
        node_feats: Tensor,
        edge_feats: Tensor,
        adj: Tensor,
        num_nodes: int,
    ) -> Tensor:
        row = node_feats.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)
        col = row.transpose(1, 2)
        edge = edge_feats.unsqueeze(3).expand(-1, -1, -1, self.num_attn_heads, -1)
        concat = torch.cat((row, col, edge), dim=4)
        logits = concat @ self.attn_matrix.T
        logits = logits.sum(dim=-1)
        
        # Use a large negative value instead of -inf for numerical stability
        mask_value = -1e9  # A large negative number instead of -inf
        logits = logits.masked_fill(adj.unsqueeze(-1) == 0, mask_value)
        
        # Apply LeakyReLU and clip for numerical stability
        logits = self.attn_leaky_relu(logits)
        
        # Clamp values to prevent extreme values going into softmax
        logits = torch.clamp(logits, min=-1e9, max=1e9)
        
        # Apply softmax and handle NaNs
        coeffs = F.softmax(logits, dim=2)
        
        # Replace any remaining NaNs with zeros
        coeffs = torch.nan_to_num(coeffs, 0.0)
        
        return coeffs

    def _execute_message_passing(self, node_feats: Tensor, coeffs: Tensor, B: int, N: int) -> Tensor:
        msg = torch.einsum("bmax,bnma->bnax", node_feats, coeffs)
        return msg.reshape(B, N, -1)

    # ---------------------------------------------------------------------
    def forward(self, data: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        node, edge, adj = [t.to(self.device) for t in data]
        B, N, _ = node.shape

        # preprocess -------------------------------------------------------
        node_res = node
        edge_res = edge

        node = self.node_projection(self.layer_norm_1(node))
        edge = self.edge_mlp(self.layer_norm_2(edge)) + edge_res
        # **NEW normalisation on edge stream**
        edge = self.layer_norm_edge_out(edge)

        node = node.view(B, N, self.num_attn_heads, -1)

        coeffs = self._compute_attn_coeffs(node, edge, adj, N)
        node = self._execute_message_passing(node, coeffs, B, N)

        node = self.out_node_projection(node)
        node = self.dropout(node)

        node = node + self.residual_proj(node_res)
        if self.use_leaky_relu:
            node = self.leaky_relu(node)
        node = self.layer_norm_out(node)
        return node, edge, adj


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

        self.global_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.use_cross = use_cross
        if use_cross:
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.gamma = nn.Parameter(torch.tensor(0.5))

        self.norm = nn.LayerNorm(embed_dim)
        # **NEW LayerNorm inside FFN**
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
        x, e, a = self.local((x, e, a))

        # 2. global self‑attention -----------------------------------------
        g_out, _ = self.global_attn(x, x, x)

        # 3. optional cross‑graph attention --------------------------------
        if self.use_cross and context is not None and context.size(-1) == x.size(-1):
            c_out, _ = self.cross_attn(x, context, context)
        else:
            c_out = torch.zeros_like(x)

        # 4. gated fusion & feed‑forward -----------------------------------
        res = self.local.residual_proj(residual) if hasattr(self.local, "residual_proj") else residual
        fused = res + x + self.gamma * g_out + self.gamma * c_out
        fused = self.norm(fused)
        out = self.mlp(fused)
        return out, e, a


class GlobalAttentionPooling(nn.Module):
    """Node → graph embedding via attention pooling."""

    def __init__(self, in_features: int, out_features: int = 1, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.global_attn = nn.Linear(in_features, 1)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        scores = F.softmax(self.global_attn(x), dim=1)
        scores = self.dropout(scores)
        pooled = (scores.transpose(1, 2) @ x).squeeze(1)
        pooled = self.dropout(pooled)
        return self.proj(pooled)


class GraphAttentionEncoder(nn.Module):
    """Stack of GPS layers followed by global pooling."""

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
        pooling_dim: int,
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
        self.global_pool = GlobalAttentionPooling(in_features=out_features, out_features=out_features, hidden_dim=pooling_dim, dropout=dropout)

    def forward(self, node_feats: Tensor, edge_feats: Tensor, adj: Tensor, *, context: Tensor | None = None) -> Tensor:
        x, e, a = node_feats, edge_feats, adj
        for layer in self.gps_layers:
            x, e, a = layer((x, e, a), context=context)
        return self.global_pool(x)


class DualGraphAttentionNetwork(nn.Module):
    """Drug‑graph encoder + Protein‑graph encoder + regression head."""

    def __init__(
        self,
        drug_in_features: int,
        prot_in_features: int,
        *,
        hidden_size: int = 64,
        emb_size: int = 64,
        drug_edge_features: int = 17,
        prot_edge_features: int = 1,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        mlp_dropout: float = 0.2,
        pooling_dim: int = 128,
        mlp_hidden: int = 128,
        device: Union[str, torch.device] = "cpu",
        z_emb_dim: int = 16,
        num_atom_types: int = 100,
        num_res_types: int = 40,
        use_cross: bool = False,
    ) -> None:
        super().__init__()
        device = torch.device(device)

        # --- embedding + normalisation ------------------------------------
        self.drug_feat_prep = FeaturePrep(num_atom_types, z_emb_dim)
        self.prot_feat_prep = FeaturePrep(num_res_types, z_emb_dim)
        self.drug_ln = nn.LayerNorm(drug_in_features + z_emb_dim)
        self.prot_ln = nn.LayerNorm(prot_in_features + z_emb_dim)

        # --- encoders ------------------------------------------------------
        self.drug_encoder = GraphAttentionEncoder(
            drug_in_features + z_emb_dim,
            hidden_size,
            emb_size,
            drug_edge_features,
            num_layers,
            num_heads,
            dropout,
            mlp_dropout,
            pooling_dim,
            device,
            use_cross=use_cross,
        )
        self.prot_encoder = GraphAttentionEncoder(
            prot_in_features + z_emb_dim,
            hidden_size,
            emb_size,
            prot_edge_features,
            num_layers,
            num_heads,
            dropout,
            mlp_dropout,
            pooling_dim,
            device,
            use_cross=use_cross,
        )

        self.co_attn = CoAttention(emb_size, num_heads, dropout)

        # graph‑level embedding norms --------------------------------------
        self.pool_norm_drug = nn.LayerNorm(emb_size)
        self.pool_norm_prot = nn.LayerNorm(emb_size)

        # prediction head ---------------------------------------------------
        self.mlp = nn.Sequential(
            nn.Linear(emb_size * 2, mlp_hidden),
            nn.CELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.CELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, 1),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        drug_atomic_ids: Tensor,
        drug_node_feats: Tensor,
        drug_edge_feats: Tensor,
        drug_adj: Tensor,
        prot_res_ids: Tensor,
        prot_node_feats: Tensor,
        prot_edge_feats: Tensor,
        prot_adj: Tensor,
    ) -> Tensor:
        # 0. embed + **LayerNorm** ----------------------------------------
        d_x = self.drug_ln(self.drug_feat_prep(drug_atomic_ids, drug_node_feats))
        p_x = self.prot_ln(self.prot_feat_prep(prot_res_ids, prot_node_feats))
        d_e, d_a = drug_edge_feats, drug_adj
        p_e, p_a = prot_edge_feats, prot_adj

        # 1. coupled GPS encoding -----------------------------------------
        for d_layer, p_layer in zip(self.drug_encoder.gps_layers, self.prot_encoder.gps_layers):
            d_x, d_e, d_a = d_layer((d_x, d_e, d_a), context=p_x)
            p_x, p_e, p_a = p_layer((p_x, p_e, p_a), context=d_x)

        # 2. extra node‑level co‑attention --------------------------------
        d_x, p_x = self.co_attn(d_x, p_x)

        # 3. global pooling + **LayerNorm** -------------------------------
        drug_emb = self.pool_norm_drug(self.drug_encoder.global_pool(d_x))
        prot_emb = self.pool_norm_prot(self.prot_encoder.global_pool(p_x))

        # 4. regression head ----------------------------------------------
        score = self.mlp(torch.cat([drug_emb, prot_emb], dim=-1)).squeeze(-1)
        return score
        
        
class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network for learning node representations and predicting pCHEMBL scores.

    Instance Attributes:
        - gat_layers: nn.Sequential containing all GAT layers in sequence
        - global_attn_pooling: Another nn.Module which conducts global attention pooling after the GAT layers
    """
    gat_layers: nn.Module
    global_attn_pooling: nn.Module

    def __init__(self, device: str | torch.device, in_features: int, out_features: int, num_edge_features: int,
                 hidden_size: int, num_layers: int, num_attn_heads: int, dropout: float, pooling_dropout: float,
                 pooling_dim: int) -> None:
        """Initialize the Graph Attention Network"""
        super().__init__()

        if num_layers == 1:
            layers = [GraphAttentionLayer(device, in_features, out_features,
                                          num_edge_features, num_attn_heads,
                                          dropout, use_leaky_relu=False)]
        else:
            layers = [GraphAttentionLayer(device, in_features, hidden_size,
                                          num_edge_features, num_attn_heads, dropout=dropout)]

            for _ in range(num_layers - 2):
                layers.append(GraphAttentionLayer(device, hidden_size, hidden_size,
                                                  num_edge_features, num_attn_heads, dropout=dropout))

            layers.append(GraphAttentionLayer(device, hidden_size, out_features,
                                              num_edge_features, num_attn_heads=1,
                                              dropout=dropout, use_leaky_relu=False))

        self.gat_layers = nn.Sequential(*layers)
        self.global_attn_pooling = GlobalAttentionPooling(out_features, 1, pooling_dim, dropout=pooling_dropout)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor,
                adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute and forward pass of the GAT
        """
        # Initial node feature shape: [B, N, F_in]
        input_tuple = (node_features, edge_features, adjacency_matrix)

        # [B, N, F_in] -> [B, N, F_out]
        updated_node_features = self.gat_layers(input_tuple)[0]

        # Perform global attention pooling for final learning process
        # [B, N, F_out] -> [B, 1]
        pchembl_scores = self.global_attn_pooling(updated_node_features)

        # Normalize pChEMBL scores into the range (0, 14) using sigmoid
        pchembl_scores = 14 * torch.sigmoid(pchembl_scores)

        # Final shape: [B, 1]
        return pchembl_scores


class GlobalAttentionPooling(nn.Module):
    """Global attention pooling layer for aggregating node features.

    Instance Attributes:
        - global_attn: A linear layer that projects input onto logits
        - final_projection: A multi-layer perceptron that acts as final projection after attention
        - dropout: Dropout probability. Defaults to 0.2.
    """
    global_attn: nn.Module
    final_projection: nn.Module
    dropout: float

    def __init__(self, in_features: int, out_features: int = 1, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        """Initialize Global Attention Pooling"""
        super().__init__()

        self.global_attn = nn.Linear(in_features, 1)
        self.dropout = nn.Dropout(dropout)

        self.final_projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Added normalization
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The input node features (x) has shape: [B, N, F_out]
        """
        attn_logits = self.global_attn(x)  # [B, N, 1]
        attn_scores = F.softmax(attn_logits, dim=1)  # Normalize across nodes

        # Apply dropout to attention scores
        attn_scores = self.dropout(attn_scores)

        # [B, 1, N] @ [B, N, F_out] -> [B, 1, F_out]
        pooled_features = attn_scores.transpose(1, 2) @ x

        # [B, 1, F_out] -> [B, F_out]
        pooled_features = pooled_features.squeeze(1)
        pooled_features = self.dropout(pooled_features)

        # [B, F_out] -> [B, 1]
        return self.final_projection(pooled_features)


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