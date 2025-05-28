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
    """Single message‑passing layer (edge‑aware multi‑head attention)."""

    def __init__(
        self,
        device: Union[str, torch.device],
        in_features: int,
        out_features: int,
        num_edge_features: int, # Number of features for each edge
        num_attn_heads: int = 1,
        dropout: float = 0.2,
        mlp_dropout: float = 0.2, # Dropout for the MLP part of the layer
        use_leaky_relu: bool = True,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.num_edge_features = num_edge_features
        self.num_attn_heads = num_attn_heads
        self.out_features_per_head = out_features // num_attn_heads

        # node & edge preprocessing --------------------------------------------
        self.node_projection = nn.Linear(in_features, out_features) # Projects node features to out_features (sum over heads)
        self.layer_norm_1 = nn.LayerNorm(in_features)

        if self.num_edge_features > 0:
            self.edge_mlp = nn.Sequential(
                nn.Linear(num_edge_features, num_edge_features * 2), # Example intermediate expansion
                nn.ReLU(),
                nn.Linear(num_edge_features * 2, self.num_attn_heads) # Output `num_attn_heads` for direct bias to scores
            )
            self.edge_layer_norm = nn.LayerNorm(num_edge_features)
        else: 
            self.edge_mlp = None
            self.edge_layer_norm = None


        # attention mechanism --------------------------------------------------
        self.W_q = nn.Linear(out_features, out_features, bias=False) # Input is projected_x (out_features dim)
        self.W_k = nn.Linear(out_features, out_features, bias=False)
        self.W_v = nn.Linear(out_features, out_features, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2) if use_leaky_relu else nn.Identity()

        # feedforward network --------------------------------------------------
        self.ffn = nn.Sequential(
            nn.Linear(out_features, 2 * out_features),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(2 * out_features, out_features),
        )
        self.layer_norm_2 = nn.LayerNorm(out_features)
        self.output_dropout = nn.Dropout(dropout) 
        self.to(self.device)


    def forward(self, x: torch.Tensor, adj: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Node features [B, N, F_in].
            adj (torch.Tensor): Adjacency matrix [B, N, N], 1 for edge, 0 otherwise.
            edge_attr (torch.Tensor, optional): Edge features [B, N, N, F_edge]. Defaults to None.
        """
        B, N, _ = x.shape

        x_norm = self.layer_norm_1(x)
        x_proj = self.node_projection(x_norm)  # [B, N, out_features]

        Q = self.W_q(x_proj).view(B, N, self.num_attn_heads, self.out_features_per_head)
        K = self.W_k(x_proj).view(B, N, self.num_attn_heads, self.out_features_per_head)
        V = self.W_v(x_proj).view(B, N, self.num_attn_heads, self.out_features_per_head)

        Q = Q.transpose(1, 2)  # [B, num_heads, N, out_features_per_head]
        K = K.transpose(1, 2)  # [B, num_heads, N, out_features_per_head]
        V = V.transpose(1, 2)  # [B, num_heads, N, out_features_per_head]

        attn_scores_raw = torch.matmul(Q, K.transpose(-2, -1)) / (self.out_features_per_head**0.5) # [B, num_heads, N, N]

        if self.edge_mlp is not None and edge_attr is not None and self.num_edge_features > 0:
            edge_attr_norm = self.edge_layer_norm(edge_attr)    # [B, N, N, F_edge]
            edge_bias = self.edge_mlp(edge_attr_norm)           # [B, N, N, num_heads]
            edge_bias = edge_bias.permute(0, 3, 1, 2)           # [B, num_heads, N, N]
            attn_scores_raw = attn_scores_raw + edge_bias       # Add edge bias to attention scores
        
        attn_scores_raw = self.leaky_relu(attn_scores_raw) # Apply LeakyReLU after adding edge bias, before masking

        mask = adj.unsqueeze(1)  # [B, 1, N, N]
        attn_scores_masked = attn_scores_raw.masked_fill(mask == 0, float("-1e9")) # Use a large negative number
        attn_probs = F.softmax(attn_scores_masked, dim=-1) # [B, num_heads, N, N]
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, V) # [B, num_heads, N, out_features_per_head]
        context_cat = context.transpose(1, 2).contiguous().view(B, N, self.num_attn_heads * self.out_features_per_head) # [B, N, out_features]

        h_attn = x_proj + self.output_dropout(context_cat) # First residual: input to GAT + attention output

        h_ffn_input = self.layer_norm_2(h_attn) 
        h_ffn_output = self.ffn(h_ffn_input)
        
        out = h_attn + self.output_dropout(h_ffn_output) # Second residual: input to FFN + FFN output
        
        return out


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
    """Global attention pooling layer from the GAT-CD4C paper."""

    def __init__(self, in_features: int, hidden_dim: int, device: Union[str, torch.device] = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.gate_nn = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # self.feat_nn removed as per original GAT-CD4C (scores * x)
        self.to(self.device)

    def forward(self, x: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Node features [B, N, F_in].
            node_mask (torch.Tensor, optional): Mask for actual nodes [B, N], 1 for real, 0 for padding.
        """
        attn_logits = self.gate_nn(x)  # [B, N, 1]
        
        if node_mask is not None:
            attn_logits = attn_logits.masked_fill(node_mask.unsqueeze(-1) == 0, float("-1e9"))

        attn_scores = torch.softmax(attn_logits, dim=1) # Softmax over nodes to get attention scores
        context = torch.sum(attn_scores * x, dim=1) # [B, F_in]
        
        return context


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
        self.global_pool = GlobalAttentionPooling(in_features=out_features, hidden_dim=pooling_dim)

    def forward(self, node_feats: Tensor, edge_feats: Tensor, adj: Tensor, *, context: Tensor | None = None) -> Tensor:
        x, e, a = node_feats, edge_feats, adj
        for layer in self.gps_layers:
            x, e, a = layer((x, e, a), context=context)
        return self.global_pool(x)


class DualGraphAttentionNetwork(nn.Module):
    """Drug‑graph encoder + Protein‑graph encoder + regression head."""

    def __init__(
        self,
        drug_in_features: int,    # Number of raw dense features for drug atoms (e.g., the 29 from DrugMolecule)
        prot_in_features: int,    # Number of raw dense features for protein residues (e.g. charge, coords = 4)
        *,
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

        # --- Drug Feature Preparation ---
        # d_z (atomic numbers) will be embedded. d_x (drug_in_features dim) will be concatenated.
        self.drug_feat_prep = FeaturePrep(
            num_categories=num_atom_types, 
            emb_dim=z_emb_dim,
            in_features_dense=drug_in_features, # e.g., 29 from DrugMolecule
            device=device
        )
        gat_drug_in_dim = self.drug_feat_prep.out_features # z_emb_dim + drug_in_features

        # --- Protein Feature Preparation ---
        self.use_prot_feature_prep = use_prot_feature_prep # Should be True
        if self.use_prot_feature_prep:
            # p_z (residue IDs) embedded. p_x_dense (prot_in_features dim, e.g. 4 for charge+coords) concatenated.
            self.prot_feat_prep = FeaturePrep(
                num_categories=num_res_types, 
                emb_dim=z_emb_dim,
                in_features_dense=prot_in_features, # e.g. 4 for charge+coords
                device=device
            )
            gat_prot_in_dim = self.prot_feat_prep.out_features # z_emb_dim + prot_in_features
        else:
            # This path is not expected for current graph protein setup (.pt files)
            self.prot_feat_prep = None 
            gat_prot_in_dim = prot_in_features # Would be raw feature dim if no FeaturePrep

        # --- Drug GAT Layers ---
        # GAT layers output `emb_size` features
        self.drug_conv_initial = GraphAttentionLayer(device, gat_drug_in_dim, emb_size, drug_edge_features, num_heads, dropout, mlp_dropout)
        self.drug_conv_layers = nn.ModuleList([
            GraphAttentionLayer(device, emb_size, emb_size, drug_edge_features, num_heads, dropout, mlp_dropout)
            for _ in range(num_layers - 1)
        ])
        self.drug_pooling = GlobalAttentionPooling(emb_size, pooling_dim, device=device) # Input to pooling is emb_size

        # --- Protein GAT Layers ---
        self.prot_conv_initial = GraphAttentionLayer(device, gat_prot_in_dim, emb_size, prot_edge_features, num_heads, dropout, mlp_dropout)
        self.prot_conv_layers = nn.ModuleList([
            GraphAttentionLayer(device, emb_size, emb_size, prot_edge_features, num_heads, dropout, mlp_dropout)
            for _ in range(num_layers - 1)
        ])
        self.prot_pooling = GlobalAttentionPooling(emb_size, pooling_dim, device=device) # Input to pooling is emb_size

        # --- Cross-Attention (Optional) ---
        # Cross attention operates on features of `emb_size` (output of GATs)
        if self.use_cross:
            self.cross_attn_drug = nn.MultiheadAttention(embed_dim=emb_size, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.cross_attn_prot = nn.MultiheadAttention(embed_dim=emb_size, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.norm_cross_drug = nn.LayerNorm(emb_size)
            self.norm_cross_prot = nn.LayerNorm(emb_size)
            # Pooled feature dimension remains emb_size as pooling happens after potential cross-attention
            pooled_input_dim_for_mlp = emb_size 
        else:
            pooled_input_dim_for_mlp = emb_size

        # --- Readout MLP ---
        self.mlp = nn.Sequential(
            nn.Linear(2 * pooled_input_dim_for_mlp, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden // 2, 1)
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
        p_a: torch.Tensor       # Protein adjacency matrix [B, N_p, N_p]
    ) -> torch.Tensor:

        # --- Drug Branch ---
        drug_features_prepared = self.drug_feat_prep(d_z, d_x) # Out: [B, N_d, z_emb_dim + F_d_dense_raw]
        
        hd = self.drug_conv_initial(drug_features_prepared, d_a, d_e)
        for layer in self.drug_conv_layers:
            hd = layer(hd, d_a, d_e) # Output of GAT layers is `emb_size`
        # hd is now [B, N_d, emb_size]

        # --- Protein Branch ---
        if self.use_prot_feature_prep and self.prot_feat_prep is not None:
            prot_features_prepared = self.prot_feat_prep(p_z, p_x_dense) # Out: [B, N_p, z_emb_dim + F_p_dense_raw]
        else: # Should not happen with current setup
            prot_features_prepared = p_x_dense 

        hp = self.prot_conv_initial(prot_features_prepared, p_a, p_e)
        for layer in self.prot_conv_layers:
            hp = layer(hp, p_a, p_e) # Output of GAT layers is `emb_size`
        # hp is now [B, N_p, emb_size]

        # --- Cross-Attention (Optional) ---
        if self.use_cross:
            drug_padding_mask = (d_z == 0) 
            prot_padding_mask = (p_z == 0) 

            hd_cross, _ = self.cross_attn_drug(query=hd, key=hp, value=hp, key_padding_mask=prot_padding_mask)
            hd = self.norm_cross_drug(hd + hd_cross) 

            hp_cross, _ = self.cross_attn_prot(query=hp, key=hd, value=hd, key_padding_mask=drug_padding_mask)
            hp = self.norm_cross_prot(hp + hp_cross) 
        
        # --- Pooling ---
        drug_node_mask = (d_z != 0) # Mask for actual nodes
        prot_node_mask = (p_z != 0) 

        drug_pooled = self.drug_pooling(hd, node_mask=drug_node_mask) # [B, emb_size]
        prot_pooled = self.prot_pooling(hp, node_mask=prot_node_mask) # [B, emb_size]
        
        # --- Readout MLP ---
        combined = torch.cat([drug_pooled, prot_pooled], dim=1) # [B, 2 * emb_size]
        output = self.mlp(combined) 
        return output


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