"""Module for implementing Graph Attention Networks (GAT) components."""

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax, to_dense_batch


# -----------------------------------------------------------------------------
# RBF distance encoder
# -----------------------------------------------------------------------------
class RBFEncoder(nn.Module):
    """Gaussian radial-basis-function distance encoder.

    Converts scalar distances (Å) into a num_rbf-dimensional feature vector
    via learnable-free Gaussian basis functions.
    """

    def __init__(self, num_rbf: int = 16, d_min: float = 0.0, d_max: float = 12.0):
        super().__init__()
        centers = torch.linspace(d_min, d_max, num_rbf)
        self.register_buffer('centers', centers)
        self.gamma = (num_rbf / (d_max - d_min)) ** 2

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """d: [E] → [E, num_rbf]"""
        return torch.exp(-self.gamma * (d.unsqueeze(-1) - self.centers) ** 2)


# -----------------------------------------------------------------------------
# Sparse protein local message passing (GATv2 + RBF edge features)
# -----------------------------------------------------------------------------
class SparseGATLayer(MessagePassing):
    """GATv2-style message passing for sparse protein graphs.

    Attention coefficient for edge (i,j):
        a(i,j) = softmax_j(LeakyReLU([Wh_i || Wh_j || rbf(d_ij)] · a))

    This injects 16-D RBF-encoded distances directly into the attention score,
    making the local aggregation spatially aware without dense [N×N] tensors.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_rbf: int = 16,
                 num_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__(aggr='add', flow='source_to_target', node_dim=0)
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.out_features = out_features

        self.rbf = RBFEncoder(num_rbf)
        self.lin = nn.Linear(in_features, out_features, bias=False)
        # Attention vector: one per head, over [h_i || h_j || rbf(d)]
        self.attn = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim + num_rbf))
        self.attn_act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.residual_proj = (nn.Linear(in_features, out_features, bias=False)
                              if in_features != out_features else nn.Identity())

        nn.init.xavier_uniform_(self.attn.data)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        x:          [N, in_features]
        edge_index: [2, E]
        edge_attr:  [E, 1]  raw distances (Å)
        → [N, out_features]
        """
        residual = x
        x_proj = self.lin(x)                                       # [N, out_features]
        out = self.propagate(edge_index, x=x_proj, edge_attr=edge_attr)
        out = self.out_proj(self.dropout(out))
        return self.norm(out + self.residual_proj(residual))

    def message(self,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                index: torch.Tensor) -> torch.Tensor:
        rbf = self.rbf(edge_attr.squeeze(-1))                      # [E, num_rbf]
        E = x_i.size(0)
        xi_h = x_i.view(E, self.num_heads, self.head_dim)          # [E, H, D]
        xj_h = x_j.view(E, self.num_heads, self.head_dim)          # [E, H, D]
        rbf_h = rbf.unsqueeze(1).expand(-1, self.num_heads, -1)    # [E, H, rbf]

        cat = torch.cat([xi_h, xj_h, rbf_h], dim=-1)              # [E, H, 2D+rbf]
        attn_logits = (cat * self.attn).sum(-1)                    # [E, H]
        attn_logits = self.attn_act(attn_logits)
        # Normalize per target node across all its incoming edges
        attn_w = pyg_softmax(attn_logits, index)                   # [E, H]
        attn_w = self.dropout(attn_w)

        msg = xj_h * attn_w.unsqueeze(-1)                         # [E, H, D]
        return msg.view(E, -1)                                     # [E, out_features]


# -----------------------------------------------------------------------------
# Bipartite cross-attention: drug atoms → protein residues (or vice versa)
# -----------------------------------------------------------------------------
class BipartiteCrossAttention(MessagePassing):
    """Sparse bipartite message passing for drug-protein cross-graph attention.

    Source nodes send RBF-distance-weighted attention messages to target nodes
    via an explicit sparse edge index (cross_ei).  No dense N×M matrix is built.

    Convention: edge_index[0] = source global indices,
                edge_index[1] = target global indices.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_rbf: int = 16,
                 num_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__(aggr='add', flow='source_to_target', node_dim=0)
        assert out_features % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.out_features = out_features

        self.rbf = RBFEncoder(num_rbf)
        self.lin_src = nn.Linear(in_features, out_features, bias=False)
        self.lin_tgt = nn.Linear(in_features, out_features, bias=False)
        self.attn = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim + num_rbf))
        self.attn_act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_features, out_features)

        nn.init.xavier_uniform_(self.attn.data)

    def forward(self,
                x_src: torch.Tensor,
                x_tgt: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                num_tgt: int) -> torch.Tensor:
        """
        x_src:      [N_src, in_features]   source node embeddings
        x_tgt:      [N_tgt, in_features]   target node embeddings (query in attention)
        edge_index: [2, E]   row 0 = src idx, row 1 = tgt idx
        edge_attr:  [E, 1]   distances (Å)
        num_tgt:    total number of target nodes
        → additive delta: [N_tgt, out_features]
        """
        if edge_index.size(1) == 0:
            return torch.zeros(num_tgt, self.out_features,
                               device=x_src.device, dtype=x_src.dtype)

        src_proj = self.lin_src(x_src)    # [N_src, out_F]
        tgt_proj = self.lin_tgt(x_tgt)    # [N_tgt, out_F]
        out = self.propagate(edge_index,
                             x=(src_proj, tgt_proj),
                             edge_attr=edge_attr,
                             size=(x_src.size(0), num_tgt))
        return self.out_proj(self.dropout(out))

    def message(self,
                x_j: torch.Tensor,
                x_i: torch.Tensor,
                edge_attr: torch.Tensor,
                index: torch.Tensor) -> torch.Tensor:
        # x_j = source projected, x_i = target projected
        rbf = self.rbf(edge_attr.squeeze(-1))                      # [E, num_rbf]
        E = x_j.size(0)
        xj_h = x_j.view(E, self.num_heads, self.head_dim)
        xi_h = x_i.view(E, self.num_heads, self.head_dim)
        rbf_h = rbf.unsqueeze(1).expand(-1, self.num_heads, -1)

        cat = torch.cat([xi_h, xj_h, rbf_h], dim=-1)
        attn_logits = (cat * self.attn).sum(-1)
        attn_logits = self.attn_act(attn_logits)
        attn_w = pyg_softmax(attn_logits, index)
        attn_w = self.dropout(attn_w)

        msg = xj_h * attn_w.unsqueeze(-1)
        return msg.view(E, -1)


# -----------------------------------------------------------------------------
# GPS layer for dense drug graphs
# -----------------------------------------------------------------------------
class GPSLayer(nn.Module):
    """GPS layer for dense drug graphs: local dense GAT + global self-attention."""

    def __init__(self,
                 local_layer: nn.Module,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float):
        super().__init__()
        self.local = local_layer
        self.global_attn = nn.MultiheadAttention(embed_dim, num_heads,
                                                 dropout=dropout,
                                                 batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self,
                x_e_a: tuple[Tensor, Tensor, Tensor]
                ) -> tuple[Tensor, Tensor, Tensor]:
        x, edge, adj = x_e_a
        local_x, edge, adj = self.local((x, edge, adj))
        global_out, _ = self.global_attn(local_x, local_x, local_x)
        fused = self.norm(local_x + global_out)
        out = self.mlp(fused)
        return out, edge, adj


# -----------------------------------------------------------------------------
# GPS layer for sparse protein graphs
# -----------------------------------------------------------------------------
class SparseProteinGPSLayer(nn.Module):
    """GPS layer for sparse protein graphs.

    Three sub-steps per forward pass:
      1. Local:  SparseGATLayer — RBF-aware GATv2 message passing on protein graph.
      2. Global: nn.MultiheadAttention — global self-attention.
                 Uses to_dense_batch to unpack the sparse tensor into a padded
                 sequence [B, N_max, F] so that nn.MultiheadAttention can be used
                 directly; p_batch drives the key-padding mask.
      3. Cross:  BipartiteCrossAttention — drug atoms → protein residues via
                 the 5 Å sparse cross-edge index (cross_ei, cross_ea).
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_rbf: int = 16,
                 num_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        self.out_features = out_features

        # Step 1: local sparse GAT
        self.local = SparseGATLayer(in_features, out_features, num_rbf, num_heads, dropout)

        # Step 2: global self-attention (operates on densified batch)
        self.global_attn = nn.MultiheadAttention(out_features, num_heads,
                                                 dropout=dropout,
                                                 batch_first=True)
        self.norm_global = nn.LayerNorm(out_features)

        # Step 3: drug → protein cross-attention
        self.cross_attn = BipartiteCrossAttention(out_features, out_features,
                                                  num_rbf, num_heads, dropout)
        self.norm_cross = nn.LayerNorm(out_features)

        self.mlp = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
        )
        self.norm_mlp = nn.LayerNorm(out_features)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                p_batch: torch.Tensor,
                drug_flat: torch.Tensor,
                cross_ei: torch.Tensor,
                cross_ea: torch.Tensor) -> torch.Tensor:
        """
        x:          [N_total, in_features]   sparse concatenated protein nodes
        edge_index: [2, E_prot]              protein COO edge index
        edge_attr:  [E_prot, 1]              protein edge distances
        p_batch:    [N_total]                batch assignment vector (0 … B-1)
        drug_flat:  [B*H, out_features]      drug nodes flattened for cross-attention
        cross_ei:   [2, E_cross]             row 0 = drug global idx, row 1 = prot global idx
        cross_ea:   [E_cross, 1]             drug-protein distances (Å)
        → [N_total, out_features]
        """
        # 1. Local sparse message passing
        x = self.local(x, edge_index, edge_attr)               # [N_total, out_F]

        # 2. Global self-attention via to_dense_batch
        # to_dense_batch packs sparse nodes into [B, N_max, F] with a boolean mask
        # where True = real node.  MHA expects key_padding_mask where True = ignore,
        # so we invert the mask.
        x_dense, mask = to_dense_batch(x, p_batch)            # [B, N_max, F], mask True=real
        key_pad = ~mask                                        # True = padding (MHA convention)
        global_out, _ = self.global_attn(x_dense, x_dense, x_dense,
                                         key_padding_mask=key_pad)
        # Unpack: recover only the real-node rows using the same mask
        x = self.norm_global(x + global_out[mask])            # [N_total, out_F]

        # 3. Drug → protein cross-attention
        prot_delta = self.cross_attn(
            x_src=drug_flat,
            x_tgt=x,
            edge_index=cross_ei,
            edge_attr=cross_ea,
            num_tgt=x.size(0),
        )                                                      # [N_total, out_F]
        x = self.norm_cross(x + prot_delta)

        # 4. MLP with residual
        x = self.norm_mlp(x + self.mlp(x))
        return x


# -----------------------------------------------------------------------------
# Drug graph encoder (dense, unchanged interface)
# -----------------------------------------------------------------------------
class GraphAttentionEncoder(nn.Module):
    """Encode a drug graph (dense) with stacked GPS layers and global attention pooling."""

    def __init__(self,
                 in_features: int,
                 hidden_size: int,
                 out_features: int,
                 num_edge_features: int,
                 num_layers: int,
                 num_attn_heads: int,
                 dropout: float,
                 pooling_dim: int,
                 device: torch.device):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_f  = in_features if i == 0 else hidden_size
            out_f = out_features if i == num_layers - 1 else hidden_size
            heads = 1 if i == num_layers - 1 else num_attn_heads
            local = GraphAttentionLayer(device, in_f, out_f, num_edge_features,
                                        heads, dropout,
                                        use_leaky_relu=(i != num_layers - 1))
            layers.append(GPSLayer(local, embed_dim=out_f, num_heads=heads, dropout=dropout))
        self.gat_layers = nn.ModuleList(layers)
        self.global_pool = GlobalAttentionPooling(out_features, out_features, pooling_dim, dropout)

    def forward(self,
                node_feats: torch.Tensor,
                edge_feats: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        x, e, a = node_feats, edge_feats, adj
        for layer in self.gat_layers:
            x, e, a = layer((x, e, a))
        return self.global_pool(x)


# -----------------------------------------------------------------------------
# Sparse global attention pooling (for protein)
# -----------------------------------------------------------------------------
class SparseGlobalAttentionPooling(nn.Module):
    """Global attention pooling for sparse protein graphs.

    Converts sparse [N_total, F] + batch vector to per-graph embeddings [B, out_features].
    Uses to_dense_batch internally so padding is handled correctly.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.attn = nn.Linear(in_features, 1)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """x: [N_total, F], batch: [N_total] → [B, out_features]"""
        x_dense, mask = to_dense_batch(x, batch)              # [B, N_max, F], True=real
        logits = self.attn(x_dense)                           # [B, N_max, 1]
        logits = logits.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        scores = F.softmax(logits, dim=1)                     # [B, N_max, 1]
        scores = self.dropout(scores)
        pooled = (scores.transpose(1, 2) @ x_dense).squeeze(1)  # [B, F]
        pooled = self.dropout(pooled)
        return self.proj(pooled)                              # [B, out_features]


# -----------------------------------------------------------------------------
# Dual-encoder model (updated for sparse protein + sparse cross-graph edges)
# -----------------------------------------------------------------------------
class DualGraphAttentionNetwork(nn.Module):
    """Drug-target interaction predictor (pChEMBL regression).

    Drug graph:    dense [B, H, F]         processed by GPS layers
                   (GraphAttentionLayer + nn.MultiheadAttention).
    Protein graph: sparse COO [N_total, F] processed by SparseProteinGPSLayer
                   (SparseGATLayer + to_dense_batch MHA + BipartiteCrossAttention).
    Cross-edges:   5 Å sparse drug→protein edges built in collate_drug_prot.
    """

    def __init__(self,
                 drug_in_features: int,
                 prot_in_features: int,
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
                 device: torch.device = torch.device("cpu")):
        super().__init__()

        # Drug encoder: dense GPS layers
        self.drug_encoder = GraphAttentionEncoder(
            drug_in_features, hidden_size, emb_size, drug_edge_features,
            num_layers, num_heads, dropout, pooling_dim, device
        )

        # Protein encoder: sparse GPS layers with cross-attention
        prot_layers = []
        for i in range(num_layers):
            in_f  = prot_in_features if i == 0 else hidden_size
            out_f = emb_size         if i == num_layers - 1 else hidden_size
            heads = 1                if i == num_layers - 1 else num_heads
            prot_layers.append(SparseProteinGPSLayer(
                in_features=in_f,
                out_features=out_f,
                num_heads=heads,
                dropout=dropout,
            ))
        self.prot_gps_layers = nn.ModuleList(prot_layers)
        self.prot_pool = SparseGlobalAttentionPooling(emb_size, emb_size, pooling_dim, dropout)

        # Final regression MLP
        self.mlp = nn.Sequential(
            nn.Linear(emb_size * 2, mlp_hidden),
            nn.CELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.CELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self,
                drug_node_feats: torch.Tensor,   # [B, H, F_drug]
                drug_edge_feats: torch.Tensor,   # [B, H, H, F_e_drug]
                drug_adj: torch.Tensor,          # [B, H, H]
                prot_ns: torch.Tensor,           # [N_total, F_prot]  sparse
                prot_ei: torch.Tensor,           # [2, E_prot]        COO
                prot_ea: torch.Tensor,           # [E_prot, 1]
                prot_batch: torch.Tensor,        # [N_total]
                cross_ei: torch.Tensor,          # [2, E_cross]
                cross_ea: torch.Tensor           # [E_cross, 1]
                ) -> torch.Tensor:
        d_x, d_e, d_a = drug_node_feats, drug_edge_feats, drug_adj
        p_x = prot_ns
        B = d_x.size(0)
        H = d_x.size(1)

        for d_layer, p_layer in zip(self.drug_encoder.gat_layers,
                                    self.prot_gps_layers):
            # Drug: dense local GAT + global self-attention
            d_x, d_e, d_a = d_layer((d_x, d_e, d_a))

            # Protein: sparse local GAT + global self-attn + drug→protein cross-attn
            # drug_flat provides the source embeddings for BipartiteCrossAttention.
            # cross_ei[0] indexes into drug_flat; cross_ei[1] indexes into p_x.
            drug_flat = d_x.reshape(B * H, -1)               # [B*H, F_drug_out]
            p_x = p_layer(p_x, prot_ei, prot_ea, prot_batch,
                          drug_flat, cross_ei, cross_ea)

        drug_emb = self.drug_encoder.global_pool(d_x)        # [B, emb_size]
        prot_emb = self.prot_pool(p_x, prot_batch)           # [B, emb_size]

        return self.mlp(torch.cat([drug_emb, prot_emb], dim=-1))  # [B, 1]


# =============================================================================
# Legacy classes — kept for benchmark.py and GraphAttentionNetwork compatibility
# =============================================================================

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


class GraphAttentionLayer(nn.Module):
    """Single Graph attention layer for performing message passing on graph.

    Instance Attributes:
        - device (torch.device): The device to perform computations on.
        - node_projection (nn.Module): Linear transformation applied to input node features.
        - layer_norm_1 (nn.Module): Layer normalization applied to input node features.
        - layer_norm_2 (nn.Module): Layer normalization applied to edge features.
        - edge_mlp (nn.Module): MLP used for edge feature transformation.
        - use_leaky_relu (bool): Whether to use LeakyReLU activation.
        - leaky_relu (nn.Module): LeakyReLU activation function.
        - num_attn_heads (int): Number of attention heads.
        - head_size (int): Size of each attention head.
        - attn_matrix (nn.Parameter): Attention weight matrix.
        - attn_leaky_relu (nn.Module): LeakyReLU activation function for attention scores.
        - out_node_projection (nn.Module): Linear transformation applied after attention computation.
        - dropout (nn.Module): Dropout layer to prevent overfitting.
        - residual_proj (nn.Module): Linear transformation for residual connection, or Identity if not needed.
    """
    device: str | torch.device
    node_projection: nn.Module
    layer_norm_1: nn.Module
    layer_norm_2: nn.Module
    edge_mlp: nn.Module
    use_leaky_relu: bool
    leaky_relu: nn.Module
    num_attn_heads: int
    head_size: int
    attn_matrix: nn.Module
    attn_leaky_relu: nn.Module
    out_node_projection: nn.Module
    dropout: nn.Module
    residual_proj: nn.Module

    def __init__(self, device: str | torch.device, in_features: int, out_features: int,
                 num_edge_features: int, num_attn_heads: int = 1, dropout: float = 0.2,
                 use_leaky_relu: bool = True) -> None:
        """Initialize a single GAT layer"""
        super().__init__()
        self.device = device

        self.node_projection = nn.Linear(in_features, out_features)
        self.layer_norm_1 = nn.LayerNorm(in_features)

        self.edge_mlp = nn.Sequential(
            nn.Linear(num_edge_features, 2 * num_edge_features),
            nn.GELU(),
            nn.Linear(2 * num_edge_features, num_edge_features)
        )
        self.layer_norm_2 = nn.LayerNorm(num_edge_features)

        self.use_leaky_relu = use_leaky_relu
        if use_leaky_relu:
            self.leaky_relu = nn.LeakyReLU(0.2)

        self.num_attn_heads = num_attn_heads
        self.head_size = out_features // num_attn_heads
        self.attn_matrix = nn.Parameter(torch.empty((num_attn_heads, 2 * self.head_size + num_edge_features)))
        self.attn_leaky_relu = nn.LeakyReLU(0.2)

        # Final MLP layer
        self.out_node_projection = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)

        # Initialize necessary parameters using a Xavier uniform distribution.
        nn.init.xavier_uniform_(self.node_projection.weight.data, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.attn_matrix.data, gain=math.sqrt(2))

        # Add residual projection if in_features doesn't match out_features.
        if in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the forward pass of the graph attention layer."""
        # Initial node_features shape: [B, N, F_in]
        node_features, edge_features, adjacency_matrix = [t.to(self.device) for t in x]
        batch_size, num_nodes, _ = node_features.shape

        # Save node and edge residual for later addition.
        node_residual = node_features
        edge_residual = edge_features

        # [B, N, F_in] -> [B, N, F_out]
        new_node_features = self.node_projection(self.layer_norm_1(node_features))

        # [B, N, N, F_edge] -> [B, N, N, F_edge]
        edge_normalized = self.layer_norm_2(edge_features)
        edge_update = self.edge_mlp(edge_normalized)
        new_edge_features = edge_update + edge_residual

        # Split the node_features for every attention head.
        # [B, N, F_out] -> [B, N, num_heads, F_out // num_heads]
        new_node_features = new_node_features.view(batch_size, num_nodes, self.num_attn_heads, -1)

        # attn_coeffs shape: [B, N, N, num_heads]
        attn_coeffs = self._compute_attn_coeffs(new_node_features, new_edge_features, adjacency_matrix, num_nodes)

        # [B, N, num_heads, F_out // num_attn_heads] -> [B, N, F_out]
        new_node_features = self._execute_message_passing(new_node_features, attn_coeffs, batch_size, num_nodes)

        # Do final projection and dropout
        # The shape remains [B, N, F_out]
        new_node_features = self.out_node_projection(new_node_features)
        new_node_features = self.dropout(new_node_features)

        # Apply residual connection.
        # If dimensions differ, project the residual to the correct dimension.
        node_residual = self.residual_proj(node_residual)
        new_node_features = new_node_features + node_residual

        # Optionally apply activation.
        if self.use_leaky_relu:
            new_node_features = self.leaky_relu(new_node_features)

        return new_node_features, new_edge_features, adjacency_matrix

    def _compute_attn_coeffs(self, node_features: torch.Tensor, edge_features: torch.Tensor,
                             adjacency_matrix: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute attention coefficients for message passing."""
        # [B, N, num_heads, F_out // num_heads] -> [B, N, N, num_heads, F_out // num_heads]
        row_node_features = node_features.unsqueeze(2).repeat(1, 1, num_nodes, 1, 1)

        # [B, N, N, num_heads, F_out // num_heads] -> [B, N, N, num_heads, F_out // num_heads]
        # Although dimensions are the same, however 2nd and 3rd dimension are transposed
        col_node_features = row_node_features.transpose(1, 2)

        # [B, N, N, F_edge] -> [B, N, N, num_heads, F_edge]
        unsqueezed_edge_features = edge_features.unsqueeze(3).repeat(1, 1, 1, self.num_attn_heads, 1)

        # [B, N, N, num_heads, 2 * (F_out // num_heads) + F_edge]
        attn_input = torch.cat((row_node_features, col_node_features, unsqueezed_edge_features), dim=4)

        # [B, N, N, num_heads, 2 * (F_out // num_heads) + F_edge] @ [2 * (F_out // num_heads) + F_edge, num_heads]
        # --> [B, N, N, num_heads, num_heads]
        attn_logits = attn_input @ self.attn_matrix.transpose(0, 1)

        # [B, N, N, num_heads, num_heads] -> [B, N, N, num_heads]
        attn_logits = attn_logits.sum(dim=-1)

        # [B, N, N] -> [B, N, N, 1]
        reshaped_adjacency_matrix = adjacency_matrix.unsqueeze(-1)

        # Apply attention masking, similar to that of autoregression
        # The shape is still [B, N, N, num_heads]
        attn_logits = attn_logits.masked_fill(reshaped_adjacency_matrix == 0, float('-inf'))

        # Use LeakyReLU then normalize all values using softmax
        # The shape is still [B, N, N, num_heads]
        attn_logits = self.attn_leaky_relu(attn_logits)
        attn_coeffs = F.softmax(attn_logits, dim=2)

        # Any nodes that don't have any connections (i.e. nodes created to pad the input data to the
        # required size) will have all their attention logits equal to -inf. In this case, softmax will
        # output NaN, so replace all NaN values with 0.
        attn_coeffs = attn_coeffs.nan_to_num(0)

        # Final shape: [B, N, N, num_heads]
        return attn_coeffs

    def _execute_message_passing(self, node_features: torch.Tensor, attn_coeffs: torch.Tensor,
                                 batch_size: int, num_nodes: int) -> torch.Tensor:
        """Perform message passing based on computed attention coefficients."""
        # [B, N, num_heads, F_out // num_heads] EINSUM [B, N, N, num_heads]
        # -> [B, N, num_heads, F_out // num_heads]
        new_node_features = torch.einsum('bmax, bnma -> bnax', node_features, attn_coeffs)

        # Concatenate output for different attention heads together.
        # [B, N, num_heads, F_out // num_heads] -> [B, N, F_out]
        return new_node_features.view(batch_size, num_nodes, -1)


if __name__ == '__main__':
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
            'torch.nn.functional',
            'torch_geometric.nn',
            'torch_geometric.utils',
        ],
        'disable': ['R0914', 'E1101', 'R0913', 'R0902', 'E9959'],
        'allowed-io': ['main'],
        'max-line-length': 120,
    })
