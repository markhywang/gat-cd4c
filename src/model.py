"""Module for implementing Graph Attention Networks (GAT) components."""

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class FeaturePrep(nn.Module):
    """
    Concatenate a learnable embedding (e.g. atomic number or residue type)
    in front of the dense node-feature vector.
    """
    def __init__(self, num_types: int, emb_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_types, emb_dim)

    def forward(self, ids: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        # ids:  [B, N]   (long)
        # feats:[B, N, F] (float)
        emb = self.embed(ids)              # [B, N, emb_dim]
        return torch.cat([emb, feats], dim=-1)


class CoAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        # MultiheadAttention expects (L, B, D), we’ll treat batch=1 and transpose
        self.drug2prot = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.prot2drug = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm_drug = nn.LayerNorm(embed_dim)
        self.norm_prot = nn.LayerNorm(embed_dim)
    
    def forward(self, drug_feats, prot_feats):
        """
        drug_feats:  Tensor of shape (N_drug, D)
        prot_feats:  Tensor of shape (N_prot, D)
        returns updated (drug_feats, prot_feats) of same shapes
        """
        # add batch dim: (L, B, D) with B=1
        d = drug_feats.unsqueeze(1)    # (N_drug, 1, D)
        p = prot_feats.unsqueeze(1)    # (N_prot, 1, D)
        
        # Drug attends to Protein
        # Query=d, Key=Value=p
        attn_dp, _ = self.drug2prot(query=d, key=p, value=p)
        drug_updated = self.norm_drug(d + attn_dp).squeeze(1)  # residual + LN → (N_drug, D)
        
        # Protein attends to Drug
        attn_pd, _ = self.prot2drug(query=p, key=d, value=d)
        prot_updated = self.norm_prot(p + attn_pd).squeeze(1)  # → (N_prot, D)
        
        return drug_updated, prot_updated


# -----------------------------------------------------------------------------
# GPSLayer: combine local GAT with global self-attention in one residual block
# -----------------------------------------------------------------------------
class GPSLayer(nn.Module):
    """Combine local GAT with global self-attention + optional cross-attention."""
    def __init__(self,
                 local_layer: nn.Module,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 use_cross: bool):
        super().__init__()
        self.local = local_layer
        self.global_attn = nn.MultiheadAttention(embed_dim, num_heads,
                                                 dropout=dropout,
                                                 batch_first=True)

        self.use_cross = use_cross
        if use_cross:
            # cross-attn: query self, key/value other
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads,
                                                     dropout=dropout,
                                                     batch_first=True)

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self,
                x_e_a: tuple[Tensor,Tensor,Tensor],
                context: Tensor | None = None
               ) -> tuple[Tensor,Tensor,Tensor]:
        x, edge, adj = x_e_a
        residual = x

        # --- local GAT ---
        local_x, edge, adj = self.local((x, edge, adj))

        # --- global self‑attn ---
        global_out, _ = self.global_attn(local_x, local_x, local_x)

        # --- optional cross‑attn from `context` (other graph) ---
        if self.use_cross and context is not None and context.size(-1) == local_x.size(-1):
            cross_out, _ = self.cross_attn(local_x, context, context)
        else:
            cross_out = 0

        # --- fuse and MLP ---
        res = (self.local.residual_proj(residual)
               if hasattr(self.local, 'residual_proj') else local_x)
        fused = res + local_x + global_out + cross_out
        fused = self.norm(fused)
        out = self.mlp(fused)
        return out, edge, adj


class GraphAttentionEncoder(nn.Module):
    """
    Encode a graph (drug or protein) with stacked GAT layers and global attention pooling.
    """
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
        # Build GAT stack
        layers = []
        for i in range(num_layers):
            in_f = in_features if i == 0 else hidden_size
            out_f = out_features if i == num_layers - 1 else hidden_size
            heads = 1 if i == num_layers - 1 else num_attn_heads
            # wrap the standard GAT layer into a GPS layer
            local = GraphAttentionLayer(device,
                                        in_f,
                                        out_f,
                                        num_edge_features,
                                        heads,
                                        dropout,
                                        use_leaky_relu=(i != num_layers - 1))
            layers.append(GPSLayer(local,
                                   embed_dim=out_f,
                                   num_heads=heads,
                                   dropout=dropout,
                                   use_cross=True))
        self.gat_layers = nn.ModuleList(layers)
        # Global attention pooling
        self.global_pool = GlobalAttentionPooling(
            in_features=out_features,
            out_features=out_features,
            hidden_dim=pooling_dim,
            dropout=dropout
        )

    def forward(self,
                node_feats: torch.Tensor,
                edge_feats: torch.Tensor,
                adj: torch.Tensor,
                context: torch.Tensor | None = None  # <— optional cross‑graph context
                ) -> torch.Tensor:
        x, e, a = node_feats, edge_feats, adj
        for layer in self.gat_layers:
            # pass the other graph’s node‑embeddings in as `context`
            x, e, a = layer((x, e, a), context=context)
        # now pool down to a single graph embedding
        return self.global_pool(x)


class DualGraphAttentionNetwork(nn.Module):
    """
    Combines a drug‐graph encoder and protein‐graph encoder, then an MLP for final pChEMBL prediction.
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
                 device: torch.device = torch.device("cpu"),
                 z_emb_dim: int = 16,                # NEW
                 num_atom_types: int = 100,          # NEW
                 num_res_types: int  = 40):          # NEW
        super().__init__()

        # --- prepend learnable ID embeddings ---
        self.drug_feat_prep = FeaturePrep(num_atom_types, z_emb_dim)
        self.prot_feat_prep = FeaturePrep(num_res_types,  z_emb_dim)

        # Drug and protein encoders (input dim += z_emb_dim)
        self.drug_encoder = GraphAttentionEncoder(
            drug_in_features + z_emb_dim, hidden_size, emb_size,
            drug_edge_features, num_layers, num_heads,
            dropout, pooling_dim, device
        )
        self.prot_encoder = GraphAttentionEncoder(
            prot_in_features + z_emb_dim, hidden_size, emb_size,
            prot_edge_features, num_layers, num_heads,
            dropout, pooling_dim, device
        )

        # co-attention module between graphs
        self.co_attn = CoAttention(embed_dim=emb_size, num_heads=num_heads, dropout=dropout)

        # Final MLP
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
                drug_atomic_ids: torch.Tensor,     # [B, N] (long)
                drug_node_feats: torch.Tensor,     # [B, N, F_d]
                drug_edge_feats: torch.Tensor,
                drug_adj: torch.Tensor,
                prot_res_ids: torch.Tensor,        # [B, N] (long)
                prot_node_feats: torch.Tensor,     # [B, N, F_p]
                prot_edge_feats: torch.Tensor,
                prot_adj: torch.Tensor) -> torch.Tensor:

        # --- prepend embeddings ---
        d_x = self.drug_feat_prep(drug_atomic_ids, drug_node_feats)
        p_x = self.prot_feat_prep(prot_res_ids,  prot_node_feats)
        d_e, d_a = drug_edge_feats, drug_adj
        p_e, p_a = prot_edge_feats, prot_adj

        # --- coupled GPS encoder ---
        for d_layer, p_layer in zip(self.drug_encoder.gat_layers,
                                    self.prot_encoder.gat_layers):
            d_x, d_e, d_a = d_layer((d_x, d_e, d_a), context=p_x)
            p_x, p_e, p_a = p_layer((p_x, p_e, p_a), context=d_x)
        
        # --- integrate CoAttention on node-level features ---
        B = d_x.size(0)
        updated_drug, updated_prot = [], []
        for batch in range(B):
            d_feats, p_feats = d_x[batch], p_x[batch]
            d_up, p_up = self.co_attn(d_feats, p_feats)
            updated_drug.append(d_up)
            updated_prot.append(p_up)
        
        # stack back to [B, N, D]
        d_x = torch.stack(updated_drug, dim=0)
        p_x = torch.stack(updated_prot, dim=0)

        # --- global pooling ---
        drug_emb = self.drug_encoder.global_pool(d_x)   # [B, emb_size]
        prot_emb = self.prot_encoder.global_pool(p_x)   # [B, emb_size]

        # --- prediction head ---
        x = torch.cat([drug_emb, prot_emb], dim=-1)     # [B, 2*emb_size]
        score = self.mlp(x).squeeze(-1)                 # [B]
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

        # Post-attention LayerNorm
        self.layer_norm_out = nn.LayerNorm(out_features)

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
        
        # Post norm
        new_node_features = self.layer_norm_out(new_node_features)

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
