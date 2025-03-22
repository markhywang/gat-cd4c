import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphAttentionNetwork(nn.Module):
    def __init__(self, device, in_features: int, out_features: int, num_edge_features: int,
                 hidden_size: int, num_layers: int, num_attn_heads: int, dropout: int, pooling_dim: int) -> None:
        super().__init__()

        if num_layers == 1:
            layers = [GraphAttentionLayer(device, in_features, out_features, num_edge_features, dropout=0.0, use_leaky_relu=False)]
        else:
            layers = [GraphAttentionLayer(device, in_features, hidden_size, num_edge_features, num_attn_heads, dropout=dropout)]
            for i in range(num_layers - 2):
                layers.append(GraphAttentionLayer(device, hidden_size, hidden_size, num_edge_features, num_attn_heads, dropout=dropout))
            layers.append(GraphAttentionLayer(device, hidden_size, out_features, num_edge_features, dropout=0.0, use_leaky_relu=False))

        self.gat_layers = nn.Sequential(*layers)
        self.global_attn_pooling = GlobalAttentionPooling(out_features, 1, pooling_dim, dropout=0.0)

    def forward(self, node_features, edge_features, adjacency_matrix) -> torch.Tensor:
        # Initial node feature shape: [B, N, F_in]
        input_tuple = (node_features, edge_features, adjacency_matrix)

        # [B, N, F_in] -> [B, N, F_out]
        updated_node_features, _, _ = self.gat_layers(input_tuple)

        # Perform global attention pooling for final learning process
        # [B, N, F_out] -> [B, 1]
        pchembl_scores = self.global_attn_pooling(updated_node_features)

        # Final shape: [B, 1]
        return pchembl_scores


class GlobalAttentionPooling(nn.Module):
    def __init__(self, in_features: int, out_features: int = 1, hidden_dim: int = 128, dropout: int = 0.2) -> None:
        super().__init__()

        self.global_attn = nn.Linear(in_features, 1)
        self.dropout = nn.Dropout(dropout)

        self.final_projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
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
    def __init__(self, device, in_features: int, out_features: int,
                 num_edge_features: int, num_attn_heads: int = 1, dropout: int = 0.2, use_leaky_relu: bool = True) -> None:
        super().__init__()
        
        self.device = device
        self.projection = nn.Linear(in_features, out_features)
        self.layer_norm_1 = nn.LayerNorm(in_features)

        self.use_leaky_relu = use_leaky_relu
        if use_leaky_relu:
            self.leaky_relu = nn.LeakyReLU(0.2)
            self.layer_norm_2 = nn.LayerNorm(out_features)

        self.num_attn_heads = num_attn_heads
        self.head_size = out_features // num_attn_heads
        self.attn_matrix = nn.Parameter(torch.empty((num_attn_heads, 2 * self.head_size + num_edge_features)))
        self.attn_leaky_relu = nn.LeakyReLU(0.2)

        # Final MLP layer because apparently that's what every person does in papers
        self.out_projection = nn.Linear(out_features, out_features)

        self.dropout = nn.Dropout(dropout)

        # Initialize the projection weights and attention matrix using a Xavier uniform distribution.
        nn.init.xavier_uniform_(self.projection.weight.data, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.attn_matrix.data, gain=math.sqrt(2))

        # Add residual projection if in_features doesn't match out_features.
        if in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Initial node_features shape: [B, N, F_in]
        node_features, edge_features, adjacency_matrix = [t.to(self.device) for t in x]
        batch_size, num_nodes, num_node_features = node_features.shape

        # Save residual for later addition.
        residual = node_features

        # [B, N, F_in] -> [B, N, F_out]
        new_node_features = self.projection(self.layer_norm_1(node_features))

        # Split the node_features for every attention head.
        # [B, N, F_out] -> [B, N, num_heads, F_out // num_heads]
        new_node_features = new_node_features.view(batch_size, num_nodes, self.num_attn_heads, -1)

        # attn_coeffs shape: [B, N, N, num_heads]
        attn_coeffs = self._compute_attn_coeffs(new_node_features, edge_features, adjacency_matrix, num_nodes)

        # [B, N, num_heads, F_out // num_attn_heads] -> [B, N, F_out]
        new_node_features = self._execute_message_passing(new_node_features, attn_coeffs, batch_size, num_nodes)

        # Do final projection and dropout
        # The shape remains [B, N, F_out]
        new_node_features = self.out_projection(new_node_features)
        new_node_features = self.dropout(new_node_features)

        # Apply residual connection.
        # If dimensions differ, project the residual to the correct dimension.
        residual = self.residual_proj(residual)
        new_node_features = new_node_features + residual

        # Optionally apply layer normalization and activation.
        if self.use_leaky_relu:
            new_node_features = self.leaky_relu(self.layer_norm_2(new_node_features))

        return new_node_features, edge_features, adjacency_matrix

    def _compute_attn_coeffs(self, node_features: torch.Tensor, edge_features: torch.Tensor,
                             adjacency_matrix: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Input node_features shape: [B, N, num_heads, F_out // num_heads]
        """
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
        """
        node_features shape: [B, N, num_heads, F_out // num_heads]
        attn_coeffs shape:   [B, N,    N     ,     num_heads     ]
        """
        # [B, N, num_heads, F_out // num_heads] EINSUM [B, N, N, num_heads]
        # -> [B, N, num_heads, F_out // num_heads]
        new_node_features = torch.einsum('bmax, bnma -> bnax', node_features, attn_coeffs)

        # Concatenate output for different attention heads together.
        # [B, N, num_heads, F_out // num_heads] -> [B, N, F_out]
        return new_node_features.view(batch_size, num_nodes, -1)
