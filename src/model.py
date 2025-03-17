import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_edge_features: int, hidden_size: int,
                 num_layers: int, num_attn_heads: int) -> None:
        super().__init__()

        if num_layers == 1:
            layers = [GraphAttentionLayer(in_features, out_features, num_edge_features, use_leaky_relu=False)]
        else:
            layers = [GraphAttentionLayer(in_features, hidden_size, num_edge_features, num_attn_heads)]
            for i in range(num_layers - 2):
                layers.append(GraphAttentionLayer(hidden_size, hidden_size, num_edge_features, num_attn_heads))
            layers.append(GraphAttentionLayer(hidden_size, out_features, num_edge_features, use_leaky_relu=False))

        self.gat_layers = nn.Sequential(*layers)

    def forward(self, node_features, edge_features, adjacency_matrix) -> torch.Tensor:
        # Initial node feature shape: [B, N, F_in]
        input_tuple = (node_features, edge_features, adjacency_matrix)

        # [B, N, F_in] -> [B, N, F_out]
        final_node_features, _, _ = self.gat_layers(input_tuple)

        # [B, N, F_out] -> [B, 1]
        pchembl_scores = final_node_features.squeeze(2).sum(dim=1)
        return pchembl_scores


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 num_edge_features: int, num_attn_heads: int = 1, use_leaky_relu: bool = True) -> None:
        super().__init__()

        self.projection = nn.Linear(in_features, out_features)
        self.layer_norm_1 = nn.LayerNorm(in_features)

        self.use_leaky_relu = use_leaky_relu
        if use_leaky_relu:
            self.leaky_relu = nn.LeakyReLU(0.2)
            self.layer_norm_2 = nn.LayerNorm(out_features)

        self.num_attn_heads = num_attn_heads
        self.head_size = out_features // num_attn_heads
        self.attn_matrix = nn.Parameter(torch.empty((num_attn_heads, 2 * self.head_size + num_edge_features)))
        self.attn_dropout = nn.Dropout(0.2)

        # Final MLP layer because apparently that's what every person does in papers
        self.projection_dropout = nn.Dropout(0.2)
        self.out_projection = nn.Linear(out_features, out_features)

        # Initialize the projection weights and attention matrix using a Xavier uniform distribution.
        nn.init.xavier_uniform_(self.projection.weight.data, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.attn_matrix.data, gain=math.sqrt(2))

    def forward(self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_features, edge_features, adjacency_matrix = x
        batch_size, num_nodes, num_node_features = node_features.shape
        new_node_features = self.projection(self.layer_norm_1(node_features))

        # Split the node_features for every attention head.
        new_node_features = new_node_features.view(batch_size, num_nodes, self.num_attn_heads, -1)

        attn_coeffs = self._compute_attn_coeffs(new_node_features, edge_features, adjacency_matrix, num_nodes)
        new_node_features = self._execute_message_passing(new_node_features, attn_coeffs, batch_size, num_nodes)

        new_node_features = self.out_projection(new_node_features)
        new_node_features = self.projection_dropout(new_node_features)

        if self.use_leaky_relu:
            new_node_features = self.leaky_relu(self.layer_norm_2(new_node_features))

        return new_node_features, edge_features, adjacency_matrix

    def _compute_attn_coeffs(self, node_features: torch.Tensor, edge_features: torch.Tensor,
                             adjacency_matrix: torch.Tensor, num_nodes: int) -> torch.Tensor:
        # Repeat along the 3rd dimension. This will create a tensor of shape [1, num_nodes,
        # num_nodes, num_node_features].
        row_node_features = node_features.unsqueeze(2).repeat(1, 1, num_nodes, 1, 1)
        # Repeat along the 2nd dimension. This will create a tensor of shape [1, num_nodes,
        # num_nodes, num_node_features].
        col_node_features = row_node_features.swapaxes(1, 2)
        # Repeat along the 4th dimension (create one copy per attention head).
        unsqueezed_edge_features = edge_features.unsqueeze(3).repeat(1, 1, 1, self.num_attn_heads, 1)

        attn_input = torch.cat((row_node_features, col_node_features, unsqueezed_edge_features), dim=4)
        attn_logits = torch.einsum('bnmax, ax -> bnma', attn_input, self.attn_matrix)

        # Add a 4th dimension to the adjacency matrix and duplicate it over that dimension for every attention head.
        # Then, apply the mask to the attention logits.
        reshaped_adjacency_matrix = adjacency_matrix.unsqueeze(3).repeat(1, 1, 1, self.num_attn_heads)
        attn_logits[reshaped_adjacency_matrix == 0] = float('-inf')

        attn_coeffs = F.softmax(attn_logits, dim=2)

        # Andrej Karpathy does this, so I guess it works (not sure why)
        attn_coeffs = self.attn_dropout(attn_coeffs)

        return attn_coeffs

    def _execute_message_passing(self, node_features: torch.Tensor, attn_coeffs: torch.Tensor,
                                 batch_size: int, num_nodes: int) -> torch.Tensor:
        new_node_features = torch.einsum('bmax, bnma -> bnax', node_features, attn_coeffs)
        # Concatenate output for different attention heads together.
        new_node_features = new_node_features.view(batch_size, num_nodes, -1)
        return new_node_features
