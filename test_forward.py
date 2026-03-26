"""
Diagnostic forward-pass test for the refactored DualGraphAttentionNetwork.

Builds minimal dummy tensors matching the new sparse signatures and runs one
forward pass, printing tensor shapes at each stage.
"""

import sys
sys.path.insert(0, 'src')

import torch
from model import DualGraphAttentionNetwork

# ── Config ────────────────────────────────────────────────────────────────────
B          = 2      # batch size
H          = 80     # drug hard limit (max_nodes)
F_DRUG_N   = 29     # drug node features
F_DRUG_E   = 17     # drug edge features
F_PROT_N   = 24     # protein node features (one-hot AA + charge + coords)
HIDDEN     = 32     # smaller hidden for speed
EMB        = 32
N_LAYERS   = 2
N_HEADS    = 4
N_PROT     = [50, 40]  # residues per protein in the batch

torch.manual_seed(0)
device = torch.device('cpu')

# ── Drug tensors (dense) ──────────────────────────────────────────────────────
drug_ns = torch.randn(B, H, F_DRUG_N)
drug_es = torch.randn(B, H, H, F_DRUG_E)
# Symmetric adjacency with self-loops masked out
drug_as = torch.randint(0, 2, (B, H, H)).float()
for b in range(B):
    drug_as[b] = (drug_as[b] + drug_as[b].t()).clamp(0, 1)
    drug_as[b].fill_diagonal_(0)

# ── Protein tensors (sparse COO) ──────────────────────────────────────────────
prot_ns_list, prot_ei_list, prot_ea_list, prot_batch_list = [], [], [], []
node_offset = 0
for b_idx, n in enumerate(N_PROT):
    # Node features
    prot_ns_list.append(torch.randn(n, F_PROT_N))
    # Sequential + a few random spatial edges (undirected)
    seq_src = torch.arange(n - 1)
    seq_dst = torch.arange(1, n)
    rand_src = torch.randint(0, n, (20,))
    rand_dst = torch.randint(0, n, (20,))
    src = torch.cat([seq_src, seq_dst, rand_src, rand_dst])
    dst = torch.cat([seq_dst, seq_src, rand_dst, rand_src])
    # Remove self-loops
    mask = src != dst
    src, dst = src[mask], dst[mask]
    ei = torch.stack([src, dst]) + node_offset   # offset into global index
    ea = torch.rand(ei.size(1), 1) * 10.0        # distances 0–10 Å
    prot_ei_list.append(ei)
    prot_ea_list.append(ea)
    prot_batch_list.append(torch.full((n,), b_idx, dtype=torch.long))
    node_offset += n

prot_ns    = torch.cat(prot_ns_list, dim=0)     # [N_total, F_PROT_N]
prot_ei    = torch.cat(prot_ei_list, dim=1)     # [2, E_prot]
prot_ea    = torch.cat(prot_ea_list, dim=0)     # [E_prot, 1]
prot_batch = torch.cat(prot_batch_list, dim=0)  # [N_total]

N_TOTAL = prot_ns.size(0)

# ── Cross-graph edges (drug global idx → protein global idx) ──────────────────
# Drug global idx = b_idx * H + local_drug_idx  (convention from collate_drug_prot)
E_CROSS = 30
cross_drug_idx = torch.cat([
    torch.randint(0,      H, (E_CROSS // 2,)),   # sample from batch 0 drug atoms
    torch.randint(H, 2 * H, (E_CROSS // 2,)),    # sample from batch 1 drug atoms
])
# Protein global offsets: batch 0 → 0..N_PROT[0]-1, batch 1 → N_PROT[0]..N_TOTAL-1
cross_prot_idx = torch.cat([
    torch.randint(0,         N_PROT[0],           (E_CROSS // 2,)),
    torch.randint(N_PROT[0], N_TOTAL,             (E_CROSS // 2,)),
])
cross_ei = torch.stack([cross_drug_idx, cross_prot_idx])  # [2, E_cross]
cross_ea = torch.rand(E_CROSS, 1) * 5.0                   # distances 0–5 Å

# ── Print input shapes ────────────────────────────────────────────────────────
print("=== Input shapes ===")
print(f"  drug_ns:    {tuple(drug_ns.shape)}")
print(f"  drug_es:    {tuple(drug_es.shape)}")
print(f"  drug_as:    {tuple(drug_as.shape)}")
print(f"  prot_ns:    {tuple(prot_ns.shape)}  (N_total={N_TOTAL})")
print(f"  prot_ei:    {tuple(prot_ei.shape)}")
print(f"  prot_ea:    {tuple(prot_ea.shape)}")
print(f"  prot_batch: {tuple(prot_batch.shape)}")
print(f"  cross_ei:   {tuple(cross_ei.shape)}")
print(f"  cross_ea:   {tuple(cross_ea.shape)}")

# ── Build model ───────────────────────────────────────────────────────────────
model = DualGraphAttentionNetwork(
    drug_in_features  = F_DRUG_N,
    prot_in_features  = F_PROT_N,
    hidden_size       = HIDDEN,
    emb_size          = EMB,
    drug_edge_features= F_DRUG_E,
    prot_edge_features= 1,
    num_layers        = N_LAYERS,
    num_heads         = N_HEADS,
    dropout           = 0.0,   # 0 dropout so output is deterministic
    mlp_dropout       = 0.0,
    pooling_dim       = 64,
    mlp_hidden        = 64,
    device            = device,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n=== Model: {total_params:,} parameters ===")

# ── Forward pass ──────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    out = model(
        drug_ns, drug_es, drug_as,
        prot_ns, prot_ei, prot_ea, prot_batch,
        cross_ei, cross_ea,
    )

print(f"\n=== Output shape: {tuple(out.shape)} (expected [{B}, 1]) ===")
print(f"Output values: {out.squeeze(-1).tolist()}")
print("\nForward pass OK.")
