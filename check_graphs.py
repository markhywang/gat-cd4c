import torch
import glob
import os
from torch_geometric.data import Data

# Add necessary torch_geometric classes to safe globals
torch.serialization.add_safe_globals([Data])

def check_graph(path):
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
        has_nan_x = torch.isnan(data.x).any()
        has_nan_edge = torch.isnan(data.edge_attr).any()
        if has_nan_x or has_nan_edge:
            print(f"Found NaN in {path}:")
            if has_nan_x:
                print("  - NaN in node features")
            if has_nan_edge:
                print("  - NaN in edge features")
            return True
        return False
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return False

def main():
    graph_dir = "../data/protein_graphs"
    if not os.path.exists(graph_dir):
        print(f"Directory {graph_dir} does not exist")
        return

    graph_files = glob.glob(os.path.join(graph_dir, "*.pt"))
    if not graph_files:
        print(f"No .pt files found in {graph_dir}")
        return

    print(f"Checking {len(graph_files)} protein graphs...")
    bad_files = []
    for f in graph_files:
        if check_graph(f):
            bad_files.append(f)

    if bad_files:
        print("\nFound bad files:")
        for f in bad_files:
            print(f"  {f}")
        print(f"\nTotal: {len(bad_files)} bad files")
    else:
        print("\nNo bad files found!")

if __name__ == "__main__":
    main() 