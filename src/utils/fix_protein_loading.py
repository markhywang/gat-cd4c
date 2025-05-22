#!/usr/bin/env python3
import argparse
import hashlib
import os
import pathlib
import re
import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple, Set

# Define the hash function used to generate protein graph filenames
# This must match what embed_proteins.py used
def _generate_protein_graph_filename(identifier: str) -> str:
    MAX_IDENTIFIER_LEN_BEFORE_HASH = 100
    ALLOWED_CHARS_REGEX = r"^[A-Za-z0-9_\-\.]+$"
    is_too_long = len(identifier) > MAX_IDENTIFIER_LEN_BEFORE_HASH
    has_disallowed_chars = not re.match(ALLOWED_CHARS_REGEX, identifier)
    if is_too_long or has_disallowed_chars:
        hashed_identifier = hashlib.md5(identifier.encode()).hexdigest()
        return f"seq-{hashed_identifier}.pt"
    else:
        return f"{identifier}.pt"

def find_matching_graph_file(protein_id, protein_seq, graph_dir):
    """Find the right graph file by trying different potential identifiers"""
    candidates = [
        protein_id,                  # Simple protein ID
        protein_seq,                 # Full protein sequence
        f"{protein_id}_{protein_seq[:10]}",  # ID_seq-start 
        protein_seq[:100],           # First 100 chars of seq
        str([protein_id, protein_seq]),  # String repr of [ID, seq] list
    ]
    
    for candidate in candidates:
        filename = _generate_protein_graph_filename(candidate)
        path = os.path.join(graph_dir, filename)
        if os.path.exists(path):
            return path, candidate, filename
    
    return None, None, None

def load_dataset_proteins(dataset_file, is_csv=True):
    """Load proteins from dataset file"""
    print(f"Loading dataset from {dataset_file}")
    if is_csv:
        df = pd.read_csv(dataset_file)
    else:  # Tab-delimited
        df = pd.read_csv(dataset_file, sep='\t')
    
    # Extract protein IDs and sequences
    proteins = {}
    
    # Debug info about dataframe
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns}")
    
    # Handle DAVIS dataset format
    if 'Target_ID' in df.columns and 'Target' in df.columns:
        print(f"Found DAVIS-style columns: Target_ID and Target")
        # Check if Target_ID column contains pandas Series objects
        sample_id = df['Target_ID'].iloc[0]
        print(f"Sample Target_ID type: {type(sample_id)}, value (truncated): {str(sample_id)[:100]}")
        
        for _, row in df.iterrows():
            target_id = row['Target_ID']
            target_seq = row['Target']
            # Handle case where Target_ID might be a list
            if isinstance(target_id, list) and len(target_id) > 0:
                target_id = str(target_id[0])
            # Handle case where Target_ID is a pandas Series (should never happen in normal cases)
            elif hasattr(target_id, 'iloc') and hasattr(target_id, 'values'):
                print(f"WARNING: Found pandas Series as Target_ID! First few values: {str(target_id.head())}")
                # Take just the first value as a fallback
                try:
                    target_id = str(target_id.iloc[0])
                    print(f"Extracted first value from Series: {target_id}")
                except:
                    target_id = str(target_id)
            proteins[str(target_id)] = str(target_seq)
    # Handle KIBA dataset format        
    elif 'ID2' in df.columns and 'X2' in df.columns:
        print(f"Found KIBA-style columns: ID2 and X2")
        for _, row in df.iterrows():
            target_id = row['ID2']
            target_seq = row['X2']
            # Clean up quotation marks if present (KIBA has quoted strings)
            if isinstance(target_id, str):
                target_id = target_id.strip('"\'')
            if isinstance(target_seq, str):
                target_seq = target_seq.strip('"\'')
            proteins[str(target_id)] = str(target_seq)
    else:
        print("ERROR: Dataset file doesn't have expected columns (Target_ID/Target or ID2/X2)")
        return {}
    
    print(f"Found {len(proteins)} unique proteins in dataset")
    return proteins

def create_symlinks(proteins, graph_dir, output_dir=None, dry_run=True):
    """Create symlinks from identified files to standardized names"""
    if output_dir is None:
        output_dir = graph_dir

    os.makedirs(output_dir, exist_ok=True)
    
    found = 0
    not_found = 0
    id_to_filename = {}
    filename_to_ids = {}
    
    # First pass - find all matching files
    print("Finding matching graph files...")
    for protein_id, protein_seq in tqdm(proteins.items()):
        path, matched_identifier, filename = find_matching_graph_file(protein_id, protein_seq, graph_dir)
        
        if path:
            found += 1
            id_to_filename[protein_id] = filename
            
            if filename not in filename_to_ids:
                filename_to_ids[filename] = []
            filename_to_ids[filename].append(protein_id)
        else:
            not_found += 1
            print(f"No graph file found for protein {protein_id}")
    
    print(f"\nFound graph files for {found} proteins, missing {not_found} proteins")
    
    # Check for collisions - multiple proteins mapping to same file
    collisions = {f: ids for f, ids in filename_to_ids.items() if len(ids) > 1}
    if collisions:
        print(f"\nWARNING: Found {len(collisions)} files with multiple protein IDs:")
        for filename, ids in collisions.items():
            print(f"  {filename}: {', '.join(ids)}")
    
    # Second pass - create symlinks for simple access
    if dry_run:
        print("\nDRY RUN - would create these symlinks:")
    else:
        print("\nCreating symlinks...")
    
    for protein_id, filename in id_to_filename.items():
        src_path = os.path.join(graph_dir, filename)
        # Create a standardized destination with simple ID
        dst_filename = f"{protein_id}.pt"
        dst_path = os.path.join(output_dir, dst_filename)
        
        if os.path.exists(dst_path) and not os.path.islink(dst_path):
            print(f"WARNING: {dst_path} exists and is not a symlink. Skipping.")
            continue
        
        if dry_run:
            print(f"Would link: {src_path} -> {dst_path}")
        else:
            try:
                # Remove existing symlink if present
                if os.path.islink(dst_path):
                    os.unlink(dst_path)
                
                # Create relative symlink
                os.symlink(os.path.basename(src_path), dst_path)
                print(f"Created symlink: {dst_path} -> {os.path.basename(src_path)}")
            except Exception as e:
                print(f"Error creating symlink for {protein_id}: {e}")
    
    # Create a mapping file for future reference
    mapping_file = os.path.join(output_dir, "protein_file_mapping.csv")
    with open(mapping_file, "w") as f:
        f.write("protein_id,graph_filename\n")
        for protein_id, filename in id_to_filename.items():
            f.write(f"{protein_id},{filename}\n")
    
    print(f"\nMapping saved to {mapping_file}")

def patch_dataset_code(dataset_file):
    """Patch utils/dataset.py to handle the simple ID filenames"""
    # Read the file
    with open(dataset_file, "r") as f:
        lines = f.readlines()
    
    # Look for load_protein method
    patched = False
    for i, line in enumerate(lines):
        if "def load_protein" in line and "self" in line and "pid" in line:
            # Find the end of the function
            j = i + 1
            indent = len(line) - len(line.lstrip())
            func_end = -1
            
            while j < len(lines):
                if not lines[j].strip() or lines[j].startswith(' ' * indent):
                    j += 1
                else:
                    func_end = j
                    break
            
            if func_end > 0:
                # Insert simplified lookup before the graph filename calculation
                patch_lines = [
                    "        # First try simple ID-based filename\n",
                    "        simple_path = os.path.join(self.protein_graph_dir, f\"{pid_to_load}.pt\")\n",
                    "        if os.path.exists(simple_path):\n",
                    "            print(f\"DEBUG: Found protein graph using simple path: {simple_path}\")\n",
                    "            prot_graph = torch.load(simple_path)\n", 
                    "            return prot_graph, pid_to_load\n",
                    "\n"
                ]
                
                # Find position to insert after checking pid_to_load
                for k in range(i+1, j):
                    if "graph_filename = _generate_protein_graph_filename" in lines[k]:
                        lines[k:k] = patch_lines
                        patched = True
                        break
    
    if patched:
        # Create backup
        backup_file = dataset_file + ".bak"
        with open(backup_file, "w") as f:
            f.writelines(lines)
        
        # Write patched file
        with open(dataset_file, "w") as f:
            f.writelines(lines)
        
        print(f"Patched {dataset_file} (backup saved as {backup_file})")
    else:
        print(f"Could not patch {dataset_file} - load_protein method not found or format unexpected")

def main():
    parser = argparse.ArgumentParser(description="Fix protein graph loading by creating symlinks with simple names")
    
    parser.add_argument("--dataset", choices=["KIBA", "DAVIS", "both"], default="both",
                        help="Dataset to process (KIBA, DAVIS, or both)")
    parser.add_argument("--graph-dir", default="data/protein_graphs",
                        help="Directory where protein graphs are stored")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to store symlinks (defaults to graph-dir)")
    parser.add_argument("--davis-path", default="data/DAVIS_dataset.csv", 
                        help="Path to DAVIS dataset CSV")
    parser.add_argument("--kiba-path", default="data/kiba.tab",
                        help="Path to KIBA dataset tab-delimited file") 
    parser.add_argument("--patch-code", action="store_true",
                        help="Patch utils/dataset.py to try simple filenames first")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't create actual symlinks, just print what would be done")
    
    args = parser.parse_args()
    
    all_proteins = {}
    
    # Process DAVIS dataset
    if args.dataset in ["DAVIS", "both"]:
        davis_proteins = load_dataset_proteins(args.davis_path, is_csv=True)
        all_proteins.update(davis_proteins)
        print(f"Added {len(davis_proteins)} proteins from DAVIS")
    
    # Process KIBA dataset  
    if args.dataset in ["KIBA", "both"]:
        kiba_proteins = load_dataset_proteins(args.kiba_path, is_csv=False)
        all_proteins.update(kiba_proteins)
        print(f"Added {len(kiba_proteins)} proteins from KIBA")
    
    print(f"\nTotal unique proteins: {len(all_proteins)}")
    
    # Create symlinks
    create_symlinks(all_proteins, args.graph_dir, args.output_dir, args.dry_run)
    
    # Patch dataset code if requested
    if args.patch_code:
        patch_dataset_code("utils/dataset.py")

if __name__ == "__main__":
    main() 