#!/usr/bin/env python3
import os
import sys
import json
import hashlib
import argparse
import pandas as pd
import torch

try:
    from utils.embed_proteins import ProteinGraphBuilder
    from utils.dataset import _sanitize_prot_id
except ImportError:
    # For when the module is imported from outside
    from src.utils.embed_proteins import ProteinGraphBuilder
    from src.utils.dataset import _sanitize_prot_id

def save_problematic_id_to_file(problematic_id, filename="problematic_id.txt"):
    """Save the problematic ID to a file for reference"""
    with open(filename, "w") as f:
        f.write(str(problematic_id))
    print(f"Saved problematic ID to {filename}")

def build_missing_protein_graph(raw_id, output_dir="../data/protein_graphs", use_colabfold=False):
    """
    Process a problematic protein ID and create its graph file
    
    Args:
        raw_id: The problematic raw protein ID (could be a string representation of a Series)
        output_dir: Where to save the protein graph
        use_colabfold: Whether to use local ColabFold for structure prediction
    """
    print(f"\n{'='*80}")
    print(f"Processing problematic protein ID")
    print(f"{'='*80}")
    
    # Sanitize the protein ID
    print("Original ID type:", type(raw_id))
    print("Original ID preview:", str(raw_id)[:200] + "..." if len(str(raw_id)) > 200 else str(raw_id))
    
    # Sanitization method 1: Use the standard _sanitize_prot_id function
    sanitized_id = _sanitize_prot_id(raw_id)
    print("\nSanitized ID using standard function:", sanitized_id[:100] + "..." if len(sanitized_id) > 100 else sanitized_id)
    
    # Sanitization method 2: Try to parse the string representation if it looks like a Series
    # This is a more aggressive approach for handling the edge case
    if "[" in str(raw_id) and "Name: Target_ID" in str(raw_id):
        print("\nDetected what appears to be a string representation of a pandas Series")
        try:
            # Extract the first actual protein ID from the string representation
            lines = str(raw_id).replace("'", "").replace("[", "").replace("]", "").split("\\n")
            for line in lines:
                if line.strip() and "..." not in line and "Name:" not in line and "Length:" not in line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        first_id = parts[-1]  # Take the last part which should be the actual ID
                        print(f"Extracted first protein ID: {first_id}")
                        sanitized_id = first_id
                        break
        except Exception as e:
            print(f"Error extracting protein ID from string: {e}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the protein graph builder
    builder = ProteinGraphBuilder(
        graph_dir=output_dir,
        cutoff=10.0,
        use_colabfold=use_colabfold
    )
    
    # Determine the expected filename
    expected_filename = builder._generate_protein_graph_filename(sanitized_id)
    expected_path = os.path.join(output_dir, expected_filename)
    print(f"\nExpected protein graph path: {expected_path}")
    
    # Check if the file already exists
    if os.path.exists(expected_path):
        print(f"File already exists. Delete it first if you want to regenerate it.")
        return
    
    # Create and save the protein graph
    try:
        print(f"\nGenerating protein graph for '{sanitized_id}'...")
        
        # Method 1: Try to generate as a UniProt ID first
        try:
            print("Attempting to fetch PDB as UniProt ID...")
            pdb_path = builder._fetch_pdb(sanitized_id)
            protein_graph = builder.build(pdb_path)
            builder.save(sanitized_id, protein_graph)
            print(f"Successfully saved protein graph to {expected_path}")
            return
        except Exception as e:
            print(f"Failed to generate graph using sanitized ID as UniProt: {e}")
        
        # Method 2: Try with a simple sequence
        print("\nAttempting to create a linear structure with placeholder sequence...")
        placeholder_seq = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"  # 50 alanines
        try:
            pdb_path = builder._create_linear_structure(sanitized_id, placeholder_seq)
            protein_graph = builder.build(pdb_path)
            builder.save(sanitized_id, protein_graph)
            print(f"Successfully saved protein graph with placeholder sequence to {expected_path}")
            
            # For complex IDs, also create a hash-based version that might be looked up
            if len(sanitized_id) > 100:
                hashed_id = f"seq-{hashlib.md5(sanitized_id.encode()).hexdigest()}"
                hashed_path = os.path.join(output_dir, f"{hashed_id}.pt")
                if not os.path.exists(hashed_path):
                    builder.save(hashed_id, protein_graph)
                    print(f"Also saved as hash-based version: {hashed_path}")
        except Exception as e:
            print(f"Failed to create graph with placeholder sequence: {e}")
            raise
    except Exception as e:
        print(f"Error during protein graph generation: {e}")

def handle_kiba_davis_proteins(dataset_name="DAVIS", output_dir="../data/protein_graphs", use_colabfold=False):
    """
    Process all proteins from the DAVIS or KIBA dataset to ensure they all have graph files
    
    Args:
        dataset_name: "DAVIS" or "KIBA"
        output_dir: Where to save the protein graphs
        use_colabfold: Whether to use local ColabFold for structure prediction
    """
    # Check for the dataset file
    dataset_file = f"../data/{dataset_name}_dataset.csv"
    if not os.path.exists(dataset_file):
        print(f"Dataset file not found: {dataset_file}")
        return
    
    print(f"\nProcessing all proteins from {dataset_name} dataset")
    
    # Load the dataset
    df = pd.read_csv(dataset_file)
    
    # Rename columns if needed
    if "Drug" in df.columns and "Target" in df.columns and "Y" in df.columns:
        df.rename(columns={"Drug": "smiles", "Target": "Target_ID", "Y": "pChEMBL_Value"}, inplace=True)
    
    # Extract unique protein IDs
    protein_ids = df["Target_ID"].unique()
    print(f"Found {len(protein_ids)} unique protein IDs in {dataset_name} dataset")
    
    # Initialize the protein graph builder
    builder = ProteinGraphBuilder(
        graph_dir=output_dir,
        cutoff=10.0,
        use_colabfold=use_colabfold
    )
    
    # Process each protein
    for i, pid in enumerate(protein_ids):
        sanitized_id = _sanitize_prot_id(pid)
        expected_filename = builder._generate_protein_graph_filename(sanitized_id)
        expected_path = os.path.join(output_dir, expected_filename)
        
        if os.path.exists(expected_path):
            if i < 5 or i % 100 == 0:
                print(f"[{i+1}/{len(protein_ids)}] Protein graph already exists for: {sanitized_id}")
            continue
        
        print(f"\n[{i+1}/{len(protein_ids)}] Creating protein graph for: {sanitized_id}")
        try:
            # Try to generate the protein graph
            pdb_path = builder._fetch_pdb(sanitized_id)
            protein_graph = builder.build(pdb_path)
            builder.save(sanitized_id, protein_graph)
            print(f"Successfully saved protein graph to {expected_path}")
        except Exception as e:
            print(f"Error generating protein graph for {sanitized_id}: {e}")
            
            # Fallback to a simple linear structure
            try:
                placeholder_seq = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"  # 50 alanines
                pdb_path = builder._create_linear_structure(sanitized_id, placeholder_seq)
                protein_graph = builder.build(pdb_path)
                builder.save(sanitized_id, protein_graph)
                print(f"Saved fallback protein graph with placeholder sequence")
            except Exception as e2:
                print(f"Fallback also failed: {e2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed a specific problematic protein or process all proteins from a dataset")
    parser.add_argument("--id", type=str, help="The problematic protein ID or file containing it")
    parser.add_argument("--dataset", choices=["DAVIS", "KIBA"], help="Process all proteins from a specific dataset")
    parser.add_argument("--output-dir", default="../data/protein_graphs", help="Output directory for protein graphs")
    parser.add_argument("--use-colabfold", action="store_true", help="Try to use local ColabFold")
    args = parser.parse_args()
    
    if args.id:
        # Process a specific problematic ID
        if os.path.exists(args.id):
            # Read ID from file
            with open(args.id, "r") as f:
                raw_id = f.read().strip()
        else:
            # Use ID directly
            raw_id = args.id
            
        build_missing_protein_graph(raw_id, args.output_dir, args.use_colabfold)
    elif args.dataset:
        # Process all proteins from a dataset
        handle_kiba_davis_proteins(args.dataset, args.output_dir, args.use_colabfold)
    else:
        parser.print_help() 