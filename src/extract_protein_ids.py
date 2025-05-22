#!/usr/bin/env python3
import os
import sys
import json
import hashlib
import argparse
import pandas as pd
import numpy as np
import re
import torch
import subprocess

try:
    from utils.embed_proteins import ProteinGraphBuilder
    from utils.dataset import _sanitize_prot_id
except ImportError:
    # For when the module is imported from outside
    from src.utils.embed_proteins import ProteinGraphBuilder
    from src.utils.dataset import _sanitize_prot_id

def extract_protein_ids(complex_input):
    """
    Extract all protein IDs from complex inputs like lists, Series, or string representations
    
    Args:
        complex_input: Could be a string, list, Series, or string representation of these
        
    Returns:
        list: Extracted protein IDs
    """
    # Convert to string if needed
    if not isinstance(complex_input, str):
        complex_input = str(complex_input)
    
    # If it looks like a Series or array representation
    extracted_ids = []
    
    # Try to parse as JSON first (for proper arrays)
    try:
        # Replace single quotes with double quotes for JSON parsing
        json_str = complex_input.replace("'", '"')
        # Try to parse
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            # For each item, extract if it's not a long sequence
            for item in parsed:
                if isinstance(item, str):
                    # If it looks like a protein sequence (long and mostly alphabetical)
                    if len(item) > 50 and re.match(r'^[A-Za-z]+$', item):
                        continue  # Skip sequences
                    else:
                        extracted_ids.append(item)
        return extracted_ids
    except (json.JSONDecodeError, ValueError):
        # If JSON parsing fails, continue with regex approach
        pass
    
    # Extract anything that looks like a protein ID using regex
    if '[' in complex_input and ']' in complex_input:
        # This is likely an array or Series representation
        # Look for patterns like "12345  PROTEIN_NAME"
        matches = re.findall(r'(\d+\s+[A-Z0-9]+(?:\([^)]+\))?)', complex_input)
        for match in matches:
            parts = match.strip().split()
            if len(parts) >= 2:
                # Take the last part as the protein ID, typically format "12345    PROTEINID"
                protein_id = parts[-1]
                # Clean any remaining quotes or special chars
                protein_id = re.sub(r'[^\w\-\(\)]', '', protein_id)
                if protein_id and protein_id not in extracted_ids:
                    extracted_ids.append(protein_id)
    
    # If we didn't find any with the above methods, try a more aggressive approach
    if not extracted_ids:
        # Split by common separators and find potential protein IDs
        potential_ids = re.findall(r'\b[A-Z0-9]{3,10}\b', complex_input)
        for potential_id in potential_ids:
            # Ignore very common words and numbers
            if potential_id not in ['TARGET', 'NAME', 'TYPE', 'LENGTH', 'DTYPE', 'OBJECT'] and not potential_id.isdigit():
                extracted_ids.append(potential_id)
    
    # If still nothing found, use the input as is if it's short enough
    if not extracted_ids and len(complex_input) < 50:
        extracted_ids.append(complex_input)
    
    return extracted_ids

def process_and_create_graphs(input_identifier, output_dir="../data/protein_graphs", use_colabfold=False):
    """
    Process a complex input, extract protein IDs, and create graph files for each
    
    Args:
        input_identifier: Complex input that may contain multiple protein IDs
        output_dir: Directory to save protein graphs
        use_colabfold: Whether to use local ColabFold
    """
    print(f"\n{'='*80}")
    print(f"Processing input for protein IDs")
    print(f"{'='*80}")
    
    # Extract protein IDs
    print("Input:", input_identifier[:200] + "..." if len(str(input_identifier)) > 200 else input_identifier)
    protein_ids = extract_protein_ids(input_identifier)
    
    if not protein_ids:
        print("Could not extract any protein IDs from the input. Using the input directly.")
        protein_ids = [str(input_identifier)]
    
    print(f"Extracted {len(protein_ids)} protein IDs: {protein_ids}")
    
    # Create a graph for each protein ID
    builder = ProteinGraphBuilder(
        graph_dir=output_dir,
        cutoff=10.0,
        use_colabfold=use_colabfold
    )
    
    results = {"success": [], "failed": []}
    
    for protein_id in protein_ids:
        try:
            # Clean any unwanted characters
            protein_id = protein_id.strip()
            
            # Check if graph already exists
            expected_filename = builder._generate_protein_graph_filename(protein_id)
            expected_path = os.path.join(output_dir, expected_filename)
            
            if os.path.exists(expected_path):
                print(f"Protein graph already exists for: {protein_id} at {expected_path}")
                results["success"].append(protein_id)
                continue
                
            # Generate protein graph
            print(f"\nGenerating protein graph for '{protein_id}'...")
            try:
                pdb_path = builder._fetch_pdb(protein_id)
                protein_graph = builder.build(pdb_path)
                builder.save(protein_id, protein_graph)
                print(f"Successfully saved protein graph to {expected_path}")
                results["success"].append(protein_id)
            except Exception as e:
                print(f"Failed to generate graph using normal method: {e}")
                
                # Fallback to linear structure
                print("Attempting to create a linear structure...")
                placeholder_seq = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                pdb_path = builder._create_linear_structure(protein_id, placeholder_seq)
                protein_graph = builder.build(pdb_path)
                builder.save(protein_id, protein_graph)
                print(f"Successfully saved protein graph with placeholder sequence to {expected_path}")
                results["success"].append(protein_id)
        except Exception as e:
            print(f"Failed to process protein ID '{protein_id}': {e}")
            results["failed"].append(protein_id)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Successfully processed: {len(results['success'])}/{len(protein_ids)} proteins")
    if results["success"]:
        print(f"  Successful IDs: {results['success']}")
    if results["failed"]:
        print(f"  Failed IDs: {results['failed']}")
    print(f"{'='*80}")
    
    return results

def extract_from_notebook(notebook_file):
    """Extract protein IDs from error messages in a Jupyter notebook"""
    try:
        import json
        with open(notebook_file, 'r') as f:
            notebook = json.load(f)
        
        protein_ids = []
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code' and 'outputs' in cell:
                for output in cell['outputs']:
                    if 'text' in output:
                        text = ''.join(output['text'])
                        if 'Missing' in text and 'protein' in text:
                            # Extract IDs from error messages
                            matches = re.findall(r'seq-([a-f0-9]{32})\.pt', text)
                            if matches:
                                protein_ids.extend(matches)
                            # Also try to find protein names
                            names = re.findall(r'\b[A-Z0-9]{3,10}\b', text)
                            for name in names:
                                if name not in ['MISSING', 'ERROR', 'WARNING'] and not name.isdigit():
                                    protein_ids.append(name)
        
        # Deduplicate
        protein_ids = list(set(protein_ids))
        return protein_ids
    except Exception as e:
        print(f"Error extracting from notebook: {e}")
        return []

def extract_from_dataset_file(dataset_file):
    """Extract all protein IDs from a dataset file"""
    try:
        if dataset_file.endswith('.csv'):
            df = pd.read_csv(dataset_file)
        elif dataset_file.endswith('.tab'):
            df = pd.read_csv(dataset_file, sep='\t')
        else:
            print(f"Unsupported file format: {dataset_file}")
            return []
        
        # Look for Target ID column
        target_col = None
        for col in df.columns:
            if 'target' in col.lower() or 'protein' in col.lower():
                target_col = col
                break
        
        if target_col is None:
            print(f"Could not find a target/protein column in {dataset_file}")
            return []
        
        # Extract unique protein IDs
        protein_ids = df[target_col].unique()
        
        # Handle complex types like lists
        processed_ids = []
        for pid in protein_ids:
            if isinstance(pid, list):
                processed_ids.extend(pid)
            else:
                processed_ids.append(str(pid))
        
        return processed_ids
    except Exception as e:
        print(f"Error extracting from dataset file: {e}")
        return []

def batch_process_proteins(protein_ids, output_dir="../data/protein_graphs", batch_size=10, use_colabfold=False):
    """Process a large list of protein IDs in batches"""
    builder = ProteinGraphBuilder(
        graph_dir=output_dir,
        cutoff=10.0,
        use_colabfold=use_colabfold
    )
    
    results = {"success": [], "failed": []}
    total = len(protein_ids)
    
    for i in range(0, total, batch_size):
        batch = protein_ids[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch)} proteins)")
        
        for j, protein_id in enumerate(batch):
            try:
                protein_id = str(protein_id).strip()
                expected_filename = builder._generate_protein_graph_filename(protein_id)
                expected_path = os.path.join(output_dir, expected_filename)
                
                if os.path.exists(expected_path):
                    print(f"[{i+j+1}/{total}] Graph already exists for: {protein_id}")
                    results["success"].append(protein_id)
                    continue
                
                print(f"[{i+j+1}/{total}] Processing: {protein_id}")
                
                try:
                    pdb_path = builder._fetch_pdb(protein_id)
                    protein_graph = builder.build(pdb_path)
                    builder.save(protein_id, protein_graph)
                    print(f"  Success: normal method")
                    results["success"].append(protein_id)
                except Exception as e:
                    print(f"  Failed normal method: {e}")
                    print(f"  Trying linear structure...")
                    
                    try:
                        placeholder_seq = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                        pdb_path = builder._create_linear_structure(protein_id, placeholder_seq)
                        protein_graph = builder.build(pdb_path)
                        builder.save(protein_id, protein_graph)
                        print(f"  Success: linear structure")
                        results["success"].append(protein_id)
                    except Exception as e2:
                        print(f"  Failed linear structure: {e2}")
                        results["failed"].append(protein_id)
            except Exception as e:
                print(f"[{i+j+1}/{total}] Error processing {protein_id}: {e}")
                results["failed"].append(protein_id)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Batch processing summary:")
    print(f"  Successfully processed: {len(results['success'])}/{total} proteins")
    print(f"  Failed: {len(results['failed'])}/{total} proteins")
    if results["failed"]:
        print(f"  First 10 failed IDs: {results['failed'][:10]}")
    print(f"{'='*80}")
    
    return results

def save_problematic_id_to_file(problematic_id, filename="problematic_id.txt"):
    """Save the problematic ID to a file for reference"""
    with open(filename, "w") as f:
        f.write(str(problematic_id))
    print(f"Saved problematic ID to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract protein IDs and create protein graphs")
    parser.add_argument("--id", type=str, help="Complex input that may contain multiple protein IDs")
    parser.add_argument("--output-dir", default="../data/protein_graphs", help="Output directory for protein graphs")
    parser.add_argument("--use-colabfold", action="store_true", help="Try to use local ColabFold")
    parser.add_argument("--extract-from-notebook", type=str, help="Extract protein IDs from a Jupyter notebook")
    parser.add_argument("--extract-from-dataset", type=str, help="Extract all protein IDs from a dataset file")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing multiple proteins")
    parser.add_argument("--save-id-list", type=str, help="Save extracted IDs to a file instead of processing them")
    args = parser.parse_args()
    
    # Process based on input type
    if args.id:
        if os.path.exists(args.id):
            with open(args.id, "r") as f:
                input_identifier = f.read().strip()
        else:
            input_identifier = args.id
        
        process_and_create_graphs(input_identifier, args.output_dir, args.use_colabfold)
    
    elif args.extract_from_notebook:
        print(f"Extracting protein IDs from notebook: {args.extract_from_notebook}")
        protein_ids = extract_from_notebook(args.extract_from_notebook)
        
        if not protein_ids:
            print("No protein IDs found in the notebook.")
            sys.exit(1)
        
        print(f"Extracted {len(protein_ids)} protein IDs from notebook")
        
        if args.save_id_list:
            with open(args.save_id_list, 'w') as f:
                for pid in protein_ids:
                    f.write(f"{pid}\n")
            print(f"Saved {len(protein_ids)} protein IDs to {args.save_id_list}")
        else:
            batch_process_proteins(protein_ids, args.output_dir, args.batch_size, args.use_colabfold)
    
    elif args.extract_from_dataset:
        print(f"Extracting protein IDs from dataset: {args.extract_from_dataset}")
        protein_ids = extract_from_dataset_file(args.extract_from_dataset)
        
        if not protein_ids:
            print("No protein IDs found in the dataset.")
            sys.exit(1)
        
        print(f"Extracted {len(protein_ids)} protein IDs from dataset")
        
        if args.save_id_list:
            with open(args.save_id_list, 'w') as f:
                for pid in protein_ids:
                    f.write(f"{pid}\n")
            print(f"Saved {len(protein_ids)} protein IDs to {args.save_id_list}")
        else:
            batch_process_proteins(protein_ids, args.output_dir, args.batch_size, args.use_colabfold)
    
    else:
        parser.print_help() 