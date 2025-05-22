#!/usr/bin/env python3
import argparse
import hashlib
import os
import pathlib
import re
import pandas as pd
from tdc.multi_pred import DTI # Assuming this is how datasets were loaded

# --- Filename Generation Logic (copied from embed_proteins.py) ---
def _generate_protein_graph_filename(identifier: str) -> str:
    """
    Generates a safe filename for a protein graph.
    Hashes identifiers that are too long or contain disallowed characters.
    Must be identical to the one used when graphs were originally created.
    """
    MAX_IDENTIFIER_LEN_BEFORE_HASH = 100
    ALLOWED_CHARS_REGEX = r"^[A-Za-z0-9_\-\.]+$"

    is_too_long = len(identifier) > MAX_IDENTIFIER_LEN_BEFORE_HASH
    has_disallowed_chars = not re.match(ALLOWED_CHARS_REGEX, identifier)

    if is_too_long or has_disallowed_chars:
        hashed_identifier = hashlib.md5(identifier.encode()).hexdigest()
        return f"seq-{hashed_identifier}.pt"
    else:
        return f"{identifier}.pt"

def get_dataset_target_mappings(dataset_name: str, data_root_path: str = "../data"):
    """
    Fetches unique target identifiers from the specified dataset (DAVIS or KIBA)
    and processes them to get the original (likely hashed) and new simple identifiers.

    Returns:
        List of tuples: (original_tdc_target_value, new_simple_identifier_str)
    """
    print(f"Loading raw {dataset_name} dataset from TDC...")
    # Path for TDC to download/cache raw data, distinct from graph_dir
    tdc_data_path = pathlib.Path(data_root_path) / "tdc_raw_data"
    tdc_data_path.mkdir(parents=True, exist_ok=True)

    try:
        data = DTI(name=dataset_name.upper(), path=str(tdc_data_path))
        df = data.get_data()
    except Exception as e:
        print(f"ERROR: Could not load dataset {dataset_name} using TDC: {e}")
        return []

    raw_unique_targets = df["Target"].unique()
    mappings = []
    processed_simple_names_for_uniqueness_check = []

    for original_target_val in raw_unique_targets:
        new_simple_id = ""
        if isinstance(original_target_val, list) and len(original_target_val) > 0:
            new_simple_id = str(original_target_val[0])
        else:
            new_simple_id = str(original_target_val)
        
        mappings.append((original_target_val, new_simple_id))
        processed_simple_names_for_uniqueness_check.append(new_simple_id)
    
    if len(processed_simple_names_for_uniqueness_check) != len(set(processed_simple_names_for_uniqueness_check)):
        print(f"WARNING: Potential simple name collisions detected for {dataset_name}.")
        # Further analysis could be added here to show which names collide.
        # For example, count occurrences of each simple name.

    print(f"Found {len(raw_unique_targets)} unique Target entries in TDC's {dataset_name}, resulting in {len(set(processed_simple_names_for_uniqueness_check))} unique simple IDs.")
    
    # Debug: Print first 5 mappings
    print("\n--- First 5 Mappings (Original TDC Target -> New Simple ID) ---")
    for i, (orig, simple) in enumerate(mappings[:5]):
        print(f"{i+1}. Original: {orig} (type: {type(orig)}) -> Simple: {simple} (type: {type(simple)})")
    print("--- End of First 5 Mappings ---\n")

    return mappings

def main():
    parser = argparse.ArgumentParser(description="Rename hashed protein graph files to simple names.")
    parser.add_argument("--dataset", choices=["KIBA", "DAVIS"], required=True,
                        help="Dataset whose graph files to rename (KIBA or DAVIS).")
    parser.add_argument("--graph-dir", default="../data/protein_graphs",
                        help="Directory where protein graphs are stored.")
    parser.add_argument("--data-root", default="../data",
                        help="Root directory for TDC to download raw data if needed.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be renamed without actual renaming.")

    args = parser.parse_args()

    graph_dir = pathlib.Path(args.graph_dir)
    if not graph_dir.is_dir():
        print(f"ERROR: Graph directory not found: {graph_dir}")
        return

    target_mappings = get_dataset_target_mappings(args.dataset, args.data_root)
    if not target_mappings:
        print("No target mappings found. Exiting.")
        return

    renamed_count = 0
    skipped_dst_exists = 0
    skipped_src_missing = 0
    skipped_collision = 0
    
    # To detect if a simple name has already been generated, preventing overwrites
    # from different original_ids that map to the same simple_id
    created_simple_filenames = set()

    print(f"\n--- Starting Renaming Process ({'DRY RUN' if args.dry_run else 'ACTUAL RUN'}) ---")
    for original_tdc_target, new_simple_id in target_mappings:
        # This is the crucial assumption: the string representation of the original TDC Target value
        # (which could be a list or a string) was used to generate the hash.
        original_id_for_hashing = str(original_tdc_target) 
        
        hashed_filename = _generate_protein_graph_filename(original_id_for_hashing)
        simple_target_filename = _generate_protein_graph_filename(new_simple_id) # Should not hash if new_simple_id is clean

        src_path = graph_dir / hashed_filename
        dst_path = graph_dir / simple_target_filename

        if src_path == dst_path: # Already correctly named or simple name was not hashable
            # This case implies the original identifier was already simple and valid
            # OR new_simple_id is identical to original_id_for_hashing and both don't get hashed.
            if src_path.exists():
                 print(f"INFO: File '{src_path.name}' seems already correctly named. Skipping.")
            continue

        if src_path.exists():
            if dst_path.exists():
                print(f"SKIP: Target '{dst_path.name}' already exists. Source: '{src_path.name}'")
                skipped_dst_exists += 1
                created_simple_filenames.add(dst_path.name) # Mark as created
            else:
                # Collision check: has this simple filename been created from a *different* hash?
                if dst_path.name in created_simple_filenames:
                    print(f"COLLISION_SKIP: Target '{dst_path.name}' was already created from a different source hash. Current source: '{src_path.name}'")
                    skipped_collision +=1
                else:
                    action_prefix = "[DRY RUN] Would rename" if args.dry_run else "RENAMING"
                    print(f"{action_prefix}: '{src_path.name}' -> '{dst_path.name}' (Original TDC Target: {original_tdc_target}, Simple ID: {new_simple_id})")
                    if not args.dry_run:
                        try:
                            os.rename(src_path, dst_path)
                            renamed_count += 1
                            created_simple_filenames.add(dst_path.name)
                        except OSError as e:
                            print(f"ERROR renaming '{src_path.name}': {e}")
        else:
            # This means the expected hashed file for this original_tdc_target was not found.
            # Could be due to:
            # 1. embed_proteins.py skipped it for some reason (e.g. failed PDB fetch).
            # 2. The assumption of `str(original_tdc_target)` for hashing is incorrect for this entry.
            # 3. The file was already renamed manually or by a previous run if it also fit a simple name.
            print(f"INFO: Source hashed file '{src_path.name}' not found for original TDC target '{original_tdc_target}'. Skipping.")
            skipped_src_missing += 1

    print("\n--- Renaming Summary ---")
    print(f"Files renamed: {renamed_count}")
    print(f"Skipped (destination already exists): {skipped_dst_exists}")
    print(f"Skipped (source hashed file not found): {skipped_src_missing}")
    print(f"Skipped (simple name collision from different hash): {skipped_collision}")
    if args.dry_run:
        print("NOTE: This was a dry run. No files were actually changed.")

if __name__ == "__main__":
    main() 