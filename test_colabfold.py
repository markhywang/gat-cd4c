#!/usr/bin/env python3
"""
ColabFold Test Script

This script tests if ColabFold is correctly installed and functioning by
predicting the structure of a small test protein.
"""

import os
import sys
import time
import argparse
import subprocess
import tempfile
from pathlib import Path

TEST_SEQUENCE = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"  # GB1 domain - small, fast to fold

def test_colabfold_installation():
    """Test if ColabFold is correctly installed and working."""
    print("Testing ColabFold installation...")
    
    # Check if colabfold_batch exists in PATH
    try:
        version_output = subprocess.check_output(["colabfold_batch", "--version"], 
                                               stderr=subprocess.STDOUT,
                                               text=True)
        print(f"ColabFold found: {version_output.strip()}")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("ERROR: colabfold_batch command not found in PATH.")
        print("\nTo install ColabFold, follow these instructions:")
        print("1. Install with pip: pip install colabfold")
        print("2. Or install with conda: conda install -c conda-forge -c bioconda colabfold")
        return False
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a test FASTA file
        fasta_path = temp_path / "test.fasta"
        with open(fasta_path, 'w') as f:
            f.write(">test_protein\n")
            f.write(f"{TEST_SEQUENCE}\n")
        
        print(f"Created test sequence in {fasta_path}")
        
        # Run ColabFold with minimal settings and CPU-only
        cmd = [
            "colabfold_batch",
            "--num-recycle", "1",  # Minimal recycles for speed
            "--model-type", "auto",
            "--use-gpu-relax=False",
            "--templates=False",
            "--cpu-only",
            "--db-preset=reduced_dbs",
            str(fasta_path),
            str(temp_path)
        ]
        
        print(f"Running test with command: {' '.join(cmd)}")
        print("This may take a few minutes...")
        
        start_time = time.time()
        try:
            # Run with timeout of 10 minutes
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Monitor the process
            while process.poll() is None:
                # Check if we've been running too long
                if time.time() - start_time > 600:  # 10 minutes
                    process.terminate()
                    print("ERROR: ColabFold process timed out after 10 minutes")
                    return False
                
                # Print some output
                output = process.stdout.readline()
                if output:
                    print(f"ColabFold: {output.strip()}")
                
                time.sleep(0.1)
            
            # Get final output
            stdout, stderr = process.communicate()
            
            # Check if any PDB files were generated
            pdb_files = list(temp_path.glob("*.pdb"))
            
            if pdb_files:
                print(f"SUCCESS! ColabFold generated {len(pdb_files)} PDB file(s)")
                print(f"Test completed in {time.time() - start_time:.1f} seconds")
                return True
            else:
                print("ERROR: ColabFold did not generate any PDB files")
                if stderr:
                    print(f"Error output: {stderr}")
                return False
                
        except Exception as e:
            print(f"ERROR: ColabFold execution failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Test ColabFold installation")
    args = parser.parse_args()
    
    success = test_colabfold_installation()
    
    if success:
        print("\n✅ ColabFold is correctly installed and working!\n")
        sys.exit(0)
    else:
        print("\n❌ ColabFold test failed. Please check the installation.\n")
        sys.exit(1)

if __name__ == "__main__":
    main() 