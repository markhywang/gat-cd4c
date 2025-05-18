#!/usr/bin/env python3
"""
Test script to verify ColabFold installation and functionality.
"""
import os
import sys
import subprocess
import tempfile
import pathlib
import time

# A very short test protein sequence for fast testing
TEST_SEQUENCE = "MVHLTPEEKSAVTALWGKV"  # Just 19 amino acids

def main():
    print("=== Testing ColabFold Installation ===")
    
    # Check if colabfold_batch is in PATH
    try:
        version_cmd = ["colabfold_batch", "--version"]
        result = subprocess.run(version_cmd, capture_output=True, text=True)
        print(f"ColabFold version: {result.stdout.strip()}")
    except Exception as e:
        print(f"Error running colabfold_batch: {e}")
        print("Make sure colabfold is installed and in your PATH")
        return 1
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        
        # Create FASTA file
        fasta_path = temp_path / "test.fasta"
        with open(fasta_path, 'w') as f:
            f.write(f">test\n{TEST_SEQUENCE}\n")
        
        # Set environment variables
        env = os.environ.copy()
        env["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF warnings
        
        # Run ColabFold with minimal settings
        cmd = [
            "colabfold_batch",
            "--num-recycle", "1",  # Use minimal recycles for faster testing
            "--model-type", "auto",
            "--msa-mode", "single_sequence",  # Skip MSA generation
            str(fasta_path),
            str(temp_path)
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        start_time = time.time()
        
        try:
            # Run the command with a timeout
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                    text=True, env=env)
            
            try:
                stdout, stderr = process.communicate(timeout=120)  # 2 minute timeout (shorter for small sequence)
                if process.returncode == 0:
                    print("ColabFold ran successfully!")
                    
                    # Check if PDB files were created
                    pdb_files = list(temp_path.glob("*.pdb"))
                    if pdb_files:
                        print(f"Generated {len(pdb_files)} PDB files:")
                        for pdb in pdb_files:
                            print(f" - {pdb.name}")
                        return 0
                    else:
                        print("ERROR: No PDB files were generated.")
                        return 1
                else:
                    print(f"ColabFold failed with return code {process.returncode}")
                    print(f"Error output: {stderr}")
                    return 1
                    
            except subprocess.TimeoutExpired:
                process.kill()
                print("ColabFold timed out after 2 minutes")
                return 1
                
        except Exception as e:
            print(f"Error running ColabFold: {e}")
            return 1
            
if __name__ == "__main__":
    sys.exit(main()) 