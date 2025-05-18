#!/usr/bin/env python3
"""
Test script to verify ColabFold is using GPU acceleration.
"""
import os
import sys
import subprocess
import tempfile
import pathlib
import time
import json

# A very short test protein sequence for fast testing
TEST_SEQUENCE = "MVHLTPEEKS"  # Just 10 amino acids

def check_jax_gpu():
    """Check if JAX can see the GPU."""
    try:
        import jax
        devices = jax.devices()
        for i, dev in enumerate(devices):
            print(f"JAX device {i}: {dev}")
            
        if any(d.platform == 'gpu' for d in devices):
            print("✓ JAX GPU acceleration available")
            return True
        else:
            print("✗ No JAX GPU devices found")
            return False
    except ImportError:
        print("✗ Could not import JAX")
        return False
    except Exception as e:
        print(f"✗ Error checking JAX GPU: {e}")
        return False

def check_torch_gpu():
    """Check if PyTorch can see the GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ PyTorch GPU acceleration available ({torch.cuda.get_device_name(0)})")
            print(f"  CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("✗ PyTorch GPU acceleration not available")
            return False
    except ImportError:
        print("✗ Could not import PyTorch")
        return False
    except Exception as e:
        print(f"✗ Error checking PyTorch GPU: {e}")
        return False

def main():
    print("=== Testing ColabFold GPU Acceleration ===")
    
    # Check environment variables
    gpu_env_vars = [
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "TF_FORCE_UNIFIED_MEMORY",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
    ]
    
    print("\nGPU Environment Variables:")
    for var in gpu_env_vars:
        val = os.environ.get(var, "Not set")
        print(f"  {var}: {val}")
    
    # Check if JAX and PyTorch can use the GPU
    print("\nChecking GPU availability:")
    jax_gpu = check_jax_gpu()
    torch_gpu = check_torch_gpu()
    
    # Check if colabfold_batch is in PATH
    try:
        version_cmd = ["colabfold_batch", "--version"]
        result = subprocess.run(version_cmd, capture_output=True, text=True)
        print(f"\nColabFold version: {result.stdout.strip()}")
    except Exception as e:
        print(f"\nError running colabfold_batch: {e}")
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
        
        # Run ColabFold with GPU settings
        cmd = [
            "colabfold_batch",
            "--num-recycle", "1",           # Use minimal recycles for faster testing
            "--model-type", "auto",
            "--msa-mode", "single_sequence", # Skip MSA generation
            str(fasta_path),
            str(temp_path)
        ]
        
        print(f"\nRunning command: {' '.join(cmd)}")
        start_time = time.time()
        
        try:
            # Run the command with a timeout
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                    text=True, env=env)
            
            try:
                stdout, stderr = process.communicate(timeout=120)
                end_time = time.time()
                elapsed = end_time - start_time
                
                if process.returncode == 0:
                    print(f"ColabFold ran successfully in {elapsed:.2f} seconds!")
                    
                    # Check if GPU was mentioned in output
                    gpu_used = False
                    for line in stdout.split('\n') + stderr.split('\n'):
                        if "gpu" in line.lower() or "cuda" in line.lower():
                            print(f"GPU reference found: {line.strip()}")
                            gpu_used = True
                            
                    # Check if logs mention GPU
                    log_files = list(temp_path.glob("*.log"))
                    for log_file in log_files:
                        with open(log_file, 'r') as f:
                            log_content = f.read()
                            if "gpu" in log_content.lower() or "cuda" in log_content.lower():
                                print(f"GPU reference found in log file {log_file.name}")
                                gpu_used = True
                    
                    # Check if PDB files were created
                    pdb_files = list(temp_path.glob("*.pdb"))
                    print(f"Generated {len(pdb_files)} PDB files")
                    
                    if gpu_used:
                        print("\n✓ GPU ACCELERATION DETECTED")
                    else:
                        print("\n? GPU ACCELERATION STATUS UNCERTAIN")
                        print("  ColabFold completed but no explicit GPU references were found.")
                        print("  If folding was very fast, GPU acceleration was likely working.")
                    
                    return 0
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