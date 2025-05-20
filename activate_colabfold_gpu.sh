#!/bin/bash
# This script activates the ColabFold GPU environment

# Source conda activation script
eval "$(conda shell.bash hook)"
conda activate colabfold-gpu

# Set environment variables for optimal GPU performance
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export TF_GPU_THREAD_MODE=gpu_private
export TF_ENABLE_ONEDNN_OPTS=0

echo "ColabFold GPU environment activated. You can now run:"
echo "  python src/utils/embed_proteins.py --dataset DAVIS --out-dir data/protein_graphs --use-local-colabfold --num-workers 1"
