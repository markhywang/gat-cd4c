#!/bin/bash

# This script creates a dedicated conda environment for ColabFold with compatible package versions
# To run: bash colabfold_env_setup.sh

echo "Creating colabfold-env conda environment..."
conda create -y -n colabfold-env python=3.9

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate colabfold-env

echo "Installing JAX with compatible version..."
# Install JAX first with a compatible version
pip install "jax==0.4.10" "jaxlib==0.4.10"

echo "Installing ColabFold..."
pip install "colabfold==1.5.2" alphafold-colabfold

echo "Installing other required packages..."
pip install torch torch-geometric tqdm pandas requests 

echo ""
echo "======================================"
echo "Environment setup complete!"
echo ""
echo "To use this environment:"
echo "  1. Run: conda activate colabfold-env"
echo "  2. Run your protein embedding script:"
echo "     python src/utils/embed_proteins.py --dataset DAVIS --out-dir data/protein_graphs --use-local-colabfold"
echo ""
echo "This environment uses:"
echo "  - JAX 0.4.10 (older version, compatible with ColabFold)"
echo "  - ColabFold 1.5.2"
echo "======================================" 