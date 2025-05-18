#!/bin/bash

echo "==== ColabFold Environment Setup ===="
echo "This script will create a new conda environment with compatible versions for ColabFold"
echo "It addresses the 'jax.linear_util' AttributeError by installing compatible JAX versions"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Environment name
ENV_NAME="colabfold-fixed"

echo "Creating new conda environment '$ENV_NAME'..."
conda create -y -n $ENV_NAME python=3.9

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Installing compatible JAX versions..."
# Use specific, known-working JAX versions
pip install 'jax==0.3.17' 'jaxlib==0.3.15'

echo "Installing ColabFold..."
# Install a specific, known-working ColabFold version
pip install 'colabfold==1.5.2'

# Install additional dependencies that ColabFold needs
echo "Installing additional dependencies..."
pip install 'dm-haiku==0.0.9' 'dm-tree==0.1.8' 'tensorflow==2.11.0'
pip install torch torch-geometric pandas requests tqdm

echo ""
echo "===== Installation Complete ====="
echo "To use the fixed environment, run:"
echo "  conda activate $ENV_NAME"
echo "Then run your script:"
echo "  python src/utils/embed_proteins.py --dataset DAVIS --out-dir data/protein_graphs --use-local-colabfold --num-workers 1"
echo ""
echo "Note: For CPU-only usage, this may still be slow. Consider using the linear structure fallback if speed is important." 