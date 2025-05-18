#!/bin/bash
# This script activates the ColabFold conda environment

# Source conda activation script
eval "$(conda shell.bash hook)"
conda activate colabfold-conda

echo "ColabFold conda environment activated. You can now run:"
echo "  python src/utils/embed_proteins.py --dataset DAVIS --out-dir data/protein_graphs --use-local-colabfold --num-workers 1"
