#!/bin/bash

echo "========================================"
echo " ColabFold Installation using Conda"
echo "========================================"
echo

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Environment name
ENV_NAME="colabfold-conda"

# Remove existing environment if it exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Removing existing '$ENV_NAME' environment to ensure clean installation..."
    conda env remove -n $ENV_NAME -y
fi

echo "Creating new conda environment '$ENV_NAME'..."
conda create -y -n $ENV_NAME python=3.9

echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "Installing ColabFold from conda-forge and bioconda channels..."
conda install -y -c conda-forge -c bioconda colabfold python=3.9

# Check if colabfold_batch is available
if ! command -v colabfold_batch &> /dev/null; then
    echo "Error: colabfold_batch not found after installation"
    exit 1
fi

echo "Installing additional dependencies for PyTorch serialization..."
conda install -y -c conda-forge -c pytorch pytorch torchvision
pip install torch-geometric pandas requests tqdm

# Generate activation script
ACTIVATE_SCRIPT="activate_colabfold_conda.sh"

echo "#!/bin/bash" > $ACTIVATE_SCRIPT
echo "# This script activates the ColabFold conda environment" >> $ACTIVATE_SCRIPT
echo >> $ACTIVATE_SCRIPT
echo "# Source conda activation script" >> $ACTIVATE_SCRIPT
echo "eval \"\$(conda shell.bash hook)\"" >> $ACTIVATE_SCRIPT
echo "conda activate $ENV_NAME" >> $ACTIVATE_SCRIPT
echo >> $ACTIVATE_SCRIPT
echo "echo \"ColabFold conda environment activated. You can now run:\"" >> $ACTIVATE_SCRIPT
echo "echo \"  python src/utils/embed_proteins.py --dataset DAVIS --out-dir data/protein_graphs --use-local-colabfold --num-workers 1\"" >> $ACTIVATE_SCRIPT

chmod +x $ACTIVATE_SCRIPT

echo
echo "ColabFold installed successfully at: $(which colabfold_batch)"
echo
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo
echo "To activate ColabFold before running your script, use:"
echo "  source $ACTIVATE_SCRIPT"
echo
echo "To test if ColabFold is working correctly:"
echo "  1. Activate the environment: source $ACTIVATE_SCRIPT"
echo "  2. Run: python test_colabfold_install.py"
echo
echo "Note: This installation uses conda packages which should be more stable"
echo "      than pip installations for ColabFold's complex dependencies." 