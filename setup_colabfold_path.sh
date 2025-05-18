#!/bin/bash

echo "========================================"
echo " ColabFold Complete Setup & Installation"
echo "========================================"
echo

# Function to detect shell
detect_shell() {
    if [ -n "$BASH_VERSION" ]; then
        echo "bash"
    elif [ -n "$ZSH_VERSION" ]; then
        echo "zsh"
    else
        echo "other"
    fi
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Environment name
ENV_NAME="colabfold-fixed"

# Remove existing environment if it exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Removing existing '$ENV_NAME' environment to ensure clean installation..."
    conda env remove -n $ENV_NAME -y
fi

echo "Creating new conda environment '$ENV_NAME'..."
conda create -y -n $ENV_NAME python=3.9 

echo "Installing pip packages directly in the environment..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install ColabFold with all dependencies properly
echo "Installing ColabFold with all dependencies..."
pip install "colabfold[alphafold-cpu]==1.5.2" 
pip install dm-haiku==0.0.9 dm-tree==0.1.8
pip install torch torch-geometric pandas requests tqdm

# Check if colabfold_batch is now available
if ! command -v colabfold_batch &> /dev/null; then
    echo "Error: colabfold_batch not found after installation."
    echo "Trying alternative installation method..."
    
    # Try direct installation of the specific package
    pip uninstall -y colabfold
    pip install "colabfold[alphafold-cpu]==1.5.2" --no-cache-dir
    
    if ! command -v colabfold_batch &> /dev/null; then
        echo "ERROR: Failed to install colabfold_batch command."
        echo "Please try installing using conda directly:"
        echo "  conda install -c conda-forge -c bioconda colabfold"
        exit 1
    fi
fi

echo
echo "ColabFold installed successfully at: $(which colabfold_batch)"

# Generate activation script
SHELL_TYPE=$(detect_shell)
ACTIVATE_SCRIPT="activate_colabfold.sh"

echo "#!/bin/bash" > $ACTIVATE_SCRIPT
echo "# This script activates the ColabFold environment" >> $ACTIVATE_SCRIPT
echo >> $ACTIVATE_SCRIPT

echo "# Source conda activation script" >> $ACTIVATE_SCRIPT
echo "eval \"\$(conda shell.bash hook)\"" >> $ACTIVATE_SCRIPT
echo "conda activate $ENV_NAME" >> $ACTIVATE_SCRIPT

echo >> $ACTIVATE_SCRIPT
echo "echo \"ColabFold environment activated. You can now run:\"" >> $ACTIVATE_SCRIPT
echo "echo \"  python src/utils/embed_proteins.py --dataset DAVIS --out-dir data/protein_graphs --use-local-colabfold --num-workers 1\"" >> $ACTIVATE_SCRIPT

chmod +x $ACTIVATE_SCRIPT

echo
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo
echo "To activate ColabFold before running your script, use:"
echo "  source $ACTIVATE_SCRIPT"
echo
echo "This will ensure that ColabFold is in your PATH and ready to use."
echo
echo "To test if ColabFold is working correctly:"
echo "  1. First activate the environment: source $ACTIVATE_SCRIPT"
echo "  2. Then run the test: python test_colabfold_install.py"
echo
echo "If you still have problems, try installing directly with conda:"
echo "  conda install -c conda-forge -c bioconda colabfold"
echo 