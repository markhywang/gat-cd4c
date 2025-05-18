#!/bin/bash

echo "========================================"
echo " ColabFold GPU Installation for CUDA 12.8"
echo "========================================"
echo

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Check CUDA installation
if ! command -v nvcc &> /dev/null; then
    echo "Warning: nvcc not found. CUDA toolkit might not be properly installed."
    echo "Continuing anyway, but GPU acceleration might not work."
fi

# Print CUDA version
if command -v nvcc &> /dev/null; then
    echo "CUDA version: $(nvcc --version | grep 'release' | awk '{print $6}' | cut -c2-)"
fi

# Check if GPU is available
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi -L
    echo "GPU detected!"
else
    echo "Warning: No NVIDIA GPU detected or nvidia-smi not installed."
    echo "Continuing anyway, but GPU acceleration will not work."
fi

# Environment name
ENV_NAME="colabfold-gpu"

# Remove existing environment if it exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Removing existing '$ENV_NAME' environment to ensure clean installation..."
    conda env remove -n $ENV_NAME -y
fi

echo "Creating new conda environment '$ENV_NAME'..."
conda create -y -n $ENV_NAME python=3.12.2

echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install CUDA compatible packages
echo "Installing CUDA-compatible JAX..."
# For CUDA 12.8, we need to use newer JAX versions
pip install jax==0.4.28 jaxlib==0.4.28+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "Installing ColabFold..."
conda install -y -c conda-forge -c bioconda colabfold=1.5.5

# Install additional dependencies
echo "Installing additional dependencies..."
conda install -y -c conda-forge -c pytorch pytorch torchvision torchaudio pytorch-cuda=12.1
pip install torch-geometric pandas requests tqdm

# Check if colabfold_batch is available
if ! command -v colabfold_batch &> /dev/null; then
    echo "Error: colabfold_batch not found after installation."
    exit 1
fi

# Create a GPU environment script
ACTIVATE_SCRIPT="activate_colabfold_gpu.sh"

echo "#!/bin/bash" > $ACTIVATE_SCRIPT
echo "# This script activates the ColabFold GPU environment" >> $ACTIVATE_SCRIPT
echo >> $ACTIVATE_SCRIPT
echo "# Source conda activation script" >> $ACTIVATE_SCRIPT
echo "eval \"\$(conda shell.bash hook)\"" >> $ACTIVATE_SCRIPT
echo "conda activate $ENV_NAME" >> $ACTIVATE_SCRIPT
echo >> $ACTIVATE_SCRIPT
echo "# Set environment variables for optimal GPU performance" >> $ACTIVATE_SCRIPT
echo "export XLA_PYTHON_CLIENT_PREALLOCATE=false" >> $ACTIVATE_SCRIPT
echo "export XLA_PYTHON_CLIENT_ALLOCATOR=platform" >> $ACTIVATE_SCRIPT
echo "export TF_FORCE_UNIFIED_MEMORY=1" >> $ACTIVATE_SCRIPT
echo "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8" >> $ACTIVATE_SCRIPT
echo "export TF_GPU_THREAD_MODE=gpu_private" >> $ACTIVATE_SCRIPT
echo "export TF_ENABLE_ONEDNN_OPTS=0" >> $ACTIVATE_SCRIPT
echo >> $ACTIVATE_SCRIPT
echo "echo \"ColabFold GPU environment activated. You can now run:\"" >> $ACTIVATE_SCRIPT
echo "echo \"  python src/utils/embed_proteins.py --dataset DAVIS --out-dir data/protein_graphs --use-local-colabfold --num-workers 1\"" >> $ACTIVATE_SCRIPT

chmod +x $ACTIVATE_SCRIPT

echo
echo "ColabFold installed successfully at: $(which colabfold_batch)"
echo
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo
echo "To activate ColabFold with GPU support, use:"
echo "  source $ACTIVATE_SCRIPT"
echo
echo "To test if ColabFold is using the GPU correctly:"
echo "  1. Activate the environment: source $ACTIVATE_SCRIPT"
echo "  2. Run: python test_colabfold_gpu.py"
echo 
echo "Note: If you encounter GPU memory issues, you can adjust the memory fraction"
echo "      by modifying XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 in $ACTIVATE_SCRIPT" 