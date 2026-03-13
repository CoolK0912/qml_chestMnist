#!/bin/bash
# Setup script for creating the conda environment on Purdue Scholar Cluster

echo "========================================"
echo "Setting up Quantum ML Environment"
echo "========================================"

# Load anaconda module
module load anaconda

# Create conda environment
echo "Creating conda environment: quantum_ml"
conda create -n quantum_ml python=3.10 -y

# Activate environment
source activate quantum_ml

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 11.8..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install PennyLane and quantum computing packages
echo "Installing PennyLane..."
pip install pennylane pennylane-qiskit

# Install additional ML packages
echo "Installing additional packages..."
pip install numpy scipy matplotlib pandas scikit-learn
pip install tqdm pillow opencv-python

# Install medical imaging packages
echo "Installing medical imaging packages..."
pip install medmnist

# Install monitoring tools
pip install tensorboard wandb

# Verify installations
echo ""
echo "========================================"
echo "Verifying Installation"
echo "========================================"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import pennylane as qml; print(f'PennyLane: {qml.__version__}')"
python -c "import medmnist; print('MedMNIST: OK')"

echo ""
echo "========================================"
echo "Environment setup complete!"
echo "========================================"
echo "To activate: source activate quantum_ml"
echo "To deactivate: conda deactivate"
