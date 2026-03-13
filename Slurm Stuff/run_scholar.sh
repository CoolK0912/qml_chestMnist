#!/bin/bash
#SBATCH --job-name=qml_training
#SBATCH --account=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=qml_output_%j.log
#SBATCH --error=qml_error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "========================================================================"
echo "QML Autoencoder Training - Scholar GPU"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================================================"

module load anaconda

echo ""
echo "Installing required packages..."
pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install --user pennylane pennylane-lightning medmnist numpy

echo ""
echo "Environment Information:"
python3 --version
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python3 -c "import pennylane as qml; print('PennyLane:', qml.__version__)"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "========================================================================"
echo "Starting Training"
echo "========================================================================"

python3 qml_autoencoder.py

if [ $? -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed"
    exit 1
fi

echo ""
echo "End time: $(date)"
echo "Output files: best_model.pt, final_model_complete.pt"
echo "========================================================================"
