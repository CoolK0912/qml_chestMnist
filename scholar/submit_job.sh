#!/bin/bash
#SBATCH --job-name=quantum_xray
#SBATCH --output=logs/quantum_%j.out
#SBATCH --error=logs/quantum_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --account=scholar
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@purdue.edu

# Load required modules
module purge
module load anaconda
module load cuda/11.8
module load cudnn/8.6

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment (create it first if needed)
source activate quantum_ml

# Print Python and package versions
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__); print(f'CUDA available: {torch.cuda.is_available()}')"
echo "PennyLane version:"
python -c "import pennylane as qml; print(qml.__version__)"

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the training script
echo "Starting training..."
python train.py

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $?"
fi

echo "Job finished at: $(date)"
