# Quantum-Classical Hybrid Autoencoder for Medical Image Classification

## Setup Instructions for Purdue Scholar Cluster

### 1. Initial Setup

First, log into the Purdue Scholar cluster:
```bash
ssh yournetid@scholar.rcac.purdue.edu
```

Clone or upload your project files to your home directory or scratch space:
```bash
cd $HOME
# or use scratch for large datasets
cd $RCAC_SCRATCH
mkdir quantum_xray
cd quantum_xray
```

Upload all the files to this directory.

### 2. Create Conda Environment

Make the setup script executable and run it:
```bash
chmod +x setup_env.sh
./setup_env.sh
```

This will create a conda environment named `quantum_ml` with all required packages.

### 3. Prepare Your Data

**IMPORTANT**: You need to modify `preprocess.py` to point to your actual data location.

Option A - If you have your own dataset:
```python
# In preprocess.py, modify:
DATA_ROOT = "/path/to/your/data"  # Change this to your data path
```

Option B - Use MedMNIST for testing (ChestMNIST):
```bash
# The preprocess.py will automatically download MedMNIST if available
# No changes needed
```

Option C - Custom data structure:
- Modify the `ChestXrayDataset` class in `preprocess.py`
- Update the `__init__` and `__getitem__` methods based on your data format

### 4. Test Your Setup (Optional)

Before submitting a full job, test that everything works:
```bash
# Request an interactive session with GPU
srun -A scholar --partition=gpu --gres=gpu:1 --time=00:30:00 --pty bash

# Activate environment
module load anaconda
source activate quantum_ml

# Test the data loading
python preprocess.py

# Quick training test (reduce epochs in train.py first)
# Edit train.py: n_epochs = 2  # just for testing
python train.py

# Exit interactive session when done
exit
```

### 5. Submit Training Job

Create logs directory:
```bash
mkdir -p logs
```

Update your email in `submit_job.sh`:
```bash
# Edit line:
#SBATCH --mail-user=your_email@purdue.edu
```

Submit the job:
```bash
sbatch submit_job.sh
```

### 6. Monitor Your Job

Check job status:
```bash
squeue -u $USER
```

View output in real-time:
```bash
tail -f logs/quantum_*.out
```

View errors (if any):
```bash
tail -f logs/quantum_*.err
```

Cancel a job:
```bash
scancel <job_id>
```

### 7. Retrieve Results

After training completes, your results will be in the `outputs/` directory:
- `best_model.pt` - Best model checkpoint
- `final_model_complete.pt` - Final model with full training history
- `training_results.png` - Training curves plot
- `checkpoint_epoch_*.pt` - Periodic checkpoints

Download results to your local machine:
```bash
# From your local machine:
scp yournetid@scholar.rcac.purdue.edu:/path/to/quantum_xray/outputs/* .
```

## File Descriptions

- `train.py` - Main training script with improved optimizer
- `preprocess.py` - Data loading and preprocessing (CUSTOMIZE THIS!)
- `submit_job.sh` - SLURM job submission script
- `setup_env.sh` - Environment setup script
- `requirements.txt` - Python package requirements
- `QUICK_FIX.py` - Optimizer improvements documentation
- `improved_training_setup.py` - Alternative training configurations
- `optimizer_experiments.py` - Different optimizer strategies

## Resource Allocation

Current job settings (in `submit_job.sh`):
- **GPUs**: 1 GPU
- **CPUs**: 8 cores
- **Memory**: 64 GB
- **Time**: 24 hours
- **Partition**: gpu

Adjust these based on your needs and allocation.

## Troubleshooting

### Out of Memory (OOM) Errors
```bash
# Reduce batch size in train.py:
batch_size = 32  # or 16
```

### Slow Data Loading
```bash
# Increase number of workers (if you have more CPUs):
n_workers = 16  # match your CPU allocation
```

### Job Killed (Time Limit)
```bash
# In submit_job.sh, increase time:
#SBATCH --time=48:00:00  # 48 hours
```

### Module Load Errors
```bash
# Check available modules:
module avail

# Try different CUDA versions:
module load cuda/11.7
# or
module load cuda/12.1
```

### Data Not Found
```bash
# Make sure preprocess.py points to correct path
# Check data exists:
ls -la /path/to/your/data
```

## Advanced Usage

### Using Multiple GPUs
Modify `submit_job.sh`:
```bash
#SBATCH --gres=gpu:2  # Request 2 GPUs
```

And wrap your model in train.py:
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

### Resuming from Checkpoint
In `train.py`, add before training loop:
```python
# Load checkpoint
checkpoint = torch.load('outputs/checkpoint_epoch_100.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Experiment Tracking with W&B
```bash
# Install wandb
pip install wandb

# Login (one time)
wandb login

# In train.py, add:
import wandb
wandb.init(project="quantum-xray", config={...})
# Log metrics: wandb.log({"train_loss": loss, ...})
```

## Optimizer Settings

The improved optimizer settings add noise for better exploration:

- **Learning rate**: 3e-4 (6x higher than original)
- **Weight decay**: 1e-4 (10x higher)
- **Epsilon**: 1e-5 (100x higher - adds gradient noise)
- **Beta1**: 0.85 (reduced momentum for more noise)
- **AMSGrad**: Enabled
- **Scheduler**: Cosine annealing with warm restarts

See `QUICK_FIX.py` for detailed explanations and alternative configurations.

## Expected Training Time

On a single NVIDIA A100 GPU:
- ~75 seconds per epoch
- ~300 epochs (with early stopping)
- Total: ~6-8 hours

## Getting Help

- Scholar cluster documentation: https://www.rcac.purdue.edu/knowledge/scholar
- Email RCAC support: scholar-help@purdue.edu
- Check queue status: https://www.rcac.purdue.edu/compute/scholar

## Citation

If you use this code, please cite the relevant papers and acknowledge Purdue RCAC resources.
