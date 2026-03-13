# Quantum-Classical Hybrid Autoencoder for ChestMNIST Classification

A hybrid quantum-classical deep learning model for chest X-ray pathology classification using the ChestMNIST dataset. Combines a convolutional neural network encoder with a PennyLane variational quantum circuit.

## Architecture

```
Input (1×224×224)
       │
  CNN Encoder
  Conv2D: 224 → 112 → 56 → 28
       │
  Quantum Latent Space
  PennyLane circuit: 6 qubits, 6 layers (amplitude encoding)
       │
  QuantumHead Classifier
  Linear → sigmoid (14-class multi-label output)
```

- **Classical encoder**: 4-stage Conv2D feature pyramid, outputs a 6-dimensional quantum-compatible latent vector
- **Quantum circuit**: Amplitude-encoded variational circuit with trainable rotation gates (RX, RY, RZ) and CNOT entanglement
- **Classifier head**: Linear layer mapping quantum expectation values to 14 pathology labels

## Results

| Model | AUC (target) |
|-------|-------------|
| Baseline (random) | ~8–9% |
| Quantum-Classical Hybrid | >10–15% |

Training runs on Purdue Scholar GPU cluster (NVIDIA A100), ~6–8 hours per full run.

## Tech Stack

- **PyTorch** — CNN encoder and training loop
- **PennyLane** — variational quantum circuit (`default.qubit` / GPU backend)
- **MedMNIST** — ChestMNIST dataset (224×224 variant)
- **SLURM** — job scheduling on Purdue Scholar cluster
- **W&B** *(optional)* — experiment tracking

## Repository Structure

```
QML Project/
├── scholar/                  # Purdue Scholar GPU cluster deployment
│   ├── train.py              # Main training script (512 lines)
│   ├── preprocess.py         # Data loading & preprocessing
│   ├── submit_job.sh         # SLURM job submission script
│   ├── setup_env.sh          # Conda environment setup
│   ├── requirements.txt      # Python dependencies
│   ├── optimizer_experiments.py
│   ├── QUICK_FIX.py          # Optimizer tuning notes
│   ├── QUICKSTART.sh
│   └── README.md             # Scholar cluster deployment guide
├── Slurm Stuff/              # Additional SLURM utilities
├── requirements.txt          # Root-level Python dependencies (same as scholar/)
├── .gitignore
└── README.md
```

## Local Setup

```bash
# Clone the repo
git clone https://github.com/CoolK0912/qml_chestMnist.git
cd qml_chestMnist

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

ChestMNIST will be downloaded automatically via `medmnist` on first run.

## Training Locally

```bash
# From the scholar/ directory (or copy train.py to root)
python scholar/train.py
```

Training hyperparameters are configured at the top of `train.py`.

## Scholar Cluster Deployment

See **[`scholar/README.md`](scholar/README.md)** for full step-by-step instructions including:
- SSH setup, conda environment creation
- Data preparation options
- SLURM job submission and monitoring
- Troubleshooting common errors

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pennylane>=0.30.0
matplotlib>=3.7.0
timm>=0.9.0
huggingface-hub>=0.16.0
medvae>=0.1.0
pillow>=9.5.0
tqdm>=4.65.0
```

## License

MIT — see [LICENSE](LICENSE)
