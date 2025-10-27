# Quantum-Classical Hybrid Autoencoder for Medical Image Classification

This project implements a hybrid quantum-classical machine learning model for classifying chest X-ray images from the ChestMNIST dataset.

## Project Structure

```
QML Project/
├── QML_Autoencoder.ipynb    # Main notebook with hybrid model
├── preprocess.py             # Data loading and preprocessing
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── quantum_timeline_final_part1.png
├── quantum_timeline_final_part2.png
└── .venv/                    # Virtual environment
```

## What This Project Does

The model combines:
1. **Classical Autoencoder**: Compresses 224×224 chest X-ray images into 64-dimensional latent vectors
2. **Quantum Circuit**: Uses 6 qubits with entangling layers to classify the compressed features
3. **Multi-label Classification**: Predicts 14 different chest pathologies

### Architecture Flow

```
Input Image (224×224)
    ↓
Classical Encoder (CNN)
    ↓
Latent Vector (64D)
    ↓
Quantum Circuit (6 qubits)
    ↓
Classification (14 labels)
```

## Setup

### 1. Install Dependencies & Fix Kernel Issues

**IMPORTANT: If your kernel is failing to start, use this setup script:**

#### Option A: Automated Setup (Recommended)

Run the setup script:

```bash
cd "/Users/nadavyaniv/Projects/QML Project"
python3 setup_kernel.py
```

This will:
- Install all required packages in the `.venv`
- Register a Jupyter kernel called "Python (QML Project)"
- Make the kernel available to VS Code

#### Option B: Manual Installation

```bash
cd "/Users/nadavyaniv/Projects/QML Project"
source .venv/bin/activate  # On Mac/Linux
# or
.venv\Scripts\activate     # On Windows

# Install packages
pip install ipykernel jupyter torch torchvision numpy pennylane medmnist

# Register the kernel
python -m ipykernel install --user --name=qml-project --display-name="Python (QML Project)"

deactivate
```

#### Option C: Bash Script

```bash
cd "/Users/nadavyaniv/Projects/QML Project"
chmod +x install_kernel.sh
./install_kernel.sh
```

### 2. Running the Project

#### Option A: Using VS Code (Recommended)

1. Open `QML_Autoencoder.ipynb` in VS Code
2. **IMPORTANT**: Click the kernel selector in the top right
3. Select **"Python (QML Project)"** from the kernel list
   - If you don't see it, run `python3 setup_kernel.py` first
4. Run all cells (Run All button at the top)

#### Option B: Using Jupyter Notebook

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. Start Jupyter:
   ```bash
   jupyter notebook
   ```

3. Open `QML_Autoencoder.ipynb` in your browser

4. Run all cells in order (Cell → Run All)

## File Details

### `preprocess.py`

Handles data loading and preparation:
- Downloads ChestMNIST dataset
- Resizes images to 224×224
- Creates PyTorch DataLoaders
- **Exports**: `train_loader`, `test_loader`, `train_dataset`, `test_dataset`

Can be run standalone:
```bash
python preprocess.py
```

Or imported in the notebook:
```python
from preprocess import train_loader, test_loader, train_dataset, test_dataset
```

### `QML_Autoencoder.ipynb`

Main training notebook containing:
- Model architecture (Autoencoder + QuantumHead)
- Training loop
- Evaluation functions
- Imports data from `preprocess.py`

## Model Configuration

Default settings (in notebook Cell 2):
- **Qubits**: 6
- **Quantum layers**: 2
- **Latent dimension**: 64 (2^6)
- **Image size**: 224×224
- **Batch size**: 32
- **Training epochs**: 5
- **Optimizer**: Adam (lr=2e-3, weight_decay=1e-5)
- **Loss function**: BCEWithLogitsLoss (multi-label classification)

## Dataset

**ChestMNIST** from the MedMNIST collection:
- Training samples: ~78,000
- Test samples: ~22,000
- Image format: Grayscale chest X-rays
- Labels: 14 pathology classes (multi-label)
- Classes include: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, etc.

## Training

The model trains for 5 epochs by default. Each epoch:
1. Forward pass through autoencoder + quantum circuit
2. Calculate classification loss
3. Backpropagate and update weights
4. Evaluate on test set

Expected output:
```
Epoch 1: train loss = 0.XXXX
Test BCEWithLogits loss = 0.XXXX

Epoch 2: train loss = 0.XXXX
...
```

## Troubleshooting

### Kernel Fails to Start ⚠️

**This is the most common issue!** If the kernel won't start in VS Code:

1. **Run the setup script**:
   ```bash
   cd "/Users/nadavyaniv/Projects/QML Project"
   python3 setup_kernel.py
   ```

2. **Restart VS Code** completely (Quit and reopen)

3. **Select the correct kernel**:
   - Open `QML_Autoencoder.ipynb`
   - Click the kernel selector (top right)
   - Choose **"Python (QML Project)"**

4. **Still not working?** Try this:
   - Open VS Code Command Palette (`Cmd+Shift+P`)
   - Type "Python: Select Interpreter"
   - Choose the one at: `/Users/nadavyaniv/Projects/QML Project/.venv/bin/python`

### Import Errors

If you get `ModuleNotFoundError`:
1. Make sure you ran `setup_kernel.py` first
2. Check you selected "Python (QML Project)" kernel
3. Try restarting the kernel (Restart button in notebook toolbar)

### CUDA/GPU Issues

The model automatically detects GPU availability. If you want to force CPU:
```python
device = torch.device("cpu")
```

### Memory Issues

If you run out of memory:
- Reduce batch_size in Cell 2 (try 16 or 8)
- Reduce img_size (try 128 or 64)
- Reduce n_qubits (try 4 or 5)

## How Files Work Together

```
preprocess.py
    ↓ (creates data loaders)
    ↓
QML_Autoencoder.ipynb
    ↓ (imports: from preprocess import train_loader, test_loader)
    ↓
    Defines Models (Autoencoder, QuantumHead, HybridQML)
    ↓
    Trains on the data
    ↓
    Outputs: Trained model + loss metrics
```

## Results

After training, the model outputs:
- Training loss per epoch
- Test loss per epoch
- A trained model that can classify chest X-rays using quantum-enhanced features

## Next Steps

- Experiment with different quantum circuit architectures
- Try different numbers of qubits and layers
- Implement reconstruction loss for the autoencoder
- Add model saving/loading
- Visualize latent space representations
- Compare with classical-only baseline

## References

- **PennyLane**: https://pennylane.ai/
- **MedMNIST**: https://medmnist.com/
- **PyTorch**: https://pytorch.org/

---

**Last Updated**: October 16, 2025
**Python Version**: 3.13.7
**Main Dependencies**: PyTorch 2.9.0, PennyLane 0.43.0
