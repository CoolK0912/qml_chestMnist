# AI Agent Instructions for QML ChestMNIST Project

This document guides AI coding assistants in understanding and working with this quantum-classical hybrid machine learning project.

## Project Overview

This is a hybrid quantum-classical machine learning model for chest X-ray classification using:
- Classical autoencoder for image feature extraction (224×224 → 64D)
- Quantum circuit (6 qubits) for classification
- Multi-label classification (14 chest pathologies)

## Critical Architecture Components

### Data Flow
1. Images → `preprocess.py` → DataLoaders
2. DataLoaders → Classical Encoder → 64D latent vector
3. Latent vector → Quantum Circuit → 14-class prediction

### Key Files
- `QML_Autoencoder.ipynb`: Main notebook with model definition and training
- `preprocess.py`: Data preprocessing and DataLoader creation
- `setup_kernel.py`: Environment setup script

## Important Conventions

### Quantum Circuit Design
- Uses 6 qubits with 2 entangling layers by default
- AmplitudeEmbedding for latent vector input
- StronglyEntanglingLayers for quantum processing
- PauliZ measurements on all qubits

### Safety Measures
```python
# Critical: Aggressive input validation in QuantumHead
z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)
z = torch.clamp(z, min=-5, max=5)
norm = torch.sqrt(torch.sum(z**2, dim=1, keepdim=True) + 1e-10)
```

### Training Stability
- Uses gradient clipping (max_norm=1.0)
- Small learning rate (5e-5)
- Conservative weight initialization
- Handles NaN/Inf values extensively

## Common Workflows

### Setup
1. Always run `setup_kernel.py` first for environment setup
2. Select "Python (QML Project)" kernel in notebooks
3. Configure parameters in notebook Cell 2 (n_qubits, img_size, etc.)

### Memory Management
- Reduce batch_size for memory issues (default: 32)
- Image size can be reduced (224→128/64)
- Number of qubits can be decreased (6→4/5)

### Adding Features
1. Modify `preprocess.py` for data changes
2. Update model architecture in notebook
3. Test with small batch first
4. Monitor for NaN values in quantum circuit

## Integration Points

### DataLoader Interface
```python
from preprocess import train_loader, test_loader, train_dataset, test_dataset
# Expects: (image, label) tuples
# image: torch.Tensor [1, H, W]
# label: torch.Tensor [14] (multi-hot)
```

### Model Output Format
```python
# Forward pass returns:
logits  # shape: [batch_size, 14]
# Apply sigmoid for probabilities
probs = torch.sigmoid(logits)
```

## Common Pitfalls

1. Never skip gradient clipping - critical for quantum circuit stability
2. Always validate quantum circuit inputs for NaN/Inf
3. Don't assume single-label data - dataset has multi-label samples
4. Check device placement (CPU/GPU) when modifying model

## Project-Specific Patterns

### Error Handling
```python
# Quantum circuit failure recovery pattern
try:
    expvals = qnode(single_input, self.q_weights)
except Exception as e:
    print(f"⚠️ Quantum circuit failed: {e}")
    expvals = torch.zeros(self.n_qubits)
```

### Input Normalization
Always normalize quantum inputs to unit vectors:
```python
norm = torch.sqrt(torch.sum(z**2, dim=1, keepdim=True) + 1e-10)
z_normalized = z / norm
```