#!/usr/bin/env python3
"""
Quantum-Classical Hybrid Autoencoder for Medical Image Classification
Training script for Purdue Scholar Cluster
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane.qnn import TorchLayer
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import time
import os

print("="*80)
print("QUANTUM-CLASSICAL HYBRID AUTOENCODER")
print("="*80)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"PennyLane version: {qml.__version__}")
print("="*80 + "\n")

# ============================================================================
# CONFIGURATION
# ============================================================================
n_qubits = 6
n_layers = 6
latent_dim = 2**n_qubits  # 64
img_size = 224
batch_size = 64
n_epochs = 300
n_workers = 8  # Increased for cluster

print(f"Configuration:")
print(f"  Qubits: {n_qubits}")
print(f"  Quantum layers: {n_layers}")
print(f"  Latent dimension: {latent_dim}")
print(f"  Image size: {img_size}x{img_size}")
print(f"  Batch size: {batch_size}")
print(f"  Epochs: {n_epochs}")
print(f"  Workers: {n_workers}\n")

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")
from preprocess import train_dataset, test_dataset
from torch.utils.data import Subset, DataLoader

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}\n")

# Filter single-label samples
print("Filtering single-label samples...")
train_single_label_indices = []
pos_counts = torch.zeros(14, dtype=torch.float)

for i in range(len(train_dataset)):
    _, label = train_dataset[i]
    if isinstance(label, np.ndarray):
        lab = torch.from_numpy(label)
    elif isinstance(label, torch.Tensor):
        lab = label
    else:
        lab = torch.tensor(label)
    lab = lab.squeeze()
    
    if lab.ndim == 0:
        if int(lab.item()) is not None:
            if 0 <= int(lab.item()) < 14:
                train_single_label_indices.append(i)
                pos_counts[int(lab.item())] += 1
    else:
        if lab.sum() == 1:
            train_single_label_indices.append(i)
            pos_counts += lab.float()

train_subset = Subset(train_dataset, train_single_label_indices)

# Filter test set
test_single_label_indices = []
for i in range(len(test_dataset)):
    _, label = test_dataset[i]
    if label.sum() == 1:
        test_single_label_indices.append(i)

test_subset = Subset(test_dataset, test_single_label_indices)

# Calculate class weights
total_single_label_samples = len(train_single_label_indices)
eps = 1e-6
class_weight = total_single_label_samples / (pos_counts + eps)
class_weight = class_weight / class_weight.mean()

print(f"Filtered Training: {total_single_label_samples} single-label images")
print(f"Filtered Test: {len(test_single_label_indices)} single-label images\n")

# Create data loaders
train_loader = DataLoader(
    train_subset, 
    batch_size=batch_size, 
    shuffle=True, 
    drop_last=True, 
    num_workers=n_workers, 
    pin_memory=True, 
    persistent_workers=True
)

test_loader = DataLoader(
    test_subset, 
    batch_size=batch_size, 
    shuffle=False, 
    drop_last=False, 
    num_workers=n_workers, 
    pin_memory=True, 
    persistent_workers=True
)

print(f"Training batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}\n")

# ============================================================================
# CLASS BASIS MAPPING
# ============================================================================
label_map_bits = [
    "111100", "011100", "101100", "110100", "111000", "111110", "000011",
    "111101", "001100", "100011", "010100", "010011", "001011", "000111",
]

CLASS_BASIS_IDX = torch.tensor([int(b, 2) for b in label_map_bits], dtype=torch.long)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
print("Building model...")

class Encoder(nn.Module):
    def __init__(self, latent_dim=64, img_size=224):
        super().__init__()
        self.encoder = nn.Sequential(
            # 224 -> 112
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 112 -> 56
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 56 -> 28
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, latent_dim)
        )
        
    def forward(self, x): 
        return self.encoder(x)


# Quantum circuit
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def qnode(inputs, weights):
    x = inputs.reshape(inputs.shape[0], n_layers, n_qubits)
    for l in range(n_layers):
        qml.AngleEmbedding(x[..., l, :], wires=range(n_qubits), rotation='Y')
        qml.StronglyEntanglingLayers(weights[l:l+1], wires=range(n_qubits))
    return qml.probs(wires=range(n_qubits))


class QuantumHead(nn.Module):
    def __init__(self, n_layers, n_qubits, n_classes, latent_dim, class_basis_idx):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = nn.Linear(latent_dim, n_layers * n_qubits)
        nn.init.xavier_uniform_(self.encoder.weight, gain=0.01)
        nn.init.zeros_(self.encoder.bias)
        
        self.weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = TorchLayer(qnode, self.weight_shapes)
        self.register_buffer("class_idx", class_basis_idx.clone().detach().long())
    
    def forward(self, h):
        z = self.encoder(h)
        probs = self.qlayer(z)
        probs_cat = probs.index_select(dim=1, index=self.class_idx)
        log_probs = torch.log(probs_cat).clamp(min=-100)
        return log_probs


class HybridQML(nn.Module):
    def __init__(self, img_size, latent_dim, n_classes=14):
        super().__init__()
        self.enc = Encoder(latent_dim, img_size)
        self.qhead = QuantumHead(n_layers, n_qubits, n_classes, latent_dim, CLASS_BASIS_IDX)
    
    def forward(self, x):
        h = self.enc(x)
        out = self.qhead(h)
        return out


# ============================================================================
# TRAINING SETUP WITH IMPROVED OPTIMIZER
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

model = HybridQML(img_size=img_size, latent_dim=latent_dim, n_classes=14).to(device)

# IMPROVED OPTIMIZER - MORE NOISE FOR EXPLORATION
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=3e-4,              # 6x higher - adds noise through bigger steps
    weight_decay=1e-4,    # 10x higher - stronger regularization
    eps=1e-5,             # 100x higher - gradient noise
    betas=(0.85, 0.95),   # Lower beta1 - less smoothing = more noise
    amsgrad=True          # Better handling of noisy gradients
)

# Add scheduler for periodic LR changes (more noise)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=25,               # Restart every 25 epochs
    T_mult=2,
    eta_min=1e-6
)

# Label smoothing loss
class LabelSmoothingNLLLoss(nn.Module):
    def __init__(self, weight=None, smoothing=0.1, reduction='none'):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, log_probs, targets):
        n_classes = log_probs.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(true_dist * log_probs).sum(dim=-1)
        if self.weight is not None:
            loss = loss * self.weight[targets]
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

criterion = LabelSmoothingNLLLoss(
    weight=class_weight.to(device), 
    smoothing=0.1,
    reduction='none'
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Optimizer: AdamW (lr=3e-4, eps=1e-5, weight_decay=1e-4)")
print(f"Scheduler: CosineAnnealingWarmRestarts")
print(f"Loss: NLL with label smoothing (0.1)\n")

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_one_epoch(epoch):
    model.train()
    total, n = 0.0, 0
    start_time = time.time()
    sum_p_correct = 0.0
    sum_invalid = 0.0

    for i, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        targets = labels.argmax(dim=1).to(device).long()

        optimizer.zero_grad(set_to_none=True)

        log_probs = model(imgs)
        batch_loss = criterion(log_probs, targets)
        loss = batch_loss.mean()
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # Metrics
        p_correct = log_probs.exp()[torch.arange(len(targets), device=device), targets]
        sum_p_correct += p_correct.sum().item()
        invalid_prob = 1.0 - log_probs.exp().sum(dim=1)
        sum_invalid += invalid_prob.sum().item()
        total += batch_loss.sum().item()
        n += imgs.size(0)

    # Step scheduler
    scheduler.step()
    
    epoch_time = time.time() - start_time
    avg_loss = total / n
    train_accuracy = 100.0 * (sum_p_correct / n)
    invalid_occurrence = 100.0 * (sum_invalid / n)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch} | {epoch_time:.1f}s | LR: {current_lr:.2e} | "
          f"Train Loss: {avg_loss:.5f} | Train Acc: {train_accuracy:.2f}% | "
          f"Invalid: {invalid_occurrence:.2f}%")
    
    return avg_loss, epoch_time, train_accuracy


@torch.no_grad()
def evaluate():
    model.eval()
    total, n = 0.0, 0
    sum_p_correct = 0.0
    sum_invalid = 0.0

    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        targets = labels.argmax(dim=1).to(device).long()

        log_probs = model(imgs)
        batch_loss = criterion(log_probs, targets)

        p_correct = log_probs.exp()[torch.arange(len(targets), device=device), targets]
        sum_p_correct += p_correct.sum().item()
        invalid_prob = 1.0 - log_probs.exp().sum(dim=1)
        sum_invalid += invalid_prob.sum().item()
        total += batch_loss.sum().item()
        n += imgs.size(0)

    test_loss = total / n
    test_accuracy = 100.0 * (sum_p_correct / n)
    invalid_occurrence = 100.0 * (sum_invalid / n)
    return test_loss, test_accuracy, invalid_occurrence


# ============================================================================
# TRAINING LOOP
# ============================================================================
print("="*80)
print(f"Starting training for {n_epochs} epochs...")
print("="*80 + "\n")

# Create outputs directory
os.makedirs('outputs', exist_ok=True)

# Track results
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
epoch_times = []
best_test_loss = float('inf')
best_epoch = 0
best_model = None
epochs_since_improvement = 0
improvement_patience = 80

total_start = time.time()

for epoch in range(1, n_epochs + 1):
    print("="*80)
    train_loss, epoch_time, train_acc = train_one_epoch(epoch)
    test_loss, test_acc, invalid_occ = evaluate()
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    epoch_times.append(epoch_time)
    
    # Track best model
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_epoch = epoch
        epochs_since_improvement = 0
        best_model = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'invalid_occ': invalid_occ
        }
    else:
        epochs_since_improvement += 1
    
    print(f"Test Loss: {test_loss:.5f} | Best: {best_test_loss:.5f} (epoch {best_epoch}) | "
          f"Test Acc: {test_acc:.2f}% | Invalid: {invalid_occ:.2f}%")
    
    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        checkpoint_path = f'outputs/checkpoint_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    if epochs_since_improvement >= improvement_patience:
        print(f"\nEarly stopping at epoch {epoch} - no improvement for {improvement_patience} epochs")
        break

total_time = time.time() - total_start
print(f"\n{'='*80}")
print(f"Training complete in {total_time/60:.2f} minutes")
print(f"{'='*80}\n")

# ============================================================================
# SAVE RESULTS
# ============================================================================
# Save best model
if best_model is not None:
    torch.save(best_model, 'outputs/best_model.pt')
    print(f"Best model saved (epoch {best_model['epoch']}, "
          f"test loss: {best_model['test_loss']:.5f}, "
          f"test acc: {best_model['test_acc']:.2f}%)")

# Save final model
torch.save({
    'n_epochs': len(train_losses),
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_losses': train_losses,
    'test_losses': test_losses,
    'test_accuracies': test_accuracies,
    'epoch_times': epoch_times,
    'best_test_loss': best_test_loss,
    'best_epoch': best_epoch,
    'config': {
        'n_qubits': n_qubits,
        'n_layers': n_layers,
        'latent_dim': latent_dim,
        'img_size': img_size,
        'batch_size': batch_size,
    }
}, 'outputs/final_model_complete.pt')

print("Complete training history saved\n")

# ============================================================================
# PLOT RESULTS
# ============================================================================
print("Generating plots...")
plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(1, 2, 1)
epochs_range = range(1, len(train_losses) + 1)
plt.plot(epochs_range, train_losses, 'b', label='Train Loss', linewidth=2)
plt.plot(epochs_range, test_losses, 'r', label='Test Loss', linewidth=2)
plt.axhline(y=best_test_loss, color='g', linestyle='--', 
            label=f'Best Test ({best_test_loss:.4f})', linewidth=1)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Progress', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Time plot
plt.subplot(1, 2, 2)
plt.bar(epochs_range, epoch_times, color='steelblue', alpha=0.7)
plt.axhline(y=np.mean(epoch_times), color='r', linestyle='--', 
            label=f'Average ({np.mean(epoch_times):.1f}s)', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.title('Time per Epoch', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/training_results.png', dpi=150, bbox_inches='tight')
print("Training curves saved to outputs/training_results.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)
print(f"Total epochs run: {len(train_losses)}")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Average time per epoch: {np.mean(epoch_times):.1f} seconds")
print(f"\nLoss Progression:")
print(f"  Initial train loss: {train_losses[0]:.4f}")
print(f"  Final train loss: {train_losses[-1]:.4f}")
print(f"  Best test loss: {best_test_loss:.4f} (epoch {best_epoch})")
print(f"\nAccuracy Progression:")
print(f"  Initial test accuracy: {test_accuracies[0]:.2f}%")
print(f"  Final test accuracy: {test_accuracies[-1]:.2f}%")
print(f"  Best test accuracy: {max(test_accuracies):.2f}%")
print("="*80)
print("All results saved to outputs/ directory")
print("="*80)
