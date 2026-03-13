"""
IMPROVED TRAINING SETUP WITH NOISE AND BETTER OPTIMIZATION

Replace the training setup section in your notebook with this code.
This adds multiple techniques to improve the terrible 8-9% accuracy.
"""

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = HybridQML(img_size=img_size, latent_dim=latent_dim, n_classes=14).to(device)

# ============================================================================
# IMPROVED OPTIMIZER CONFIGURATION
# ============================================================================

# Option A: Aggressive Learning with Noise (RECOMMENDED FIRST TRY)
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=3e-4,              # 6x higher than original (5e-5)
    weight_decay=1e-4,    # 10x higher for better regularization
    eps=1e-5,             # 100x higher epsilon adds gradient noise
    betas=(0.85, 0.95),   # Lower beta1 = less momentum smoothing = more noise
    amsgrad=True          # Better convergence with noisy gradients
)

# Add Cosine Annealing with Warm Restarts (periodic noise injection)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=25,               # Restart every 25 epochs
    T_mult=2,             # Double restart period each time
    eta_min=1e-6          # Minimum learning rate
)

# ============================================================================
# IMPROVED LOSS FUNCTION WITH LABEL SMOOTHING
# ============================================================================
# Label smoothing adds noise to targets, preventing overconfidence

class LabelSmoothingNLLLoss(nn.Module):
    def __init__(self, weight=None, smoothing=0.1, reduction='none'):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, log_probs, targets):
        n_classes = log_probs.size(-1)
        # Convert hard targets to soft targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Weighted cross entropy
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
    smoothing=0.1,  # 10% label smoothing
    reduction='none'
)

print(f"Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Optimizer: AdamW with lr={optimizer.param_groups[0]['lr']:.2e}, eps={optimizer.param_groups[0]['eps']:.2e}")
print(f"Scheduler: CosineAnnealingWarmRestarts")
print(f"Label Smoothing: 0.1\n")


# ============================================================================
# IMPROVED TRAINING LOOP WITH GRADIENT CLIPPING
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
        
        # IMPORTANT: Clip gradients to prevent explosion with higher LR
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # Training accuracy
        p_correct = log_probs.exp()[torch.arange(len(targets), device=device), targets]
        sum_p_correct += p_correct.sum().item()

        invalid_prob = 1.0-log_probs.exp().sum(dim=1)
        sum_invalid += invalid_prob.sum().item()

        total += batch_loss.sum().item()
        n += imgs.size(0)

    # Step the scheduler after each epoch
    scheduler.step()
    
    epoch_time = time.time() - start_time
    avg_loss = total / n
    train_accuracy = 100.0 * (sum_p_correct / n)
    invalid_occurrence = 100.0 * (sum_invalid / n)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch} | {epoch_time:.1f}s | LR: {current_lr:.2e} | "
          f"Train Loss: {avg_loss:.5f} | Train Acc: {train_accuracy:.2f}% | "
          f"Train Invalid: {invalid_occurrence:.2f}%")
    return avg_loss, epoch_time, train_accuracy


# ============================================================================
# ALTERNATIVE CONFIGURATIONS (comment/uncomment to try)
# ============================================================================

"""
# Option B: CYCLICAL LEARNING RATE (more aggressive noise)
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-4,
    weight_decay=1e-4,
    eps=1e-5,
    betas=(0.9, 0.95)
)

scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=1e-5,         # Min LR
    max_lr=1e-3,          # Max LR (periodic spikes add noise)
    step_size_up=10,
    mode='triangular2',
    cycle_momentum=True,
    base_momentum=0.8,
    max_momentum=0.95
)

# In training loop: scheduler.step() after each BATCH (not epoch)
"""

"""
# Option C: NUCLEAR OPTION (maximum exploration)
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-3,              # Very high LR
    weight_decay=5e-4,
    eps=1e-4,             # High epsilon noise
    betas=(0.8, 0.9),     # Low momentum
    amsgrad=True
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10,
    verbose=True
)

# In training loop: scheduler.step(test_loss) after evaluation
# And use: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
"""


print("\n" + "="*80)
print("READY TO TRAIN!")
print("="*80)
print("\nIf accuracy is still poor after these changes, the problem may be:")
print("1. Quantum circuit design (try more layers, different gates)")
print("2. Encoder architecture (might need more capacity)")
print("3. Data preprocessing (check image quality/normalization)")
print("4. Label mapping (the basis state encoding might be suboptimal)")
print("\nRun your training loop now with these improved settings!")
