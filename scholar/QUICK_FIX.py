"""
QUICK FIX: EXACT CODE TO REPLACE IN YOUR NOTEBOOK
==================================================

Your current 8-9% accuracy is barely better than random guessing (7.14% for 14 classes).

STEP 1: Replace your optimizer initialization
----------------------------------------------
"""

# DELETE THIS (lines around your current optimizer setup):
"""
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=5e-5,
    weight_decay=1e-5,
    eps=1e-7
)

criterion = nn.NLLLoss(weight=class_weight.to(device), reduction='none')
"""

# REPLACE WITH THIS:
import torch
import torch.nn as nn

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=3e-4,              # 6x HIGHER (adds noise through bigger steps)
    weight_decay=1e-4,    # 10x HIGHER (stronger regularization)
    eps=1e-5,             # 100x HIGHER (gradient noise)
    betas=(0.85, 0.95),   # LOWER beta1 (less smoothing = more noise)
    amsgrad=True          # Better handling of noisy gradients
)

# Add scheduler for periodic LR changes (more noise)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=25,               # Restart every 25 epochs
    T_mult=2,
    eta_min=1e-6
)

# Better loss with label smoothing
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


"""
STEP 2: Modify your training loop
----------------------------------
"""

# ADD these two lines in train_one_epoch(), after loss.backward():
        # loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ADD THIS
        # optimizer.step()

# ADD this line at the END of train_one_epoch(), before return:
        # ... all your existing code ...
    scheduler.step()  # ADD THIS (after the for loop)
    
    epoch_time = time.time() - start_time
    # ... rest of function


"""
STEP 3: Monitor learning rate
------------------------------
"""

# MODIFY the print statement in train_one_epoch() to show LR:
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch} | {epoch_time:.1f}s | LR: {current_lr:.2e} | "
          f"Train Loss: {avg_loss:.5f} | Train Acc: {train_accuracy:.2f}% | "
          f"Train Invalid: {invalid_occurrence:.2f}%")


"""
==================================================
SUMMARY OF CHANGES:
==================================================

1. Learning rate: 5e-5 → 3e-4 (6x increase)
   - Bigger steps help escape local minima
   
2. Weight decay: 1e-5 → 1e-4 (10x increase)
   - Stronger regularization prevents overfitting
   
3. Epsilon: 1e-7 → 1e-5 (100x increase)
   - Adds noise to gradient calculations
   
4. Beta1: 0.9 → 0.85 (decrease)
   - Less momentum smoothing = more responsive to changes
   
5. AMSGrad: False → True
   - Better convergence with noisy gradients
   
6. Added scheduler: CosineAnnealingWarmRestarts
   - Learning rate periodically spikes and decays
   - Helps escape local minima through LR noise
   
7. Added gradient clipping: max_norm=1.0
   - Prevents exploding gradients from high LR
   
8. Added label smoothing: 0.1
   - Prevents overconfident wrong predictions
   - Acts as regularization

==================================================
EXPECTED IMPROVEMENTS:
==================================================

- Should see accuracy improve beyond 10-15% quickly
- Training will be less smooth (more noise is GOOD)
- May see loss jump up occasionally (that's the LR restarts)
- If it still fails, the problem is likely the quantum circuit design

==================================================
IF THIS DOESN'T WORK:
==================================================

Try the "Nuclear Option" - maximum noise:

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-3,              # Even higher!
    weight_decay=5e-4,
    eps=1e-4,             # Even more noise!
    betas=(0.7, 0.9),     # Even less smoothing!
    amsgrad=True
)

# And use stronger gradient clipping:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

==================================================
"""

print(__doc__)
