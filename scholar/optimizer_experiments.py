"""
Different AdamW configurations to add noise and improve exploration.
Try these one at a time to see which works best for your quantum model.
"""

# ============================================================================
# OPTION 1: High Learning Rate + High Epsilon (Maximum Noise)
# ============================================================================
# This adds significant noise through high epsilon and aggressive learning rate
optimizer_option1 = """
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-3,              # 20x higher learning rate
    weight_decay=1e-4,    # 10x higher weight decay
    eps=1e-4,             # 1000x higher epsilon (major gradient noise)
    betas=(0.8, 0.95),    # Lower beta1 for less momentum smoothing
    amsgrad=True          # AMSGrad variant for better handling of noisy gradients
)
"""

# ============================================================================
# OPTION 2: Moderate Noise + Learning Rate Warmup
# ============================================================================
# More conservative approach with warmup schedule
optimizer_option2 = """
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=3e-4,              # 6x higher learning rate
    weight_decay=5e-5,    # 5x higher weight decay
    eps=1e-5,             # 100x higher epsilon
    betas=(0.85, 0.95),   # Moderate momentum reduction
    amsgrad=True
)

# Add learning rate scheduler with warmup
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=20,               # Restart every 20 epochs
    T_mult=2,             # Double the period after each restart
    eta_min=1e-6          # Minimum learning rate
)

# In training loop, add after optimizer.step():
# scheduler.step()
"""

# ============================================================================
# OPTION 3: Cyclical Learning Rate (Noise through LR oscillation)
# ============================================================================
# Uses cyclical LR to add implicit noise and escape local minima
optimizer_option3 = """
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-4,              # Base learning rate
    weight_decay=1e-4,    
    eps=1e-5,             
    betas=(0.9, 0.95)
)

# Cyclical learning rate: oscillates between min and max
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=1e-5,         # Minimum LR
    max_lr=1e-3,          # Maximum LR (adds noise at peaks)
    step_size_up=10,      # Steps to go from min to max
    mode='triangular2',   # Halves max_lr after each cycle
    cycle_momentum=True,  # Also cycle momentum for extra noise
    base_momentum=0.8,
    max_momentum=0.95
)

# In training loop, add after optimizer.step():
# scheduler.step()
"""

# ============================================================================
# OPTION 4: Stochastic Weight Averaging (SWA) - Implicit Noise
# ============================================================================
# Averages multiple points in weight space for better generalization
optimizer_option4 = """
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=5e-4,
    weight_decay=1e-4,
    eps=1e-5,
    betas=(0.9, 0.95)
)

# Use SWA wrapper
swa_model = torch.optim.swa_utils.AveragedModel(model)
swa_scheduler = torch.optim.swa_utils.SWALR(
    optimizer, 
    swa_lr=1e-4,          # SWA learning rate
    anneal_epochs=10
)

# In training loop, after epoch 100:
# if epoch > 100:
#     swa_model.update_parameters(model)
#     swa_scheduler.step()
# else:
#     # Use regular scheduler if desired
#     pass
"""

# ============================================================================
# OPTION 5: Aggressive Exploration (Maximum Settings)
# ============================================================================
# Nuclear option - maximum noise and exploration
optimizer_option5 = """
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=5e-3,              # Very high learning rate
    weight_decay=1e-3,    # Very high weight decay
    eps=1e-3,             # Extremely high epsilon (massive noise)
    betas=(0.7, 0.9),     # Low momentum for maximum responsiveness
    amsgrad=True
)

# Add gradient clipping to prevent explosions with high LR
# In training loop, add after loss.backward():
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
"""

# ============================================================================
# OPTION 6: Lookahead Optimizer (Synthetic - requires installation)
# ============================================================================
# Combines fast and slow weights for better exploration
optimizer_option6 = """
# First install: pip install torch-optimizer
import torch_optimizer as optim

base_optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
    eps=1e-5
)

optimizer = optim.Lookahead(
    base_optimizer,
    k=5,                  # Update slow weights every 5 steps
    alpha=0.5             # Interpolation parameter
)
"""

# ============================================================================
# RECOMMENDED APPROACH: Try in this order
# ============================================================================
print("""
RECOMMENDED ORDER TO TRY:

1. Start with OPTION 2 (Moderate Noise + Warmup)
   - Safest approach with controlled noise addition
   
2. If still stuck, try OPTION 3 (Cyclical LR)
   - Oscillating LR helps escape local minima
   
3. If desperate, try OPTION 1 (High Noise)
   - More aggressive but might destabilize training
   
4. If nothing works, try OPTION 5 (Nuclear Option)
   - Maximum exploration but requires gradient clipping

ADDITIONAL TIPS:
- Your 8-9% accuracy is barely better than random (7.14% for 14 classes)
- The quantum circuit might be the bottleneck, not just the optimizer
- Consider:
  * Increasing n_layers beyond 6
  * Using different quantum gates
  * Simplifying the classical encoder
  * Adding dropout for regularization
  * Using label smoothing in the loss function
""")
